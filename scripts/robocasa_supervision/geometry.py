"""Exact saved-state geometry. No physics stepping or policy actions are used."""
import gzip
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

CAMERAS = ('robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand')

def load_model(path, meta=None):
    root = ET.fromstring(gzip.decompress(Path(path).read_bytes()))
    for el in root.iter():
        old = el.get('file')
        if not old:
            continue
        for package, base in [('robosuite', '/app/robosuite/robosuite'), ('robocasa', '/app/robocasa365/robocasa')]:
            mark = '/' + package + '/'
            if mark in old:
                new = base + '/' + old.rsplit(mark, 1)[1]
                if not Path(new).exists():
                    raise FileNotFoundError(new)
                el.set('file', new)
                break
    if meta:
        bodies = {b.get('name'): b for b in root.iter('body')}
        cameras = {c.get('name'): c for c in root.iter('camera')}
        for name, cfg in meta['cam_configs'].items():
            if name not in CAMERAS:
                continue
            camera = cameras.get(name)
            if camera is None:
                camera = ET.SubElement(bodies[cfg['parent_body']], 'camera', {'name':name, 'mode':'fixed'})
            for key in ('pos','quat'):
                camera.set(key,' '.join(map(str,cfg[key])))
            for key,value in cfg.get('camera_attribs',{}).items():
                camera.set(key,str(value))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))

def set_state(model, data, state):
    assert len(state) == 1 + model.nq + model.nv + model.na
    data.time = state[0]
    data.qpos[:] = state[1:1 + model.nq]
    data.qvel[:] = state[1 + model.nq:1 + model.nq + model.nv]
    if model.na:
        data.act[:] = state[-model.na:]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)

def descendants(model, body):
    ids = {int(body)}
    for b in range(int(body) + 1, model.nbody):
        if int(model.body_parentid[b]) in ids:
            ids.add(b)
    return sorted(ids)

def project(world, camera_pos, camera_rot, fovy):
    xyz = (np.asarray(world) - camera_pos) @ camera_rot
    depth = -xyz[..., 2]
    focal = .5 / np.tan(np.deg2rad(fovy) / 2)
    safe = np.where(np.abs(depth) > 1e-9, depth, np.nan)
    uv = np.stack((.5 + focal * xyz[..., 0] / safe, .5 - focal * xyz[..., 1] / safe), axis=-1)
    valid = (depth > 0) & np.isfinite(uv).all(axis=-1) & (uv >= 0).all(axis=-1) & (uv <= 1).all(axis=-1)
    return uv, valid

def fixture_names(meta, classes, ref=None):
    refs = meta.get('fixture_refs', {})
    if ref and ref in refs:
        value = refs[ref]
        if isinstance(value, str):
            return [value]
    return [name for name, entry in meta['fixtures'].items() if entry.get('cls') in classes]

def select_bodies(model, states, meta, task):
    """Task semantics select entities; recorded joint motion disambiguates doors.

    Returns separate moving bodies, never a concatenated multi-object track.
    Navigation has no manipulated object and is represented explicitly as such.
    """
    names = [model.body(b).name for b in range(model.nbody)]
    if task == 'NavigateKitchen':
        return []
    if task.startswith('PickPlace') or task == 'CoffeeSetupMug':
        if 'obj_main' not in names:
            raise ValueError('Expected manipulated obj_main; refusing distractor fallback')
        return [names.index('obj_main')]
    if task == 'TurnOnMicrowave':
        prefixes = fixture_names(meta, {'Microwave'})
        targets = [g for g in range(model.ngeom) if any(model.geom(g).name == prefix + '_start_button' for prefix in prefixes)]
        if len(targets) != 1:
            raise ValueError('Ambiguous microwave start button')
        return [int(model.geom_bodyid[targets[0]])]
    if task == 'CloseBlenderLid':
        prefixes = fixture_names(meta, {'BlenderLid'})
        candidates = [b for b, n in enumerate(names) if n in [p + '_main' for p in prefixes]]
    else:
        specs = {
            'CloseFridge': ({'FridgeFrenchDoor','FridgeSideBySide','FridgeBottomFreezer'}, 'fxtr', ('door',)),
            'OpenCabinet': ({'HingeCabinet','SingleCabinet'}, 'fxtr', ('door',)),
            'OpenDrawer': ({'Drawer'}, 'drawer', ('inner_box',)),
            'CloseToasterOvenDoor': ({'ToasterOven'}, 'toaster_oven', ('door',)),
            'OpenStandMixerHead': ({'StandMixer'}, None, ('head',)),
            'SlideDishwasherRack': ({'Dishwasher'}, 'dishwasher', ('rack',)),
            'TurnOnElectricKettle': ({'ElectricKettle'}, 'electric_kettle', ('lever','switch','button')),
            'TurnOffStove': ({'Stove','Stovetop'}, None, ('knob_' + meta.get('task_refs',{}).get('knob','MISSING'),)),
            'TurnOnMicrowave': ({'Microwave'}, None, ('start_button',)),
            'TurnOnSinkFaucet': ({'Sink'}, None, ('handle',)),
        }
        classes, ref, parts = specs[task]
        prefixes = fixture_names(meta, classes, ref)
        candidates = [b for b, n in enumerate(names) if any(n.startswith(p + '_') and any(x in n[len(p)+1:] for x in parts) for p in prefixes)]
        # Articulated root bodies, excluding fixed nested meshes and sub-components.
        moving = [b for b in candidates if model.body_jntnum[b] > 0]
        if moving:
            candidates = moving
    if not candidates:
        raise ValueError(f'No semantic body for {task}; prefixes={prefixes}')
    if len(candidates) > 1:
        ranges = []
        for b in candidates:
            js = range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b])
            variation = max([float(np.ptp(states[:, 1 + model.jnt_qposadr[j]])) for j in js] or [0.])
            ranges.append(variation)
        # Keep each deliberately moving member (e.g. left and right doors).
        candidates = [b for b, v in zip(candidates, ranges) if v > max(1e-4, max(ranges) * .03)]
    if not candidates:
        raise ValueError('Ambiguous/no moving task entity')
    return candidates

def representative_point(model, data, body, task=None):
    """A rigid point at a visible handle, otherwise visual-geometry centroid."""
    if task == 'TurnOnMicrowave':
        gids = [g for g in range(model.ngeom) if model.geom(g).name.endswith('_start_button') and model.geom_bodyid[g] == body]
        if len(gids) != 1:
            raise ValueError('Ambiguous start button within selected microwave body')
        gid = gids[0]
        local = (data.geom_xpos[gid] - data.xpos[body]) @ data.xmat[body].reshape(3,3)
        return local, np.array([gid]), 'static_button_interaction_region'
    gids = np.flatnonzero(np.isin(model.geom_bodyid, descendants(model, body)))
    visual = [int(g) for g in gids if model.geom_group[g] == 1]
    preferred = [g for g in visual if 'handle' in model.geom(g).name.lower()]
    chosen = preferred or visual or list(gids)
    if not chosen:
        raise ValueError('No geometry for selected body')
    # Equal weighting avoids letting collision geometry density choose the point.
    world = np.mean(data.geom_xpos[chosen], axis=0)
    local = (world - data.xpos[body]) @ data.xmat[body].reshape(3,3)
    return local, gids, ('handle_geom_center' if preferred else 'visual_geom_center')


def camera_delta_cm(points, origin, camera_rot):
    """Displayed unrotated RoboCasa camera axes: right, down, forward."""
    return ((np.asarray(points)-origin) @ camera_rot) * np.array([1.,-1.,-1.]) * 100
