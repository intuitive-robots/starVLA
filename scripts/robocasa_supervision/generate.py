"""Generate dense per-subtask traces and shared-z targets from RoboCasa replay.

All coordinates retain provenance and validity. Dense paths are never chunked;
only the separately named trajectory3d_chunk target uses horizon 16.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import mujoco
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from geometry import CAMERAS, load_model, set_state, select_bodies, representative_point, project, camera_delta_cm

VERSION = 'rc365-subtask-traces-v1'


def arc_sample(points, n=5):
    points = np.asarray(points)
    if len(points) == 0 or not np.isfinite(points).all():
        return np.full((n, points.shape[-1]), np.nan)
    cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=-1))]
    if cumulative[-1] < 1e-10:
        return np.repeat(points[:1], n, axis=0)
    unique, ids = np.unique(cumulative, return_index=True)
    return np.stack([np.interp(np.linspace(0, unique[-1], n), unique, points[ids, i]) for i in range(points.shape[-1])], axis=-1)


def segment_subtasks(model, states, bodies, object_world, gripper_world):
    n = len(states)
    if len(bodies) <= 1:
        return [{'start': 0, 'end': n, 'entity_index': 0 if bodies else -1,
                 'boundary_source': 'single_entity_atomic_episode', 'boundary_needs_review': False}]
    # Derive interaction centers from joint motion. Split at nearest-entity
    # hand transitions in the gap between consecutive manipulation intervals.
    intervals = []
    for k, b in enumerate(bodies):
        js = range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b])
        velocity = np.zeros(n)
        for j in js:
            q = states[:, 1 + model.jnt_qposadr[j]]
            velocity += np.r_[0., np.abs(np.diff(q))]
        if velocity.max() == 0:
            raise ValueError('Multiple objects but no reliable interaction ordering')
        active = np.flatnonzero(velocity > max(1e-5, velocity.max() * .08))
        intervals.append((int(active[0]), int(active[-1]), k))
    intervals.sort()
    bounds = [0]
    for previous, current in zip(intervals, intervals[1:]):
        lo, hi = previous[1], current[0]
        if hi <= lo:
            raise ValueError('Overlapping multi-entity motion; needs explicit subtask boundaries')
        candidates = np.arange(lo, hi + 1)
        dist_prev = np.linalg.norm(gripper_world[candidates] - object_world[candidates, previous[2]], axis=-1)
        dist_next = np.linalg.norm(gripper_world[candidates] - object_world[candidates, current[2]], axis=-1)
        transition = candidates[dist_next < dist_prev]
        bounds.append(int(transition[0]) if len(transition) else (lo + hi) // 2)
    bounds.append(n)
    return [{'start': bounds[i], 'end': bounds[i+1], 'entity_index': item[2],
             'boundary_source': 'joint_motion_order_and_gripper_nearest_entity',
             'boundary_needs_review': True} for i, item in enumerate(intervals)]


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def generate(row, args):
    start_time = time.monotonic()
    episode = int(row['episode_index'])
    out = args.output / f'episode_{episode:06d}'
    if (out / 'COMPLETE.json').exists() and not args.overwrite:
        return json.loads((out / 'COMPLETE.json').read_text())
    source = args.root / 'sources' / row['source_prefix'] / 'lerobot'
    sid = int(row['source_episode_index'])
    extra = source / 'extras' / f'episode_{sid:06d}'
    task = row['source_prefix'].split('/')[2]
    meta = json.loads((extra / 'ep_meta.json').read_text())
    states = np.load(extra / 'states.npz')['states']
    source_table = pq.read_table(source / 'data' / f'chunk-{sid // 1000:03d}' / f'episode_{sid:06d}.parquet')
    action = np.array(source_table['action'].to_pylist())
    robot_state = np.array(source_table['observation.state'].to_pylist())
    n = len(states)
    assert n == len(action) == int(row['length'])
    # Full-row source->training alignment, not only an episode-count check.
    files = sorted((args.dataset / 'data').glob('*/*.parquet'))
    local_table = pq.read_table(files, columns=['action', 'observation.state', 'frame_index'], filters=[('episode_index', '=', episode)])
    assert np.array_equal(action, np.array(local_table['action'].to_pylist())), 'Action source mismatch'
    assert np.array_equal(robot_state, np.array(local_table['observation.state'].to_pylist())), 'State source mismatch'
    assert np.array_equal(np.arange(n), np.array(local_table['frame_index'].to_pylist())), 'Frame alignment mismatch'
    model = load_model(extra / 'model.xml.gz', meta)
    model.vis.quality.offsamples = 0  # Integer segmentation IDs must not be antialiased.
    data = mujoco.MjData(model)
    bodies = select_bodies(model, states, meta, task)
    set_state(model, data, states[0])
    reps = [representative_point(model, data, b, task) for b in bodies]
    grip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'gripper0_right_grip_site')
    if grip_id < 0:
        raise ValueError('Missing gripper site')
    camera_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in CAMERAS]
    if min(camera_ids) < 0:
        raise ValueError('Camera missing')
    k = len(bodies)
    arrays = {
        'gripper_world': np.empty((n,3)), 'object_world': np.empty((n,k,3)),
        'camera_pos': np.empty((n,3,3)), 'camera_rot': np.empty((n,3,3,3)),
        'gripper_uv': np.empty((n,3,2)), 'gripper_in_frame': np.empty((n,3),bool),
        'object_uv': np.empty((n,k,3,2)), 'object_in_frame': np.empty((n,k,3),bool),
        'object_box': np.zeros((n,k,3,4)), 'object_visible': np.zeros((n,k,3),bool),
        'gripper_box': np.zeros((n,3,4)), 'gripper_visible': np.zeros((n,3),bool),
    }
    fovy = model.cam_fovy[camera_ids].copy()
    out.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=256, width=256)
    scene_options = mujoco.MjvOption()
    scene_options.geomgroup[0] = 0
    scene_options.geomgroup[1] = 1
    scene_options.sitegroup[:] = 0
    gripper_geoms = np.array([g for g in range(model.ngeom) if model.geom(g).name.startswith('gripper0_right_')])
    snapshots = set(np.linspace(0,n-1,min(5,n)).astype(int)) if args.overlays else set()
    images = {}
    for t, state in enumerate(states):
        set_state(model, data, state)
        arrays['gripper_world'][t] = data.site_xpos[grip_id]
        for j,b in enumerate(bodies):
            arrays['object_world'][t,j] = data.xpos[b] + data.xmat[b].reshape(3,3) @ reps[j][0]
        for c, camera in enumerate(camera_ids):
            pos = data.cam_xpos[camera].copy()
            rot = data.cam_xmat[camera].reshape(3,3).copy()
            arrays['camera_pos'][t,c], arrays['camera_rot'][t,c] = pos, rot
            arrays['gripper_uv'][t,c], arrays['gripper_in_frame'][t,c] = project(arrays['gripper_world'][t], pos, rot, fovy[c])
            if k:
                arrays['object_uv'][t,:,c], arrays['object_in_frame'][t,:,c] = project(arrays['object_world'][t], pos, rot, fovy[c])
            renderer.update_scene(data, camera=camera, scene_option=scene_options)
            if t in snapshots:
                renderer.disable_segmentation_rendering()
                images[t,c] = renderer.render().copy()
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render()
            geom_ids = np.where(segmentation[:,:,1] == int(mujoco.mjtObj.mjOBJ_GEOM), segmentation[:,:,0], -1)
            for j,rep in enumerate(reps):
                if rep[2] == 'static_button_interaction_region':
                    gid = int(rep[1][0])
                    from itertools import product
                    corners = np.array(list(product([-1,1],repeat=3))) * model.geom_size[gid]
                    corners = corners @ data.geom_xmat[gid].reshape(3,3).T + data.geom_xpos[gid]
                    uv, inside = project(corners, pos, rot, fovy[c])
                    center, center_ok = project(data.geom_xpos[gid], pos, rot, fovy[c])
                    if center_ok and np.isfinite(uv).all():
                        renderer.disable_segmentation_rendering()
                        renderer.enable_depth_rendering()
                        depth_image = renderer.render()
                        renderer.disable_depth_rendering()
                        renderer.enable_segmentation_rendering()
                        xpix, ypix = np.clip((center*256).astype(int),0,255)
                        expected_depth = -((data.geom_xpos[gid]-pos)@rot)[2]
                        visible = abs(float(depth_image[ypix,xpix])-expected_depth) < .015
                        arrays['object_box'][t,j,c] = np.clip(np.r_[uv.min(axis=0),uv.max(axis=0)],0,1)
                        arrays['object_visible'][t,j,c] = visible
                    continue
                y,x = np.where(np.isin(geom_ids, rep[1]))
                if len(x):
                    arrays['object_box'][t,j,c] = np.array([x.min(),y.min(),x.max()+1,y.max()+1])/256
                    arrays['object_visible'][t,j,c] = True
            y,x = np.where(np.isin(geom_ids, gripper_geoms))
            if len(x):
                arrays['gripper_box'][t,c] = np.array([x.min(),y.min(),x.max()+1,y.max()+1])/256
                arrays['gripper_visible'][t,c] = True
    renderer.close()
    subtasks = segment_subtasks(model, states, bodies, arrays['object_world'], arrays['gripper_world'])
    arrays.update({
        'subtask_id': np.full(n,-1,np.int16), 'future_frame_index': np.minimum(np.arange(n)+8,n-1),
        'future_frame_valid': np.arange(n)+8<n,
        'phase': np.zeros(n,np.int8),
        'trajectory3d_chunk': np.zeros((n,5,3)),
        'trajectory3d_chunk_valid': np.zeros(n,bool),
        'gripper_trace_remaining_3d': np.zeros((n,5,3)),
        'object_trace_remaining_3d': np.zeros((n,5,3)),
        'gripper_trace_remaining_2d': np.zeros((n,3,5,2)),
        'object_trace_remaining_2d': np.zeros((n,3,5,2)),
        'gripper_trace_valid': np.zeros((n,3),bool), 'object_trace_valid': np.zeros((n,3),bool),
        'target_point': np.zeros((n,3,2)), 'target_point_valid': np.zeros((n,3),bool),
        'ground_relation': np.zeros((n,3,2)), 'ground_relation_valid': np.zeros((n,3),bool),
        'ground_visibility': np.zeros((n,3,2)), 'ground_visibility_valid': np.zeros((n,3,2),bool),
    })
    # Match LIBERO's phase: commanded gripper at current/chunk-end frames.
    closing = action[:,11] > 0
    for sub_id, sub in enumerate(subtasks):
        a,b,j = sub['start'],sub['end'],sub['entity_index']
        assert a < b
        arrays['subtask_id'][a:b] = sub_id
        arrays['future_frame_index'][a:b] = np.minimum(np.arange(a,b)+8,b-1)
        arrays['future_frame_valid'][a:b] = np.arange(a,b)+8<b
        sub.update({'subtask_id':sub_id, 'entity':model.body(bodies[j]).name if j>=0 else None,
                    'object_trace_present':j>=0, 'instruction':meta['lang'],
                    'gripper_full_array_slice':[a,b], 'object_full_array_slice':[a,b,j],
                    'target_definition':'terminal_rigid_object_point_in_current_camera' if j>=0 else 'not_applicable_navigation'})
        for t in range(a,b):
            g3 = arc_sample(arrays['gripper_world'][t:b])
            arrays['gripper_trace_remaining_3d'][t] = camera_delta_cm(g3, arrays['gripper_world'][t], arrays['camera_rot'][t,2])
            if j>=0:
                o3 = arc_sample(arrays['object_world'][t:b,j])
                arrays['object_trace_remaining_3d'][t] = camera_delta_cm(o3, arrays['object_world'][t,j], arrays['camera_rot'][t,2])
            times = np.rint(np.linspace(t,min(t+15,b-1),5)).astype(int)
            arrays['trajectory3d_chunk'][t] = np.rint(camera_delta_cm(arrays['gripper_world'][times], arrays['gripper_world'][t], arrays['camera_rot'][t,2])) / 20
            arrays['trajectory3d_chunk_valid'][t] = times[-1] > t
            arrays['phase'][t] = (2 if closing[t] else 0) if closing[t] == closing[times[-1]] else (1 if closing[times[-1]] else 3)
            for c in range(3):
                g = arc_sample(arrays['gripper_uv'][t:b,c])
                arrays['gripper_trace_remaining_2d'][t,c] = np.nan_to_num(g)
                arrays['gripper_trace_valid'][t,c] = arrays['gripper_in_frame'][t:b,c].all() and np.isfinite(g).all()
                if j<0:
                    continue
                o = arc_sample(arrays['object_uv'][t:b,j,c])
                arrays['object_trace_remaining_2d'][t,c] = np.nan_to_num(o)
                arrays['object_trace_valid'][t,c] = arrays['object_in_frame'][t:b,j,c].all() and np.isfinite(o).all()
                pt, valid = project(arrays['object_world'][b-1,j], arrays['camera_pos'][t,c], arrays['camera_rot'][t,c], fovy[c])
                arrays['target_point'][t,c] = np.nan_to_num(pt)
                arrays['target_point_valid'][t,c] = valid
                visible = arrays['object_visible'][t,j,c]
                box = arrays['object_box'][t,j,c]
                arrays['ground_relation'][t,c] = np.nan_to_num(pt) - (box[:2]+box[2:])/2
                arrays['ground_relation_valid'][t,c] = valid and visible
                arrays['ground_visibility'][t,c] = [valid,visible]
                arrays['ground_visibility_valid'][t,c] = True
    assert (arrays['subtask_id']>=0).all()
    arrays['camera_fovy'] = fovy
    arrays['gripper_world'] = arrays['gripper_world'].astype(np.float32)
    for key in arrays:
        if arrays[key].dtype==np.float64:
            arrays[key] = arrays[key].astype(np.float32)
    np.savez_compressed(out/'targets.npz',**arrays)
    record = {
        'version':VERSION,'episode_index':episode,'source_prefix':row['source_prefix'],'source_episode_index':sid,
        'task':task,'n_frames':n,'cameras':list(CAMERAS),'coordinate_format':'normalized_xy_unclipped; world_meters; full_cam3d_cm; chunk_cam3d_rounded_cm_div20; axes_right_down_forward',
        'point_definition':[r[2] for r in reps], 'entities':[model.body(b).name for b in bodies],
        'subtasks':subtasks,'full_paths':'Dense arrays in targets.npz, sliced by subtask start:end. No chunk truncation.',
        'future_frame_policy':'offset8 within subtask; valid flag excludes padded boundary frames',
        'phase_definition':'LIBERO convention: commanded gripper at current/chunk-end; keep_open0,close1,keep_closed2,open3',
        'visibility_definition':'object visible from exact geom segmentation, except static microwave button uses projected interaction-region box and depth check; target valid from projected terminal point in image, not target occlusion',
        'alignment':'all actions, robot states and frame indices equal consolidated training copy',
        'model_sha256':hashlib.sha256((extra/'model.xml.gz').read_bytes()).hexdigest(),
        'target_fraction':float(arrays['target_point_valid'][:,0].mean()),
        'object_visible_fraction':float(arrays['object_visible'][:,:,0].mean()) if k else None,
        'object_trace_fraction':float(arrays['object_trace_valid'][:,0].mean()),
        'gripper_trace_fraction':float(arrays['gripper_trace_valid'][:,0].mean()),
        'elapsed_seconds':round(time.monotonic()-start_time,3),
    }
    for (t,c),rgb in images.items():
        im=Image.fromarray(rgb)
        im.save(out/f'replay_{t:04d}_{c}.png')
        draw=ImageDraw.Draw(im)
        sub=subtasks[int(arrays['subtask_id'][t])];j=sub['entity_index'];end=sub['end']
        for key,col in [('gripper_uv','cyan'),('object_uv','orange')]:
            if key=='object_uv' and j<0:continue
            points=arrays[key][t:end,c] if key=='gripper_uv' else arrays[key][t:end,j,c]
            valid=np.isfinite(points).all(axis=1)&(np.abs(points)<5).all(axis=1)
            if valid.all() and len(points)>1:draw.line([tuple(p*256) for p in points],fill=col,width=2)
        if j>=0 and arrays['object_visible'][t,j,c]:draw.rectangle(tuple(arrays['object_box'][t,j,c]*256),outline='orange',width=2)
        draw.text((3,3),f'{task} e{episode} t{t} s{sub["subtask_id"]}',fill='white')
        im.save(out/f'overlay_{t:04d}_{c}.png')
    write_json(out/'COMPLETE.json',record)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--num-shards',type=int,default=1);p.add_argument('--pilot',action='store_true');p.add_argument('--overlays',action='store_true');p.add_argument('--overwrite',action='store_true');p.add_argument('--episode-ids');a=p.parse_args()
    with a.manifest.open() as f:rows=list(csv.DictReader(f))
    if a.pilot:
        seen=set();rows=[r for r in rows if not (r['source_prefix'] in seen or seen.add(r['source_prefix']))]
    if a.episode_ids:
        ids=set(map(int,a.episode_ids.split(',')));rows=[r for r in rows if int(r['episode_index']) in ids]
    rows=rows[a.shard::a.num_shards];a.output.mkdir(parents=True,exist_ok=True)
    failed=0
    for row in rows:
        try:print(json.dumps(generate(row,a)),flush=True)
        except Exception:
            failed+=1;err={'episode_index':row['episode_index'],'task':row['source_prefix'],'error':traceback.format_exc()}
            print(json.dumps(err),flush=True)
            with (a.output/f'failed_shard_{a.shard}.jsonl').open('a') as f:f.write(json.dumps(err)+'\n')
    write_json(a.output/f'shard_{a.shard}_finished.json',{'episodes':len(rows),'failures':failed})
    raise SystemExit(1 if failed else 0)
if __name__=='__main__':main()
