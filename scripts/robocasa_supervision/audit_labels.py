"""Audit boundaries, camera axes, target validity and episode completeness."""
import argparse,collections,csv,json
from pathlib import Path
import numpy as np
from geometry import camera_delta_cm

p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
rows=list(csv.DictReader(a.manifest.open()));missing=[];errors=[];counts=collections.Counter();tasks={}
for row in rows:
    eid=int(row['episode_index']);root=a.labels/f'episode_{eid:06d}'
    if not (root/'COMPLETE.json').exists():missing.append(eid);continue
    try:
        m=json.loads((root/'COMPLETE.json').read_text());d=np.load(root/'targets.npz');n=int(row['length']);assert m['n_frames']==n
        seen=np.zeros(n,int)
        for sub in m['subtasks']:
            start,end,j=sub['start'],sub['end'],sub['entity_index'];assert 0<=start<end<=n
            seen[start:end]+=1
            counts['subtasks']+=1;counts['subtasks_needing_review']+=sub['boundary_needs_review']
            ts=np.arange(start,end);assert np.all(d['subtask_id'][ts]==sub['subtask_id'])
            np.testing.assert_array_equal(d['future_frame_valid'][ts],ts+8<end)
            assert np.all(d['future_frame_index'][ts]<end)
            for t in (start,(start+end-1)//2,end-1):
                selected=np.rint(np.linspace(t,min(t+15,end-1),5)).astype(int)
                expected=np.rint(camera_delta_cm(d['gripper_world'][selected],d['gripper_world'][t],d['camera_rot'][t,2]))/20
                np.testing.assert_allclose(d['trajectory3d_chunk'][t],expected,atol=.050001)
                np.testing.assert_allclose(d['gripper_trace_remaining_3d'][t,-1],camera_delta_cm(d['gripper_world'][end-1],d['gripper_world'][t],d['camera_rot'][t,2]),atol=1e-3)
                if j>=0:
                    np.testing.assert_allclose(d['object_trace_remaining_2d'][t,:,-1],d['object_uv'][end-1,j],atol=1e-5)
        assert np.all(seen==1),'Subtask gap/overlap'
        counts['episodes']+=1;counts['frames']+=n
        task=tasks.setdefault(m['task'],{'episodes':0,'frames':0,'subtasks':0,'object_trace_valid':[0]*3,'gripper_trace_valid':[0]*3,'target_point_valid':[0]*3})
        task['episodes']+=1;task['frames']+=n;task['subtasks']+=len(m['subtasks'])
        for field in ('object_trace_valid','gripper_trace_valid','target_point_valid'):
            task[field]=[int(x+y) for x,y in zip(task[field],d[field].sum(axis=0))]
    except Exception as e:errors.append({'episode_index':eid,'error':str(e)})
report={'counts':dict(counts),'missing_episodes':missing,'errors':errors,'tasks':tasks}
(a.labels/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'counts':dict(counts),'missing':len(missing),'errors':errors},indent=2))
raise SystemExit(bool(errors or (missing and not a.allow_partial)))
