"""Export masked native targets and separate object/gripper reasoning mappings.

Default refuses missing episodes. Full dense paths remain in each targets.npz;
five-point CoT records are views over those paths, not their replacement.
"""
import argparse,csv,json
from contextlib import ExitStack
from pathlib import Path
import numpy as np


def write_row(stream, name, t, prompt, answer, subtask):
    stream.write(json.dumps({'trajectory_name':name,'start_frame':t,'end_frame':t,'subtask_id':subtask,
        'conversations':[{'from':'human','value':prompt},{'from':'gpt','value':answer}]},separators=(',',':'))+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');p.add_argument('--include-unreviewed-boundaries',action='store_true');a=p.parse_args()
    expected=list(csv.DictReader(a.manifest.open()))
    missing=[int(r['episode_index']) for r in expected if not (a.labels/f"episode_{int(r['episode_index']):06d}"/'COMPLETE.json').exists()]
    if missing and not a.allow_partial:raise ValueError(f'{len(missing)} episodes missing; use audit, not zeros')
    a.output.mkdir(parents=True,exist_ok=True)
    counts={'frames':0,'object_rows':0,'gripper_rows':0,'z_rows':0,'object_full_rows':0,'gripper_full_rows':0,'unreviewed_frames':0,'inferred_boundary_frames':0,'missing_episodes':missing}
    with ExitStack() as stack:
        obj=stack.enter_context((a.output/'object_trace.jsonl').open('w'))
        grip=stack.enter_context((a.output/'gripper_trace.jsonl').open('w'))
        objfull=stack.enter_context((a.output/'object_trace_full_subtask.jsonl').open('w'))
        gripfull=stack.enter_context((a.output/'gripper_trace_full_subtask.jsonl').open('w'))
        subtasks_file=stack.enter_context((a.output/'subtasks.jsonl').open('w'))
        z=stack.enter_context((a.output/'shared_z_targets.jsonl').open('w'))
        for row in expected:
            eid=int(row['episode_index']);root=a.labels/f'episode_{eid:06d}'
            if not (root/'COMPLETE.json').exists():continue
            meta=json.loads((root/'COMPLETE.json').read_text());bundle=np.load(root/'targets.npz');data={k:bundle[k] for k in bundle.files};bundle.close()
            # Same logical episode convention used by LeRobot v3; chunk=episode//1000.
            name=f'robocasa365_target_atomic/{eid//1000}/{eid%1000}'
            for sub in meta['subtasks']:
                first=sub['start'];j=sub['entity_index']
                record=dict(sub,trajectory_name=name,episode_index=eid,cameras=meta['cameras'],dense_source=str(root/'targets.npz'),
                    gripper_five_points_full_2d=data['gripper_trace_remaining_2d'][first].tolist(),
                    gripper_five_points_valid=data['gripper_trace_valid'][first].tolist(),
                    object_five_points_full_2d=data['object_trace_remaining_2d'][first].tolist() if j>=0 else None,
                    object_five_points_valid=data['object_trace_valid'][first].tolist())
                subtasks_file.write(json.dumps(record,separators=(',',':'))+'\n')
            for t in range(meta['n_frames']):
                counts['frames']+=1;sid=int(data['subtask_id'][t]);sub=meta['subtasks'][sid];j=sub['entity_index']
                if sub['boundary_needs_review']:
                    counts['inferred_boundary_frames']+=1
                if sub['boundary_needs_review'] and not a.include_unreviewed_boundaries:
                    counts['unreviewed_frames']+=1;continue
                for key,stream,counter,subject in [('object',obj,'object_rows','object'),('gripper',grip,'gripper_rows','gripper')]:
                    if data[key+'_trace_valid'][t,0]:
                        points=np.rint(data[key+'_trace_remaining_2d'][t,0]*1000).astype(int).tolist()
                        answer='<|trace|>'+json.dumps({'trace_2d':points},separators=(',',':'))+'<|/trace|>'
                        prompt=f'Your task is {{instruction}}. Generate the 2D trajectory the {subject} should follow to complete the current subtask. Output exactly 5 points.'
                        write_row(stream,name,t,prompt,answer,sid);counts[counter]+=1
                for key,stream,counter,subject in [('object',objfull,'object_full_rows','object'),('gripper',gripfull,'gripper_full_rows','gripper')]:
                    first=sub['start']
                    if data[key+'_trace_valid'][first,0]:
                        points=np.rint(data[key+'_trace_remaining_2d'][first,0]*1000).astype(int).tolist()
                        answer='<|trace|>'+json.dumps({'trace_2d':points},separators=(',',':'))+'<|/trace|>'
                        prompt=f'Your task is {{instruction}}. Generate the full 2D trajectory of the {subject} for the current subtask, from its beginning to its end. Output exactly 5 points.'
                        write_row(stream,name,t,prompt,answer,sid);counts[counter]+=1
                targets={'trajectory3d':data['trajectory3d_chunk'][t].reshape(-1).tolist(),'phase':int(data['phase'][t])}
                masks={'trajectory3d':bool(data['trajectory3d_chunk_valid'][t]),'phase':True,'target_point':bool(data['target_point_valid'][t,0]),'object_box':bool(j>=0 and data['object_visible'][t,j,0]),'ground_relation':bool(data['ground_relation_valid'][t,0]),'ground_visibility':data['ground_visibility_valid'][t,0].tolist()}
                targets['target_point']=data['target_point'][t,0].tolist()
                targets['object_box']=data['object_box'][t,j,0].tolist() if j>=0 else [0.]*4
                targets['ground_relation']=data['ground_relation'][t,0].tolist();targets['ground_visibility']=data['ground_visibility'][t,0].tolist()
                z.write(json.dumps({'trajectory_name':name,'frame_index':t,'subtask_id':sid,'boundary_source':sub['boundary_source'],'boundary_needs_review':sub['boundary_needs_review'],'targets':targets,'valid':masks,'future_frame_index':int(data['future_frame_index'][t]),'future_frame_valid':bool(data['future_frame_valid'][t]),'dense_source':str(root/'targets.npz')},separators=(',',':'))+'\n');counts['z_rows']+=1
    counts['training_adapter_status']='Native structured target file requires explicit validity-mask loader integration; do not use LIBERO tag-presence visibility inference.'
    (a.output/'EXPORT_AUDIT.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
