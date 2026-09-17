"""Audit entity selection on every saved XML without loading meshes or rendering."""
import csv,gzip,json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
from geometry import select_bodies

root=Path('/e/scratch/m3/blank4/rc365_supervision')
rows=list(csv.DictReader(open('/e/project1/m3/blank4/code/starVLA/results_collected/robocasa_supervision/episode_sources.csv')))
report=[]
for row in rows:
    eid=int(row['episode_index']);sid=int(row['source_episode_index']);task=row['source_prefix'].split('/')[2]
    p=root/'sources'/row['source_prefix']/'lerobot/extras'/f'episode_{sid:06d}'
    try:
        tree=ET.fromstring(gzip.decompress((p/'model.xml.gz').read_bytes()))
        bs=[None]+list(tree.iter('body'));names=['world']+[b.get('name','') for b in bs[1:]]
        adr=[-1];num=[0];qadr=[];q=0
        for b in bs[1:]:
            js=[c for c in b if c.tag in ('joint','freejoint')]
            adr.append(len(qadr));num.append(len(js))
            for j in js:
                qadr.append(q);q+=7 if j.tag=='freejoint' or j.get('type')=='free' else 4 if j.get('type')=='ball' else 1
        geom_names=[];geom_bodies=[]
        for i,b in enumerate(bs[1:],1):
            for g in b.findall('geom'):geom_names.append(g.get('name',''));geom_bodies.append(i)
        model=SimpleNamespace(nbody=len(names),body=lambda b:SimpleNamespace(name=names[b]),body_jntadr=np.array(adr),body_jntnum=np.array(num),jnt_qposadr=np.array(qadr),ngeom=len(geom_names),geom=lambda g:SimpleNamespace(name=geom_names[g]),geom_bodyid=np.array(geom_bodies))
        meta=json.loads((p/'ep_meta.json').read_text());states=np.load(p/'states.npz')['states'];chosen=select_bodies(model,states,meta,task)
        result={'episode_index':eid,'task':task,'entities':[names[b] for b in chosen]}
    except Exception as e:result={'episode_index':eid,'task':task,'error':str(e)}
    report.append(result)
    if len(report)%1000==0:print(len(report),'failures',sum('error'in x for x in report),flush=True)
(root/'entity_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print('TOTAL',len(report),'FAIL',sum('error'in x for x in report),flush=True)
for r in report:
    if 'error'in r:print(json.dumps(r),flush=True)
