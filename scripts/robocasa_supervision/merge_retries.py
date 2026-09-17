"""Merge only completed retry outputs, preserving failed-attempt directories."""
import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--labels',required=True,type=Path);p.add_argument('--retries',required=True,type=Path);p.add_argument('--archive',required=True,type=Path);a=p.parse_args()
a.archive.mkdir(parents=True,exist_ok=True)
merged=[]
for done in sorted(a.retries.glob('episode_*/COMPLETE.json')):
    source=done.parent;target=a.labels/source.name
    if (target/'COMPLETE.json').exists():
        original=json.loads((target/'COMPLETE.json').read_text());retry=json.loads(done.read_text())
        if original['code_sha256']!=retry['code_sha256']:raise ValueError(f'Refusing to overwrite completed {target}')
        continue
    staged=a.labels/(source.name+'.merging')
    shutil.copytree(source,staged)
    if target.exists():target.rename(a.archive/source.name)
    staged.rename(target);merged.append(source.name)
(a.labels/'RETRY_MERGE.json').write_text(json.dumps({'source':str(a.retries),'episodes':merged},indent=2)+'\n')
print('merged',len(merged))
