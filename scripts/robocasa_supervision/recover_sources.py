"""Recover exact source extras and low-dimensional data; never extract archive paths blindly."""
import argparse, concurrent.futures, csv, json, pathlib, subprocess, tarfile, time

def recover(item):
    prefix,url,root=item
    dest=root/'sources'/prefix
    done=dest/'RECOVERED.json'
    if done.exists():return json.loads(done.read_text())
    arc=root/'source_archives'/(prefix.replace('/','_')+'.tar')
    arc.parent.mkdir(parents=True,exist_ok=True)
    legacy=root/'source_archives'/(prefix.split('/')[-2]+'_'+prefix.split('/')[-1]+'.tar')
    if legacy.exists() and not arc.exists():arc.symlink_to(legacy)
    u=url.replace('/s/','/shared/static/')+'.tar'
    subprocess.run(['curl','-L','--fail','--retry','3','--max-time','600','-C','-',u,'-o',str(arc),'--silent','--show-error'],check=True)
    count=0
    with tarfile.open(arc,'r:') as tf:
        for m in tf:
            p=pathlib.PurePosixPath(m.name)
            if not m.isfile() or not p.parts or p.parts[0]!='lerobot':continue
            if len(p.parts)<2 or p.parts[1] not in ('extras','data','meta'):continue
            if '..' in p.parts:raise ValueError(m.name)
            out=dest/pathlib.Path(*p.parts)
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_bytes(tf.extractfile(m).read());count+=1
    report={'source_prefix':prefix,'archive_url':url,'archive_bytes':arc.stat().st_size,'extracted_files':count,'root':str(dest/'lerobot')}
    done.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=pathlib.Path);p.add_argument('--root',required=True,type=pathlib.Path);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    with a.manifest.open() as f:src={r['source_prefix']:r['archive_url'] for r in csv.DictReader(f)}
    outcomes=[]
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        fs={pool.submit(recover,(k,v,a.root)):k for k,v in src.items()}
        for f in concurrent.futures.as_completed(fs):
            try:outcomes.append(f.result())
            except Exception as e:outcomes.append({'source_prefix':fs[f],'error':str(e)});print(json.dumps(outcomes[-1]),flush=True)
    (a.root/'recovery_report.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    if any('error' in x for x in outcomes):raise SystemExit(1)
if __name__=='__main__':main()
