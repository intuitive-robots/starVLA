import csv,json,math,statistics as st,collections,re,hashlib,datetime,sys
from pathlib import Path
A=Path('/e/project1/m3/blank4/code/starVLA');B=Path('/e/project1/m3/blank4/code/starVLA-upstream-merge');D=Path('/e/project1/m3/blank4/code/train_downstream');O=A/'results_collected';O.mkdir(exist_ok=True)
rows=list(csv.DictReader((O/'results_master.csv').open()))
raw={p:json.loads(Path(p).read_text()) for p in sorted({x['source_path'] for x in rows})}
def stat(s,n):
 p=s/n if n else float('nan');return {'s':s,'n':n,'p':p,'se':math.sqrt(p*(1-p)/n) if n else float('nan')}
def pool(rs):return stat(sum(int(x['successes']) for x in rs),sum(int(x['n_episodes']) for x in rs))
def fmt(v):return f"{v['p']*100:.3f} ± {v['se']*100:.3f}; {v['s']}/{v['n']}"
def link(p,label=None):return f'[{label or Path(p).name}]({p})'
def table(headers,rs):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]+['| '+' | '.join(map(str,r))+' |' for r in rs])+'\n'
def rstat(d):o=d.get('overall',d);return stat(o['success_count'],o['total_count'])
# Evaluation jobs update root JSONs while the collector and this audit run. If a source changed
# between the two snapshots, exclude that whole source from this report instead of combining CSV
# counts from one moment with perturbation data from another. The next collection will pick it up.
def row_raw_stat(x,b):
 if x['benchmark']=='robocasa365':return stat(sum(b['successes']),len(b['successes']))
 return rstat(b if 'total_count' in b.get('overall',{}) and x['suite_or_task']!='overall' else b[x['suite_or_task']])
changed_sources=set()
for x in rows:
 try:v=row_raw_stat(x,raw[x['source_path']])
 except (KeyError,TypeError):changed_sources.add(x['source_path']);continue
 if v['s']!=int(x['successes']) or v['n']!=int(x['n_episodes']):changed_sources.add(x['source_path'])
if changed_sources:
 print(f"! excluded {len(changed_sources)} source(s) modified after collection",file=sys.stderr)
 rows=[x for x in rows if x['source_path'] not in changed_sources]
 raw={p:b for p,b in raw.items() if p not in changed_sources}
(O/'raw_results_snapshot.json').write_text(json.dumps(raw,sort_keys=True))
manifest=[{'path':p,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest(),'mtime':datetime.datetime.fromtimestamp(Path(p).stat().st_mtime).isoformat()} for p in raw]
(O/'source_manifest.json').write_text(json.dumps(manifest,indent=1))
# Independent audit against the frozen raw payload, including all nested suites and RC booleans.
issues=[(p,'','source changed after collection; excluded from this snapshot') for p in sorted(changed_sources)]
for x in rows:
 b=raw[x['source_path']]
 v=row_raw_stat(x,b)
 assert v['s']==int(x['successes']) and v['n']==int(x['n_episodes']),x
 if v['n']:assert abs(v['p']-float(x['success_rate']))<1e-8,x
for p,b in raw.items():
 def walk(b,key=''):
  if not isinstance(b,dict):return
  if {'total_count','success_count','success_rate'}<=b.keys() and b['total_count']:
   if abs(b['success_rate']-b['success_count']/b['total_count'])>1e-8:issues.append((p,key,'stored rate mismatch'))
  for k,v in b.items():
   if isinstance(v,dict):walk(v,key+'/'+k)
 walk(b)
 if 'successes' in b and abs(b['success_rate']-sum(b['successes'])/len(b['successes']))>1e-8:issues.append((p,'','stored rate mismatch'))
# Root artifacts only, no suite-local copies counted a second time.
g=collections.defaultdict(list)
for x in rows:
 if x['tree']=='starVLA' and Path(x['source_path']).parent.parent.name=='results':g[(x['run'],x['benchmark'])].append(x)
roots={k:dict(pool(v),run=k[0],benchmark=k[1],path=v[0]['source_path'],arm=v[0]['arm'],seed=v[0]['seed'],suites={x['suite_or_task']:stat(int(x['successes']),int(x['n_episodes'])) for x in v},full=all(x['is_full_protocol']=='True' for x in v)) for k,v in g.items()}
plus={k:v for k,v in roots.items() if 'plus' in k[1]}
canonical={}
for run in {k[0] for k in plus}:
 candidates=[v for (r,b),v in plus.items() if r==run and v['n']>=4000 and v['full']]
 if not candidates:continue
 # Exact 1000 x 4, explicit protocol directory, then lexical path. Never score selected.
 candidates.sort(key=lambda v:(not(v['n']==4000 and all(u['n']==1000 for u in v['suites'].values())),not('4k-exact' in v['benchmark']),v['benchmark']))
 canonical[run]=candidates[0]
causal=[canonical[f'ervla_pi_causal_actiononly_pifix_nolatent_nodrop_2gpu_s{s}'] for s in (42,43,44)]
z=[canonical[f'ervla_zonly_pi_sharedz_ground_temporal_v5_s{s}'] for s in (42,43)]
families={'causal':causal,'shared-z v5':z}
for label,base,seeds in [('shared-z GR00T','ervla_zonly_gr00t_sharedz_ground_temporal_v5_s',[42,43]),('v5 GR00T','ervla_v5_gr00t_actiononly_s',[42,43]),('v5 PI','ervla_v5_pi_actiononly_pifix_s',[42,43]),('old bidir PI','ervla_pi_bidir_actiononly_pifix_nolatent_nodrop_2gpu_s',[42,43,44]),('v5 PI augmentation','ervla_v5_pi_actiononly_pifix_aug_s',[42,43]),('v5aux PI','ervla_v5aux_pi_actiononly_pifix_s',[42,43]),('shared-z augmentation','ervla_zonly_pi_sharedz_ground_temporal_aug_s',[42,43]),('cam3d masked','ervla_k_pi_cam3d_cot05_masked',[None,43])]:
 families[label]=[canonical[base+(str(s) if base.endswith('_s') else '_seed'+str(s))] if s else canonical[base] for s in seeds]
def fstats(vs):
 v=stat(sum(x['s'] for x in vs),sum(x['n'] for x in vs));v.update(mean=st.mean(x['p'] for x in vs),sd=st.stdev(x['p'] for x in vs) if len(vs)>1 else None,seeds=len(vs));return v
fs={k:fstats(v) for k,v in families.items()}
# Per category, suite-balanced sample IDs should match; counts test is necessary but not sufficient.
pert={}
for label,vs in [('causal (3 seeds)',causal),('shared-z v5 (2 seeds)',z),('shared-z zsup (seed42)',[canonical['ervla_zsup_pi_sharedz_ground_temporal']]),('encoder W (seed42)',[canonical['ervla_w_pi_encoder_actiononly']]),('encoder MLM (seed42)',[canonical['ervla_mlm_pi_encoder_control']])]:
 cs=collections.defaultdict(lambda:[0,0])
 for v in vs:
  for suite,ob in raw[v['path']].items():
   for cat,d in ob.items():
    if cat=='overall':continue
    cs[cat][0]+=d['success_count'];cs[cat][1]+=d['total_count']
 pert[label]={k:stat(*v) for k,v in cs.items()}
# RoboCasa settings-specific groups and 17-task matrix.
rc=collections.defaultdict(list)
for x in rows:
 if x['benchmark']=='robocasa365':rc[(x['run'],x['tag'],x['result_dir'])].append(x)
rcp={k:pool(v) for k,v in rc.items()}
rbench={}
for seed in [4,42]:
 for arm in ['causal','v5']:
  vs=[x for x in rows if x['run']==f'ervla_robocasa365_pi_{arm}_s{seed}' and x['tag']==f's{seed}_50k']
  rbench[(arm,seed)]={'pool':pool(vs),'tasks':{x['suite_or_task']:dict(stat(int(x['successes']),int(x['n_episodes'])),path=x['source_path']) for x in vs}}
# Check all root versus suite-local copies; snapshots remain separate regardless of equality.
dups=[]
for (run,b),v in roots.items():
 for suite,u in v['suites'].items():
  p=str(Path(v['path']).parent/suite/'overall_results.json')
  if p in raw:
   ob=raw[p];w=rstat(ob) if 'overall' in ob else rstat(ob[suite]) if suite in ob else None
   if w and (w['s'],w['n'])!=(u['s'],u['n']):dups.append({'run':run,'benchmark':b,'suite':suite,'root':u,'child':w,'source':p})
# Count missing/error sentinels; absence is evidence only with manifest matching.
fail=[{'path':str(p),'lines':p.read_text().splitlines()} for tree in [A,B] for p in (tree/'playground/Checkpoints').glob('*/results/*/failed_shards_*.txt') if p.read_text().strip()]
failunits=list(Path('/e/scratch/m3/blank4/rc365_smoke').rglob('failed_units*'))
# Paired small/full transitions, one smallest four-suite old sample per run, exact full canonical.
trans=[]
for run,full in canonical.items():
 small=[v for (r,b),v in plus.items() if r==run and 0<v['n']<4000 and len(v['suites'])==4]
 for old in sorted(small,key=lambda x:(x['n'],x['benchmark'])):
  trans.append({'run':run,'old':old,'new':full,'delta':full['p']-old['p']})
# Main report.
report=['# ERVLA results audit — 2026-09-17','',
'Computed from raw JSON snapshots, not from the status documents. Scores below are percent ± one binomial standard error, followed by successes/episodes. SE = sqrt(p(1−p)/n); it ignores within-task correlation and is not training-seed uncertainty. Reported seed SD uses ddof=1. No missing result is counted as a failure. Zero-success Wald SE is zero algebraically, not evidence of certainty (0/48 has a Wilson 95% upper bound of 7.4%).','',
'**The current evidence supports a shared-latent improvement on the internal LIBERO-plus protocol. It does not establish that bidirectionality alone wins on every benchmark, or a published-SOTA claim.** Plain LIBERO is small-n; RoboCasa has two complete internal 17-task seeds, but only a +2.27 pp pooled gain.','',
'## Audit and canonical rule','',
f'Read {len(raw)} unique raw JSON artifacts, producing {len(rows)} rows; all row counts and rates independently checked. {len(issues)} stored-rate mismatches. Snapshot: [raw_results_snapshot.json](results_collected/raw_results_snapshot.json); source hashes: [source_manifest.json](results_collected/source_manifest.json). Master CSV retains root and suite-local artifacts as different result_dir variants. Never sum those variants.','',
'Canonical LIBERO-plus: require four suites, ≥4,000 episodes and no nonempty failed_shards file at the root; prefer exactly 1,000 per suite, then an explicit 4k-exact directory, then lexical directory order. Never select by success rate or file modification time. This is the user-requested **internal full protocol**; the official benchmark evaluates all perturbation tasks, so 4,000 is not automatically an official leaderboard protocol. Root/suite-local disagreement excludes a result from an unqualified claim until explained. Identical copies are not replications.','',
'`seed` now means training seed from config.full.yaml, falling back to run name. RC JSON seed42 is the environment/server seed even for training seed4. `is_full_protocol` for RC means all declared internal manifest tasks at 48 episodes; it does not mean the official 50-task protocol. A blank checkpoint_step in LIBERO means the raw aggregate does not identify it; do not infer a checkpoint from training max_steps.','',
'## LIBERO-plus seed summaries (canonical full protocol only)','',table(['family','training seeds','mean ± SD (%)','pooled % ± SE; successes/n','raw sources'],[(k,','.join(v['seed'] for v in families[k]),f"{v['mean']*100:.3f} ± {v['sd']*100:.3f}",fmt(v),' '.join(link(x['path'],x['seed']) for x in families[k])) for k,v in fs.items()]),
'## All full-count LIBERO-plus root results, grouped by arm','',
'Every sibling directory is retained. C = canonical by the rule above; alternate rows are sensitivity evidence, never extra seeds. “Full-count only” can still have a failure sentinel or unequal suite weights.','']
for arm in sorted({v['arm'] for v in plus.values() if v['n']>=4000}):
 report += [f'### {arm}','',table(['run','directory','C','% ± SE; successes/n','source'],[(v['run'],v['benchmark'],'yes' if canonical.get(v['run'],{}).get('path')==v['path'] else 'no',fmt(v),link(v['path'])) for v in sorted(plus.values(),key=lambda v:-v['p']) if v['n']>=4000 and v['arm']==arm])]
report += ['## Perturbation breakdown','',table(['category',*pert.keys()],[(cat,*[fmt(p[cat]) for p in pert.values()]) for cat in sorted(next(iter(pert.values())))]),'',
'Sources are the canonical seed files linked above and the W, MLM and zsup rows in the full-count table. Shared-z v5 is the best replicated family; zsup is the best single-seed shared-z canonical score; W is the highest encoder-objective arm but has a different older recipe. Category contrasts below use the replicated shared-z family and W, each against pooled causal.','',
table(['category','shared-z − causal pp ± SE','W − causal pp ± SE'],[(cat,*[f"{100*(pert[k][cat]['p']-pert['causal (3 seeds)'][cat]['p']):+.3f} ± {100*math.hypot(pert[k][cat]['se'],pert['causal (3 seeds)'][cat]['se']):.3f}" for k in ['shared-z v5 (2 seeds)','encoder W (seed42)']]) for cat in sorted(next(iter(pert.values())))]),
'## Plain LIBERO — smoke only under the requested ≥4,000 rule','',
'All available root artifacts are listed separately from LIBERO-plus. Most use 10 episodes per task, not the common published 50. Neither a high point estimate nor pooling seeds repairs that protocol mismatch.','',table(['run','directory','% ± SE; successes/n','source'],[(v['run'],v['benchmark'],fmt(v),link(v['path'])) for v in sorted(roots.values(),key=lambda x:(x['run'],x['benchmark'])) if ('libero' in v['benchmark'] and 'plus' not in v['benchmark'])]),
'## RoboCasa internal 17-task benchmark, 50k checkpoints','',
'Protocol: environment seed42, 24 vector envs, 48 episodes/task, horizon500, execute8 of predicted16, native images, left external + wrist, state included. Training seeds4 and42 are separate. Training dataset is target_atomic_2cam, not Human300 pretraining. Each arm has the same 17 tasks; PickPlaceSinkToCounter is omitted by both manifests. The official atomic set has 18 tasks; official overall has 50.','',table(['arm','train seed','task count','% ± SE; successes/n','manifest'],[(arm,seed,len(v['tasks']),fmt(v['pool']),link(f'/e/scratch/m3/blank4/rc365_smoke/manifest_full_s{seed}.txt')) for (arm,seed),v in rbench.items()]),
'',table(['task','causal s4','v5 s4','causal s42','v5 s42','sources'],[(task,*[fmt(rbench[k]['tasks'][task]) for k in [('causal',4),('v5',4),('causal',42),('v5',42)]],' '.join(link(rbench[k]['tasks'][task]['path'],f'{k[0]}s{k[1]}') for k in [('causal',4),('v5',4),('causal',42),('v5',42)])) for task in sorted(rbench['causal',4]['tasks'])]),
'## State-value corruption — separate three-task diagnostic','',
'Only seed4, 50k; OpenStandMixerHead, PickPlaceCounterToStove, TurnOnElectricKettle. Format retained for st_*; code replaces the sin/cos state before tokenization. Shuffle permutes slots; random uses random angles with sin/cos pairs; zero makes all slots zero (off the sin/cos manifold). The JSON lacks an explicit corruption-mode field: tag + manifest + client code are the provenance, so future evals should serialize mode and realized input hashes.','',table(['run','tag','% ± SE; successes/n','raw task sources'],[(k[0],k[1],fmt(rcp[k]),' '.join(link(x['source_path'],x['suite_or_task']) for x in v)) for k,v in sorted(rc.items()) if k[1].startswith('st_')]),
'',table(['task','causal clean','v5 clean','causal shuffle','v5 shuffle','causal zero','v5 zero','causal random','v5 random'],[(task,*[fmt(pool([x for x in rows if x['run']==f'ervla_robocasa365_pi_{arm}_s4' and x['tag']==tag and x['suite_or_task']==task])) for tag in ['s4_50k','st_shuffle','st_zero','st_random'] for arm in ['causal','v5']]) for task in ['OpenStandMixerHead','PickPlaceCounterToStove','TurnOnElectricKettle']]),
'## No-state diagnostic — separate from corruption','',table(['run','tag','% ± SE; successes/n','raw task sources'],[(k[0],k[1],fmt(rcp[k]),' '.join(link(x['source_path'],x['suite_or_task']) for x in v)) for k,v in sorted(rc.items()) if k[1]=='nostate']),
'## Smoke / partial LIBERO-plus roots (never benchmark numbers)','',table(['run','directory','% ± SE; successes/n','source'],[(v['run'],v['benchmark'],fmt(v),link(v['path'])) for v in sorted(plus.values(),key=lambda v:(v['run'],v['benchmark'])) if 0<v['n']<4000]),
'## Other RoboCasa smoke / partial protocols','',table(['run','tag','protocol','% ± SE; successes/n','sources'],[(k[0],k[1],k[2],fmt(rcp[k]),' '.join(link(x['source_path'],x['suite_or_task']) for x in v)) for k,v in sorted(rc.items()) if not (re.fullmatch(r's\d+_50k',k[1]) or k[1].startswith('st_') or k[1]=='nostate')]),
'## Same-run directory discrepancies and small/full sensitivity','',
'Changing sample size also changes task composition, and some directories reflect inference fixes. These deltas are descriptive sensitivity, not a causal estimate of statistical shrinkage. All surviving four-suite small artifacts are shown, not just favorable transitions.','',table(['run','small directory','small % ± SE; s/n','canonical directory','full % ± SE; s/n','Δ pp'],[(x['run'],x['old']['benchmark'],fmt(x['old']),x['new']['benchmark'],fmt(x['new']),f"{100*x['delta']:+.3f}") for x in trans]),
'',f'{len(dups)} root/suite-local discrepancies found; see [audit_details.json](results_collected/audit_details.json) for every count and source. They can be stale roots while re-evaluation is writing suite files. Do not merge these partial writes with old root totals.','',
'## Inventory and failures','',
'All result directory names, including directories without root aggregates: '+', '.join('`'+x+'`' for x in sorted({p.name for p in (A/'playground/Checkpoints').glob('*/results/*') if p.is_dir()})),
'',f'{len(fail)} nonempty failed_shards files; '+(f'{len(failunits)} failed_units files found.' if failunits else '**No failed_units_* files exist under the requested scratch tree.** Completion was checked against manifest task sets and JSON successes, with rc=1 client logs read as failed attempts, not zero outcomes.'),'',table(['failed shard source','missing shards'],[(link(f['path']),'; '.join(f['lines'])) for f in fail])]
(A/'RESULTS_CONSOLIDATED.md').write_text('\n'.join(report)+'\n'+(A/'scripts/encoder_audit_narrative.md').read_text())
details={'families':fs,'perturbations':pert,'canonical':canonical,'root_results':list(roots.values()),'rc_benchmark':{f'{a}_s{s}':v for (a,s),v in rbench.items()},'rate_mismatches':issues,'root_suite_discrepancies':dups,'failed_shards':fail,'small_full_transitions':trans}
(O/'audit_details.json').write_text(json.dumps(details,indent=1))
print('RAW',len(raw),'ROWS',len(rows),'CANONICAL',len(canonical),'ROOT_DIFFS',len(dups),'RATE',issues)
print('FAMILIES',json.dumps(fs,indent=1))
print('PERT',json.dumps(pert,indent=1))
print('TRANS',len(trans), 'mean',st.mean(x['delta'] for x in trans),'median',st.median(x['delta'] for x in trans),'range',min(x['delta'] for x in trans),max(x['delta'] for x in trans))
