# ERVLA / starVLA consolidated results

Recomputed 2026-09-17 directly from JSON. `results_collected/results_master.csv` contains all 1,954 raw suite/task rows and absolute sources. `±` is binomial SE. In tables, `src` means `/e/project1/m3/blank4/code/starVLA/playground/Checkpoints/<run>/results/<dir>/overall_results.json`; RoboCasa sources are under the upstream-merge checkpoint evaluation directory.

Canonical rule: one root `overall_results.json` per run/checkpoint, direct `libero-plus` if complete, otherwise named `libero-plus-4k-exact-*`. Suite-local files remain in the master CSV but are never pooled with their root. Thus sibling result directories are separate artifacts, not silently substituted results.

## LIBERO-plus full-protocol leaderboard

| arm / run | success ± SE (n) | seed spread | src dir |
|---|---:|---:|---|
| causal `...nodrop_2gpu` s42/s43/s44 | 0.717 ± 0.004 (12,000) | .725/.715/.710; SD .0077 | `libero-plus` |
| encoder v5 PI s42/s43 | .731 ± .005 (8,000) | .729/.734; SD .0035 | `libero-plus` |
| encoder v5 PI + aug s42/s43 | .734 ± .005 (8,000) | .748/.720; SD .0196 | `libero-plus` |
| GR00T v5 action-only s42/s43 | .746 ± .005 (8,000) | .754/.739; SD .0103 | `libero-plus` |
| shared-z zonly v5 s42/s43 | **.774 ± .005 (8,000)** | .765/.783; SD .0127 | `libero-plus` |
| shared-z zonly, older backbone | .776 ± .006 (4,149) | one seed | `libero-plus` |
| shared-z zsup | **.781 ± .007 (4,000)** | one seed | `libero-plus-4k-exact-sharedz-v1` |
| encoder `w` | .764 ± .007 (4,000) | one seed | `libero-plus-4k-exact-v1` |
| encoder `w2` no-latent / latent32 | .747 / .745 ± .007 (4,000 each) | one seed | `...correctedpi-r2` |
| encoder MLM control / cam3d | .755 / .739 ± .007 (4,000 each) | one seed | `...mlm-fast-v1` |
| CoT `k...masked` s42/s43 | .769 / .742 ± .007 (4,000 each) | SD .0189 | `...v1` |
| CoT layerwise / random | .760 / .768 ± .007 (4,000 each) | one each | `...rawcot32-v1` / `libero-plus` |
| Qwen-0.8B GR00T deeps | .772 ± .007 (4,000) | one seed | `libero-plus` |

The matched shared-z-v5 minus causal effect is +.057 with combined binomial SE .0062 (9.2 SE). Encoder-v5-PI is only +.014 with combined SE .0070 (2.1 SE). GR00T-v5 is +.030 with combined SE .0068, so head and backbone confound an encoder-objective conclusion. `w`, `w2`, MLM and zsup are single-seed evidence.

## LIBERO-plus perturbations

Sources are causal s42/s43/s44 pooled, zonly-v5-s43, and `ervla_w_pi_encoder_actiononly` in `libero-plus-4k-exact-v1`; all are root JSON paths under the convention above.

| perturbation | causal | zonly-v5 | gap | encoder w | gap |
|---|---:|---:|---:|---:|---:|
| Sensor Noise | .799 ± .009 (1,914) | .818 ± .015 (638) | +.019 | .787 ± .016 (638) | -.012 |
| Background Textures | .915 ± .008 (1,293) | .882 ± .016 (431) | -.033 | .923 ± .013 (431) | +.008 |
| Robot Initial States | .509 ± .012 (1,854) | .662 ± .019 (618) | +.153 | .600 ± .020 (618) | +.091 |
| Camera Viewpoints | .518 ± .011 (1,911) | .601 ± .019 (637) | +.083 | .584 ± .020 (637) | +.066 |
| Language Instructions | .809 ± .009 (1,845) | .868 ± .014 (615) | +.059 | .867 ± .014 (615) | +.058 |
| Objects Layout | .702 ± .011 (1,824) | .827 ± .015 (608) | +.125 | .793 ± .016 (608) | +.091 |
| Light Conditions | .870 ± .009 (1,359) | .883 ± .015 (453) | +.013 | .881 ± .015 (453) | +.011 |

Categories exhaust the 4,000 episodes; there is no separate ID stratum. The shared-z result is mainly initial-state, layout, and viewpoint robustness, not an identified in-distribution gain, and it loses on backgrounds.

## Plain LIBERO

The on-disk plain protocol is only 100/suite = 400/run and is reported separately.

| arm / run | success ± SE (n) | src dir |
|---|---:|---|
| causal s42/s43/s44 pooled | .966 ± .005 (1,200) | `libero` |
| shared-z zonly / zsup | .965 ± .009 / .973 ± .008 (400 each) | `libero` |
| encoder w / MLM control | .978 ± .007 / .970 ± .009 (400 each) | `libero` |
| GR00T-v5 s42/s43 | .950 ± .011 / .968 ± .009 (400 each) | `libero` |

The best encoder-causal gap is +.012 versus combined SE about .009: below two SE and ceiling-limited.

## RoboCasa at 50k steps

Every rate was recomputed from the boolean `successes` list; no declared-rate mismatch was found. Both s4 and s42 now have all 17 x 48 task JSONs (the job-1850179 “15/34” launch status is stale; later clients often refused to overwrite valid files). There are 129 result JSONs and 144 rollout videos.

| arm | s4 | s42 | pooled | seed SD |
|---|---:|---:|---:|---:|
| causal | .363 ± .017 (816) | .348 ± .017 (816) | .355 ± .012 (1,632) | .0104 |
| v5 encoder | .373 ± .017 (816) | .384 ± .017 (816) | .378 ± .012 (1,632) | .0078 |

v5-causal is +.023 ± .017 (1.34 SE): directionally consistent but not distinguishable.

| per-task, pooled seeds (n=96/arm) | causal | v5 |
|---|---:|---:|
| CloseBlenderLid / CloseFridge / CloseToasterOvenDoor | .021 ± .015 / .177 ± .039 / .490 ± .051 | .042 ± .020 / .229 ± .043 / .573 ± .050 |
| CoffeeSetupMug / NavigateKitchen / OpenCabinet | .542 ± .051 / .125 ± .034 / .052 ± .023 | .396 ± .050 / .146 ± .036 / .125 ± .034 |
| OpenDrawer / OpenStandMixerHead / CounterToCabinet | .271 ± .045 / .802 ± .041 / .396 ± .050 | .385 ± .050 / .844 ± .037 / .344 ± .048 |
| CounterToStove / DrawerToCounter / ToasterToCounter | .635 ± .049 / .281 ± .046 / .104 ± .031 | .698 ± .047 / .260 ± .045 / .115 ± .033 |
| SlideDishwasherRack / TurnOffStove / ElectricKettle | .531 ± .051 / .312 ± .047 / .583 ± .050 | .448 ± .051 / .344 ± .048 / .688 ± .047 |
| TurnOnMicrowave / TurnOnSinkFaucet | .240 ± .044 / .479 ± .051 | .385 ± .050 / .406 ± .050 |

### State interventions: s4, three selected tasks only

Source tags are `st_shuffle`, `st_zero`, `st_random`, and `nostate`. The native reference is causal .694 ± .038 (100/144), v5 .729 ± .037 (105/144).

| input | causal | v5 | conclusion |
|---|---:|---:|---|
| shuffle | .653 ± .040 (94/144) | .604 ± .041 (87/144) | no v5 advantage |
| random | .528 ± .042 (76/144) | .625 ± .040 (90/144) | +.097, 1.7 SE |
| zero | .153 ± .030 (22/144) | .306 ± .038 (44/144) | +.153, 3.1 SE |
| no state | .000 (0/144) | .590 ± .041 (85/144) | v5 visual fallback |

Causal needs proprioception. V5 does not ignore state (zero/shuffle hurt it), but it can act without it. This supports a representation mechanism, not aggregate superiority.

## Probes and downstream evaluations

The probe scripts were regenerated from 435 JSONL records; `PROBE_TABLE.md` has 99 cells, 876/1,089 fields populated (two DROID-wrist cells absent, not imputed). Direct starVLA RoboCasa patch probes: causal .0382 versus enc-dec .2699 mean-reference R2 (n=960 each; `svlap_casa_causal.log`, `svlap_casa_encdec.log`), yet rollout is only +.023 ± .017.

| arm, external views | Bridge R2 (n=2,400) | DROID R2 (n=2,399) | harvested downstream aggregate |
|---|---:|---:|---:|
| full epoch | .1511 | .0967 | .6904 |
| final action head | .1643 | .1194 | .6836 |
| final action linear | .1995 | .1200 | .6780 |
| final action tracetime | **.2097** | **.1238** | .6400 |

Decodability ordering does not match benchmark ordering. These probe arms are also not one-to-one with all starVLA policy checkpoints, so R2 is diagnostic rather than a behavioral proxy.

## Smoke, incomplete results, and live jobs

| run | success ± SE (n) | status / source |
|---|---:|---|
| CoT ours-v3 cotw.1 | .826 ± .011 (1,153) | partial root `libero-plus`; job 1850350 running |
| CoT det-v3 cotw.1 | .822 ± .011 (1,153) | partial root `libero-plus`; job 1850351 running |
| CoT readout | .790 ± .012 (1,153) | partial; job 1850352 running |
| CoT full cotw1 | .787 ± .012 (1,153) | 1855180 running but repeatedly exhausts shards rc=134 |
| `k...masked` | .805 ± .025 (256) | became .769 ± .007 (4,000) |
| MLM control | .777 ± .026 (256) | became .755 ± .007 (4,000) |

No valid 4,000-episode CoT confirmation exists. Matched small-to-full changes range about -.052 to +.052, so a numerical “expected shrinkage” is not defensible; selection bias makes the high partial CoT scores especially unsafe. No jobs were launched in this audit.

## Discrepancies and collector fix

1. The old collector only searched one depth and missed nested/sibling results. It now recursively collects all `overall_results.json`, retains `result_dir`, recomputes RoboCasa rates, and writes `results_master.csv`.
2. Five raw hand checks match the collector: causal s42 2,900/4,000; zonly-v5-s43 3,131/4,000; encoder w 3,057/4,000; plain k 231/400; RoboCasa v5 OpenDrawer 20/48.
3. `ENCDEC_STATUS.md` is right that the original RoboCasa effect is neutral, but its seed-42-incomplete statement is stale for the files now present. Renderer/client errors did occur and are documented in logs.
4. The status narrative’s CoT .826/.822 must carry n=1,153; 1855180 currently has failed-shard evidence, not a landed re-evaluation.
5. Plain LIBERO is saturated and LIBERO-plus is entirely perturbation-stratified; neither warrants a broad “best across benchmarks” conclusion.
6. `docs/STATUS.md` flags probe measurement bugs. After regeneration, high probe R2 still fails to predict downstream aggregate ordering.
