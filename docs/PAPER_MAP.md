# From paper to code

Every table of the paper is computed from stored candidate predictions of the spotters. The final numbers come from the exact-selection run (`secondlook/reruns/exact`),
which re-runs the analysis modules of `secondlook/studies`, `secondlook/selection` and `secondlook/second_stage` with every selection made by the pinned evaluator.
`results/exact_selection/tables.md` holds the printed tables of that run; its section titles keep the numbering of the working draft, so the second column below
gives the title to search for.

`$RUN` is `$SECONDLOOK_RUNS/rerun-exact`. `ARM` is `exact` for the paper numbers and `fast` for the label-based comparison. Commands run from the repository root.
The nine rows are `{deepsolo, ticls, ppocr} x {totaltext, icdar2015, ctw1500}`.

| Paper | Section of `tables.md` | What runs | Code |
|---|---|---|---|
| Tab. 6 first step: F1, differences with 95% intervals | Table 4 | main chain, stages `op`, `band`, `test` | `reruns/exact/main_chain.py`, `selection/operating_point.py`, `second_stage/band_classifier.py`, `reruns/fast/report_main.py` |
| Tab. 7 precision, recall, output words | Table 5 | same run | same |
| Tab. 8 second step (k, crops per image, F1, gains) | Table 6 | same run, stage `band` selects k | `selection/verification_band.py`, `selection/nested_selection.py` |
| Fig. 6 band-width sensitivity; arg max of the training F1 (Sect. 4.6) | Table 15, Table 14 | same run, stages `sweep` and `targmax` | `reruns/exact/main_chain.py`, `figures/make_fig56.py` |
| Tab. 5 detection-only P/R/F, F std over eleven 80% subsets | Table 3 | stability, 12 training sets per row | `reruns/exact/stability.py`, `evaluation/detection_only.py`, `reruns/fast/report_stability.py` |
| Fig. 5 learning curve | Table 8 of rev 48 / Fig. 5 | learning curve, 11 draws per row | `studies/learning_curve.py`, `figures/make_fig56.py` |
| Tab. 9 cross-dataset (own, leave-one-out, pooled) | Table 8 / 10 | cross-dataset chain | `studies/cross_dataset.py`, `reruns/exact/cross_dataset.py`, `scripts/cross_chain.sh` |
| Tab. 10, variant signals (digit share, longest token) | not in `tables.md` | leave-one-out refits with replaced columns | `studies/loo_robust/` |
| Tab. 10, first-step feature ablation | Table 9 (user numbering) | ablations, stage `all` | `studies/ablations.py` |
| Tab. 11 second readers (PARSeq-S\*, ABINet, ViTSTR-S, shuffled) | Table 10 (user numbering) | readers chain | `reruns/exact/readers_chain.py`, `second_stage/read_*.py`, `reruns/fast/report_readers.py` |
| Reader accuracy on matched training candidates (Sect. 4.2 and 4.6) | Table 17 | reader accuracy | `second_stage/reader_accuracy.py` |
| Released and re-tuned baselines (Sect. 4.1.3) | rows "thr_train" | two-threshold grid on the training pool | `data/tasks.py` (`pick_thresholds`), `second_stage/baselines.py` |
| TiCLS fine-tune on CTW1500 (Sect. 4.2) | - | fine-tune, evaluation, candidate bank | `spotters/ticls_ctw1500_finetune.py`, `_eval.py`, `_bank.py` |
| PARSeq-S\* (Sect. 3.4, 4.2) | - | retraining without ArT | `training/parseq_star/` |
| Tab. 12 comparison with reported results | - | numbers copied from the cited papers | - |
| Figs. 1, 3, 4, 7; timing of PARSeq-S (Sect. 4.2) | - | illustrations and a micro-benchmark | not included |

## Commands

```bash
export SECONDLOOK_RUNS=... SECONDLOOK_ARTIFACTS=... SECONDLOOK_DATASETS=... SECONDLOOK_PYTHON=$(which python)
python -m secondlook.setup_configs
RUN=$SECONDLOOK_RUNS/rerun-exact

# S = all chain: first step, band selection, second step, band sweep, baselines (Tab. 6-8, Fig. 6)
python -m secondlook.reruns.exact.main_chain --arm exact --out $RUN --jobs 4 --threads 2
python -m secondlook.reruns.exact.readers_chain --arm exact --out $RUN --jobs 4        # Tab. 11

# one job per row and subset, through the job runner (log in $RUN/logs)
scripts/run_job.sh exact stab_ticls_icdar2015 secondlook.reruns.exact.stability --expert ticls --task icdar2015 --out $RUN
scripts/run_job.sh exact lc_ticls_icdar2015_s0 secondlook.studies.learning_curve --expert ticls --task icdar2015 --seed 0 --out $RUN/exact/lc/out
scripts/run_job.sh exact abl_ticls_icdar2015 secondlook.studies.ablations --expert ticls --task icdar2015 --stage all --out $RUN/exact/ablations
scripts/run_job.sh exact ablarg_ticls_icdar2015 secondlook.studies.ablations --expert ticls --task icdar2015 --stage table10_argmax --out $RUN/exact/ablations
scripts/cross_chain.sh exact ticls                                                        # Tab. 9 for one spotter

# job lists for all nine rows: python -m secondlook.reruns.exact.make_jobs {stab|lc|abl|abl_argmax} --out $RUN --arms exact > jobs.txt
#                              scripts/run_jobs.sh jobs.txt 4

# tables and figures
python -m secondlook.reruns.exact.report --out results/exact_selection
python -m secondlook.figures.make_fig56 results/exact_selection/tables.md out/figures
```

The exact-selection layer needs the cache of the pinned evaluator (`SECONDLOOK_EXACT_CACHE`, default `$RUN/exact_cache`); the first run of a row fills it.
`python -m secondlook.reruns.exact.launch fast MODULE ...` runs a module with the label-based selection instead, which is how the smoke test in the README runs.
To check that the cached exact service agrees with the whole-pool evaluator:

```bash
python -m secondlook.reruns.exact.launch fast secondlook.reruns.exact.verify_exact_service --expert ticls --task icdar2015 --out out/g1 --sets 3
```
