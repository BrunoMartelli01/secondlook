# SecondLook

Code of the experiments in *Learning Which Words to Keep: Confidence-Based Filtering and Selective Re-reading for Scene-Text Spotters*.

SecondLook sits on top of a trained scene-text spotter and decides which of its predicted words to keep, without retraining or modifying it.
A gradient-boosted classifier scores every prediction from eleven detection, recognition, geometry and transcript signals. Uncertain predictions are re-read by an
independent recogniser (PARSeq-S\*, PARSeq-S retrained without ArT), and a second classifier combines the two readings with the first-step evidence.
The experiments use DeepSolo, TiCLS and PP-OCRv5 Mobile on Total-Text, ICDAR2015 and CTW1500 (nine spotter x benchmark rows).

## Where to start

| You want to | Read |
|---|---|
| find the code behind a table or figure of the paper | [docs/PAPER_MAP.md](docs/PAPER_MAP.md) |
| follow the method step by step in the code | [docs/PIPELINE.md](docs/PIPELINE.md) |
| know which data, checkpoints and stored outputs the code expects | [docs/DATA_AND_ASSETS.md](docs/DATA_AND_ASSETS.md) |
| see the numbers that the paper reports | [results/](results/) |
| decode the `E<n>` labels in docstrings | [docs/STUDY_LABELS.md](docs/STUDY_LABELS.md) |

## Layout

```
secondlook/
  paths.py            every machine-specific location (environment variables, see below)
  setup_configs.py    writes configs/approximate/*.json for this machine from configs/templates/
  data/               candidate pools of the three spotters, their labels, one Task object per row
  features/           the eleven signals (unified_features.py is the definition used in the paper)
  selection/          cut c (F/2 rule), verification band k (nested cross-fit), baseline thresholds, exact selection
  second_stage/       crops, second readers (PARSeq-S*, ABINet, ViTSTR-S), band classifier, baselines
  evaluation/         pinned official evaluators, per-image counts, detection-only metrics
  studies/            one module per analysis (stability, ablations, learning curve, cross-dataset, tables)
  spotters/           inference adapters and candidate collection (DeepSolo, TiCLS, PP-OCR), TiCLS CTW1500 fine-tune
  reruns/fast/        final-table layer 1: unified features, label-based F1 for every selection
  reruns/exact/       final-table layer 2: every selection made with the exact pinned evaluator (the numbers of the paper)
  figures/            redraws the data plots (Figs. 5 and 6) from the result tables
configs/templates/    the three configurations, with placeholders for the local paths
scripts/              shell glue for the long jobs (job runner, cross-dataset chain)
training/parseq_star/ recipe used to retrain PARSeq-S without ArT
results/              result tables and machine-readable results of the final run
tests/                import and unit tests that need no data
```

## Setup

The analysis code runs on CPU with Python 3.8, NumPy 1.20, SciPy 1.5, scikit-learn 0.23, Shapely 2.0, OpenCV 4.5, Polygon3 and PyTorch 1.9
(see `requirements.txt`). The official DeepSolo and TiCLS evaluators and sources are third-party code that is not redistributed here; they are expected under
`$SECONDLOOK_ARTIFACTS` (see [docs/DATA_AND_ASSETS.md](docs/DATA_AND_ASSETS.md)).

```bash
export SECONDLOOK_RUNS=/path/to/runs               # stored candidate banks, readings and run outputs
export SECONDLOOK_ARTIFACTS=/path/to/artifacts     # pinned third-party sources, evaluators, checkpoints
export SECONDLOOK_DATASETS=/path/to/datasets       # Total-Text, ICDAR2015, CTW1500
export SECONDLOOK_PYTHON=$(which python)           # analysis interpreter
python -m secondlook.setup_configs                 # writes configs/approximate/*.json
PYTHONPATH=. python -m unittest discover -s tests   # no data needed
```

All commands are run from the repository root. One analysis end to end (learning-curve draw 0 of TiCLS on ICDAR2015, label-based selection):

```bash
python -m secondlook.reruns.exact.launch fast secondlook.studies.learning_curve \
    --expert ticls --task icdar2015 --seed 0 --out out/lc
```

## What is and is not in this repository

* The code of every analysis of the paper tables, the evaluation and bootstrap code, the PARSeq-S\* training recipe and the TiCLS CTW1500 fine-tune.
* Not included: datasets, spotter and reader checkpoints, third-party sources (DeepSolo, TiCLS, PARSeq, ABINet, ViTSTR, PaddleOCR) and the stored spotter outputs that every table is
  computed from. Every table is computed from stored candidate predictions; nothing is re-inferred when a table is rebuilt.
* The code is released as it was run. Selections of c, k and the baseline pair come in two protocols, label-based (`reruns/fast`) and exact (`reruns/exact`);
  the paper reports the exact one. The two layers patch the same unchanged analysis modules, so read `reruns/exact/launch.py` first.
* Figures 1, 3, 4 and 7 of the paper are illustrations drawn from stored outputs; their drawing scripts are not part of this release.
* The timing of Sect. 4.2 (PARSeq-S at 68 ms per crop) was measured with a separate micro-benchmark that is not included.

## Reproducibility notes

* Seeds: HistGradientBoosting `random_state=11`; five-fold cross-fits grouped by image; learning-curve draw `s` uses `RandomState(1000 + s)`; stability subsets use
  `RandomState(100 + s)`; paired bootstrap 2000 draws with seed 20261106 and one index matrix per row.
* The evaluators are the pinned upstream scripts of each benchmark, called unchanged; per-image counts are summed to pooled precision, recall and F1.
