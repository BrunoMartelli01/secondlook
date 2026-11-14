# The method in the code

One *row* is a spotter on a benchmark (for example TiCLS on ICDAR2015). The code builds a `Task` object per row (`secondlook/data/tasks.py` and its variants for the
full training pools, CTW1500 and PP-OCR). A task holds, for the training pool and the test split, one table per image with the candidates of the spotter:

| Field | Content |
|---|---|
| `tab[s]['lines']`, `eval_tab[s]['lines']` | the submission line of every candidate (polygon and transcription) as written to the evaluator |
| `tab[s]['xc']` | the eleven signals of the first classifier (`features/unified_features.py`, `UNIFIED_COLUMNS`) |
| `eval_tab[s]['score']`, `['rec']` | detection score and recognition confidence (geometric mean of the symbol probabilities), used by the baselines |
| `lab[s]` | label of every training candidate from the native evaluator of the benchmark (1 correct, 0 wrong, -1 do-not-care) |
| `ids`, `eval_ids` | training and test image ids |

## Steps

1. **Candidates and signals.** The spotter output is stored once (`spotters/`, `data/collect_extra_train.py`). The eleven signals are built by
   `features.unified_features.build_unified`: detector logit; geometric mean, minimum, mean entropy and minimum margin of the symbol probabilities;
   log relative area, log height, log aspect ratio of the submitted polygon; log length, digit fraction, letter fraction of the transcript. The first five depend on the
   decoder of each spotter and are documented in the module docstring.
2. **First classifier.** `HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=7, l2_regularization=5, min_samples_leaf=20, early_stopping=False, random_state=11)`
   (`studies/ablation_controls.py`, `hgb`).
3. **Threshold c.** Five-fold cross-fit grouped by image (`second_stage/band_classifier.cross_fit`); the pooled F1 of the out-of-fold probabilities is computed for every
   threshold 0.00 ... 0.99 (`selection/operating_point.py`, `FINE`, `pooled_f1`); `c = round(F*/2, 2)`. `studies/learning_curve.run` shows the rule in a few lines.
4. **Verification band k.** For each `k` in {1.2, 1.4, 1.6, 1.9, 2.2, 2.6, 3.0, all} the band is `p < k c`; nested five-fold cross-fitting grouped by image scores each band
   and the smallest `k` within 0.1 F1 points of the best is kept (`selection/nested_selection.py`, `nested_select`; grid and tolerance in `selection/verification_band.py`).
5. **Re-reading.** Candidates in the band are cropped along their polygon to 32 x 128 pixels (`second_stage/crops.py`) and read by PARSeq-S\*
   (`second_stage/read_parseq_star.py`, checkpoint from `training/parseq_star/`). The six reading features (agreement, edit similarity, sequence confidence, minimum token
   probability, mean log-probability, length difference) are built by `second_stage/band_classifier.verifier_features`.
6. **Second classifier.** Same learner, trained on the training candidates of the band with the eleven signals, `p` and the six reading features
   (`selection/nested_selection.band_fit`, `band_apply`).
7. **Decision.** A candidate with `p >= k c` is kept; a candidate in the band is kept if `q >= c`. First-step-only variant: keep if `p >= c`.
8. **Baselines.** *Released*: the threshold of the spotter's own configuration (`data/tasks.py`, `evaluation/common.author_keep`). *Re-tuned*: a detection-score threshold and a
   recognition-confidence threshold chosen jointly to maximise F1 on all training images (`data/tasks.pick_thresholds`, a grid of 135 pairs, ties to the larger score threshold).
9. **Evaluation.** The pinned upstream evaluator of each benchmark is called unchanged (`evaluation/official_benchmarks.py`; CTW1500 through `spotters/deepsolo_ctw1500.py`).
   Per-image counts (correct, predicted, ground truth) are summed to pooled precision, recall and F1. Paired image bootstrap: 2000 draws, seed 20261106, one index matrix per row
   shared by every arm (`evaluation/common.py`).

## Fast and exact selection

Steps 3, 4 and 8 choose parameters by maximising an F1 on the training pool. The *fast* layer (`reruns/fast`) uses the label-based F1 `2TP/(P+G)` of the stored candidate
labels. The *exact* layer (`reruns/exact`) replaces it with the F1 of the pinned evaluator itself: `reruns/exact/launch.py` patches `pooled_f1`, `approx_f1` and
`pick_thresholds` and serves them from `reruns/exact/exact_service.py`, which evaluates every distinct (image, set of kept candidates) once with the unchanged evaluator and caches
the per-image counts. Everything else is the same code. The paper reports the exact layer. `reruns/fast/launch.py` additionally substitutes the unified feature definitions
into every task object built in the process.

## Reading order

`studies/learning_curve.py` (one self-contained analysis), then `selection/` (the three selections), `second_stage/band_classifier.py`, `reruns/exact/launch.py`
and `studies/stability.py` (the whole pipeline for one row).
