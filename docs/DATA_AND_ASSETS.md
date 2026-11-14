# Data, checkpoints and stored outputs

None of the following is part of this repository. All locations are set through environment variables (`secondlook/paths.py`); the sub-directory names below are the
ones the code looks for.

## `$SECONDLOOK_DATASETS`

`scene-segmentation-v1/datasets/{totaltext,icdar2015,ctw1500}` with the official images and annotations of the three benchmarks
(Total-Text 1,255 training / 300 test, ICDAR2015 1,000 / 500, CTW1500 1,000 / 500). Total-Text and ICDAR2015 training pools of 1,251 and 1,000 images are used (Sect. 4.2 of the paper).

## `$SECONDLOOK_ARTIFACTS`

| Directory | Content |
|---|---|
| `deepsolo-host/DeepSolo-upstream/DeepSolo` | DeepSolo source at the commit of the paper (dbadae9) and its configurations |
| `official-benchmark-audit-20261031-v1` | the pinned upstream evaluators of Total-Text and ICDAR2015 and the ground-truth archives (`gt_totaltext.zip`, `gt_icdar2015.zip`) |
| `compact-ocr-cascade-20261103-v1` | PP-OCRv5 Mobile ONNX models and the runtime that hosts them |
| `parseq/parseq-1902db04...` | PARSeq source at the commit used for PARSeq-S and PARSeq-S\* |
| `parseq-retrain` | training data, outputs and `final_swa.ckpt` of PARSeq-S\* (`training/parseq_star/`) |
| `abinet`, `vitstr/vitstr_small_patch16_224_aug.pth` | ABINet and ViTSTR-S sources and weights (second-reader comparison) |

TiCLS (authors' released weights for Total-Text and ICDAR2015) and DeepSolo checkpoints are taken from the original projects; the TiCLS CTW1500 fine-tune is produced by
`spotters/ticls_ctw1500_finetune.py`.

## `$SECONDLOOK_RUNS`

Stored outputs, one directory per origin. Every table is computed from these; no spotter is run when a table is rebuilt.

| Directory | Content | Produced by |
|---|---|---|
| `ticls-source-banks` | TiCLS candidates and word evidence for Total-Text and ICDAR2015 | `spotters/ticls_inference.py`, `data/ticls_word_evidence.py` |
| `deepsolo-compact-outputs` | DeepSolo candidates (per-character probabilities) for Total-Text and ICDAR2015 | `spotters/deepsolo_worker.py`, `spotters/deepsolo_outputs.py` |
| `ppocr-predictions` | PP-OCRv5 Mobile predictions, train and test, three benchmarks | `spotters/ppocr.py`, `spotters/ppocr_collect.py` |
| `ctw1500-deepsolo` | DeepSolo R50 candidates on CTW1500 | `spotters/deepsolo_ctw1500.py` |
| `ctw1500-ticls-finetune` | TiCLS CTW1500 fine-tune and its candidate bank | `spotters/ticls_ctw1500_*.py` |
| `fulltrain` | candidates of the extra training images, crops and the second-reader readings of the full training pools | `data/collect_extra_train.py`, `studies/fulltrain_chain.py` |
| `candidate-banks` | per-row candidate grids and the stored second-reader readings (`p3/`) | `data/cv_counts.py`, `second_stage/crops.py`, `second_stage/read_*.py` |
| `operating-point` | per-row cut `c` and band `k` files | `selection/operating_point.py`, `selection/verification_band.py` |
| `unified-features` | PP-OCR word-level symbol statistics | `spotters/ppocr_stats.py` |
| `rerun-fast`, `rerun-exact` | outputs of the two final-table layers (`main/`, `stability/`, `lc/`, `ablations/`, `readers/`, `crossdata/`) | `reruns/` |

A few modules also read configurations that pin the hash of some of these inputs. `python -m secondlook.setup_configs` writes `configs/approximate/*.json` from
`configs/templates/*.json` for the local paths; an input that is pinned and present must match its sha256 when the configuration is loaded, an input that is pinned and absent is dropped
and listed (use `--strict-pins` to make that an error). The templates still mention a few directories of earlier studies with their original names
(`sota-expert-router-v1`, `ijdar-reuse-ctw-20261102-v1`, `icdar-router-research-20261102-v1`, `uniform-unet-fusion-20261030-v1`, `dataset-router-architecture-20261103-v1`,
`deepsolo-icdar2015-gap-20261028-v1`); they hold stored candidate lists and manifests that the pins refer to.

## Result class

Every number is an approximate reproduction in the sense of the working protocol of the study: the spotters, readers and evaluators are the published ones, the
unspecified choices (cut rule, band grid, feature definitions) are fixed in the code and documented in the paper.
