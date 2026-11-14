# Study labels in the code

Docstrings carry short labels (`E18`, `P3`, ...) from the working log of the study. The log itself is not part of this release; this table says what each label refers to.

| Label | Content | Module |
|---|---|---|
| E6 | first classifier on cached DeepSolo outputs (candidate table, features) | `data/deepsolo_table.py` |
| E7 | repeated source partitions, cohort helpers | `data/cohort.py` |
| E10 | signal ablation and learning curve of the first classifier; the `Task` classes | `data/tasks.py` |
| E11 | CTW1500 with the official DeepSolo R50 checkpoint | `spotters/deepsolo_ctw1500.py` |
| E13 | classifier family, calibration, voting | `studies/ablation_learners.py` |
| E14 | negative controls, cut-rule ablation, hyper-parameters | `studies/ablation_controls.py` |
| E15 | cross-expert agreement as a signal | `studies/ablation_agreement.py` |
| E17, E27 | TiCLS fine-tune on CTW1500, its evaluation and candidate bank | `spotters/ticls_ctw1500_*.py` |
| E18, E20, E21 | calibration protocol on the test split; the same on CTW1500 and with PP-OCR as third spotter | `data/cv_counts.py`, `data/tasks_ctw1500.py`, `data/tasks_ppocr.py` |
| E30 | the F/2 operating point | `selection/operating_point.py` |
| E31 | training-only choice of the verification band | `selection/verification_band.py` |
| E33, E36 | PARSeq-S\* (trained without ArT) as second reader and its harness | `second_stage/read_parseq_star.py`, `reader_harness.py` |
| E35 | learning curve on the nine rows | `studies/learning_curve.py` |
| E38, E39 | hold-out check; the single pipeline used for ablations, stability and tables | `selection/nested_selection.py`, `studies/` |
| E42, E43 | full official training pools; second readers on every row | `data/tasks_fulltrain.py`, `studies/fulltrain_chain.py` |
| E44 | detection-only metrics, cross-dataset fits | `evaluation/detection_only.py`, `studies/cross_dataset.py` |
| E46, E46b | unified feature definition; exact selection of c, k and the baseline | `features/unified_features.py`, `selection/exact_*.py` |
| E49 | final-table layer 1 (unified features, label-based selection) | `reruns/fast/` |
| E50 | final-table layer 2 (exact selection) | `reruns/exact/` |
| P3 | the second stage in three steps: crops, readings, band classifier | `second_stage/` |
| V-ABI, V-CLEAN | ABINet and ViTSTR-S readers | `second_stage/read_abinet.py`, `read_vitstr.py` |
