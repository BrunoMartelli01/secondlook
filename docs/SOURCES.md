# Where to download the datasets and models we used

Every entry gives the source, the exact version and the sha256 of the file that was used in the experiments. Check a download with `sha256sum`: a different hash means a different file,
whatever its name. The links are the ones published by the original authors; the hosting (OneDrive, Google Drive, Hugging Face) is theirs and may change.
Nothing listed here is redistributed in this repository.

## 1. Datasets

| Dataset | Split used | File(s) used | Source |
|---|---|---|---|
| Total-Text | 1,255 training / 300 test images (1,251 training images after removing two exact test duplicates and two evaluator-rejected images, Sect. 4.2) | the 1,555 images and annotations of `Total-Text-Dataset` | Official: <https://github.com/cs-chan/Total-Text-Dataset>. The author host was unreachable, so the copy used is the Hugging Face mirror <https://huggingface.co/datasets/yunusserhat/Total-Text-Dataset> (`huggingface-cli download yunusserhat/Total-Text-Dataset --repo-type dataset`) |
| ICDAR2015 (Challenge 4, incident scene text) | 1,000 training / 500 test images | `ch4_training_images.zip` `07e2d816dab67df4cd509641b1e1e8c720d3b15873a65b6dc9c1aa6a4de3d2c6`; `ch4_training_localization_transcription_gt.zip` `b9f2f0343d016a326bcafe3c28e0ddbda18b3a1e4a8a595784ba9dc3e305d754`; `ch4_test_images.zip` `ecfb2488333372b7381b14ccb5ac2127de88fb935f6759fe89e5c041b0e87358`; `Challenge4_Test_Task1_GT.zip` `8f434c410cfcf680b8421e3c22abf31b9297d8e255c3b13acc2890fddc6db6b0` | <https://rrc.cvc.uab.es/?ch=4&com=downloads> (a free RRC account may be required). The first three files are served at `http://rrc.cvc.uab.es/downloads/<file name>` |
| CTW1500 | 1,000 training / 500 test images, text-line level | images: `ctw1500_mmocr.zip` `855714fba7127e96ef234d307c3d0e3efbaaeb59f46832b4cd58ff54d3784c71`; transcriptions: `ctw1500_e2e_annos.zip` `ffe19e67af44bd0d028da3436d7cc2e5577a723127a23d2e5e01b26fd53903f1` | Official: <https://github.com/Yuliang-Liu/Curve-Text-Detector>. The original Box links of the official repository no longer work; the images come from <https://huggingface.co/datasets/HB16888/CTW1500> (`ctw1500_mmocr.zip`). The transcriptions are the official end-to-end annotations, Google Drive id `1tHf-n86z4zFRaRNEc02EO-hZINZqkQpz` |

Evaluation files. The evaluators need the ground-truth archives and the lexicon files that the DeepSolo and TiCLS authors distribute with their code:

| File | sha256 | Source |
|---|---|---|
| `evaluation.zip` (contains `gt_ctw1500.zip`, `gt_icdar2015.zip`, `gt_totaltext.zip`, `gt_inversetext.zip`) | `f1a96e32dca5de7b4d53d040a42b6d8e6dd9a0b9f4892134fbce7bc0f5870155` | DeepSolo README, "Evaluation ground-truth" link (see section 2) |
| `ctw1500.zip` (`train_96voc.json`, `test.json`, `weak_voc_new.txt`, `weak_voc_pair_list.txt`) | `cdba62d5bf99133d947ac4c989964ab9dd39f23c810c2eb0066b29ffbcfb90b5` | DeepSolo README, CTW1500 annotations (see section 2) |
| TiCLS evaluation archive (`dataset.zip`; ICDAR2015 lexicons and ground truth) | `6c9a8b01caa7d21d4f022be47c6b0234ef2d4920bbc9af9467963a2d89fdad0e` | TiCLS README, "Evaluation" annotations, Google Drive id `1HaMVPxx0CzGIGKGqerZtYcoBMfHIP9PM` |
| `gt_totaltext.zip` | `125637c5b017f5a6d78726e6e50deba6284c60945c825d16f2f7b53edaf60776` | ground-truth archive taken from the TiCLS evaluation archive |
| `gt_icdar2015.zip` | `8f434c410cfcf680b8421e3c22abf31b9297d8e255c3b13acc2890fddc6db6b0` | byte-identical to the RRC `Challenge4_Test_Task1_GT.zip` (same sha256) |

## 2. Spotters

### DeepSolo

* Code: <https://github.com/ViTAE-Transformer/DeepSolo> at commit `dbadae995035246bad3376c7a44c015c69e9b313` (source archive sha256 `828c63e1c19f07d03ae094e4dd39a1eb093b0660a604bef75324f8e716693bfb`).
  The README of that commit lists every checkpoint and data link below.
* Checkpoints (README model zoo, links as listed there):

| Used for | README row | File used | sha256 | Link |
|---|---|---|---|---|
| Total-Text | ViTAEv2-S, Synth150K+MLT17+IC13+IC15+TextOCR | `vitae_textocr.pth` | `54d3a454ecc704ef3168ebf9f23fc8390a3b2637d0ec83d5650e8df45e2aca4a` | <https://1drv.ms/u/s!AimBgYV7JjTlgcd6XGlbZ-I7WvGslQ?e=rrkXLx> |
| ICDAR2015 | ViTAEv2-S, Synth150K+Total-Text+MLT17+IC13+TextOCR | `official_ic15_vitae_textocr.pth` | `c0aac6fddc297ca1a7e3c2d0bbd5f7f8b2f55214795c7fa32f13fb51c9db95a8` | <https://1drv.ms/u/s!AimBgYV7JjTlgcdqw1UUnbSAG4qoWA?e=Co1prY> |
| CTW1500 | Res-50, Synth150K+Total-Text+MLT17+IC13+IC15 | `finetune_ctw_96voc.pth` | `f9061cf4ac61112a73f231e77648514adaa22ec14bbc9e48ecb02e6dd75b6550` | <https://1drv.ms/u/s!AimBgYV7JjTlgcdsiFgSz-FHgKepqQ?e=56gdHj> |

* The CTW1500 annotations and the evaluation ground-truth bundle (`ctw1500.zip`, `evaluation.zip`) are linked in the same README under "Data Preparation".
  The released-output thresholds are those of the official configurations (0.4 on Total-Text and CTW1500; ICDAR2015 uses 0.4 here instead of the official 0.3, Sect. 4.1.3 of the paper).

### TiCLS

* Code: <https://github.com/knowledge-computing/TiCLS> at commit `c4b62956754a85ef372f1546fa90a5b725a44129` (source archive
  <https://codeload.github.com/knowledge-computing/TiCLS/tar.gz/c4b62956754a85ef372f1546fa90a5b725a44129>, sha256 `195316704fdbe538df9657db5d3e9080bad4f050ed6ca9c6b58890ac67b85bb4`).
* Weights (README "Model Weights", Google Drive; open `https://drive.google.com/file/d/<id>/view`):

| File used | sha256 | Google Drive id | README row |
|---|---|---|---|
| `ticls_totaltext_finetune.pth` | `6c3e383b029751d2c59942de655efb96b7eec11ad71352aeb678e4a88df25c53` | `10orop_RrLtl2OtJfdtVbGwZ36292YcxU` | TotalText, Finetune |
| `ticls_icdar2015_finetune.pth` | `1f2c3a34250de01af8f0d790f1edae5122478b8a633ceb6a0b3a0ddf7495b081` | `16sTq5nC_YvFYDXJlzp07dMoGYTMU4T7Y` | ICDAR 2015, Finetune |
| `bart_decoder_lm_head.pt` | `2b3a593dcd12f5d7abea430734d011718eeb2e9d0a2168bb5995a261737285fb` | `1asBTIUrhkq0QoknvxaE4vNmZ0Q-2X-Or` | PLM Decoder for TiCLS |
| `ticls_totaltext_pretrain.pth` (start of our CTW1500 fine-tune) | `1c092c0d2c12f58db24a4458933c1a1f3b664d6faa528bea629c0a4696c8f0d3` | `1da8HaaVaCIqnJycc4_Y6D6S3GQ9ic1Tl` | TotalText, Pretrain |
| tokenizer `vocab.json` / `merge.txt` | `8eb09841121028dd6790d8d8c2de2f0014ef315182115d1c92e80fce8543e58e` / `bd72fd818dbe3a380483600eada15e064b9e6bad23dd3dbc29ad78b8d043be5c` | folder `160uSNy0_UpBR6-NVn_1PKPZTdwaDb0xy` | PLM Tokenizer |
| `ticls_annotation.zip` (CTW1500 line-level training and test annotations in TiCLS format) | `38b3d99ce1f960d67129ea9fe0697be98618ae3b56be29b79b6f954cc0e538f0` | README "Annotations" | Training / Evaluation |

* **TiCLS on CTW1500 is our own fine-tune** (no CTW1500 weights are released by the authors): initialised from `ticls_totaltext_pretrain.pth`, 22,000 iterations, seed 20261105,
  recipe in `secondlook/spotters/ticls_ctw1500_finetune.py`. Resulting `model_final.pth` sha256 `2b8fa201dfeec8bb828a1414a3f7554d12ab0cb773eaa33550535d757bff98b8`.
  It is not published yet; see "Our own checkpoints" below.

### PP-OCRv5 Mobile

* Detector: <https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx> at revision `e6f4fa85f00e168c862bc462aebca69eef9b3d3d`; `inference.onnx` sha256 `a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d`.
* Recogniser: <https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx> at revision `3fafbc3b5dcf93dd72add9f48368be8a3a2cd33b`; `inference.onnx` sha256 `b5f833dfc5d0eb71da397b4efa06ebeee9b431b690a47d6af40d77d8eabc557f`.
* Pre- and post-processing: <https://github.com/PaddlePaddle/PaddleOCR> at commit `dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`.

## 3. Second readers

| Reader | Source | File used | sha256 |
|---|---|---|---|
| PARSeq-S\* (second reader of the paper) | our retraining of PARSeq-S without ArT, see below | `final_swa.ckpt` | `8f3e19b3339f7d305bef42b9468eece57b352277df7cb28c22eb7c00a3c58249` |
| PARSeq-S, released (contamination comparison only) | <https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt> | `parseq-bb5792a6.pt` | `bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00` |
| PARSeq source (runtime for both) | <https://github.com/baudm/parseq> at commit `1902db043c029a7e03a3818c616c06600af574be` | `parseq-1902db043c029a7e03a3818c616c06600af574be.tar.gz` | `f57289cb1fc7fe41f488a52ea48f8a7f114252ef405b0b64efc1d528c29b48ef` |
| ABINet (comparison) | code <https://github.com/FangShancheng/ABINet> at commit `789743c56eacbf114c2d63dfaeada533098dd19a`; weights from the README "Pretrained Models", Google Drive id `1mYM_26qHUom_5NU7iutHneB_KHlLjL5y` (`workdir.zip`) | `workdir/train-abinet/best-train-abinet.pth` | `3f0fe0d698c33ffcdc2968db0f567ab21b700157fbf9e46d9c40b04e9e3330f5` |
| ViTSTR-S (comparison) | <https://github.com/roatienza/deep-text-recognition-benchmark/releases/tag/v0.1.0> | `vitstr_small_patch16_224_aug.pth` | `3ff5c0490dabcb99b8540e66feb61d60e57821b55dabaacecbf31920824e42aa` |

ABINet and ViTSTR-S are trained on MJSynth and SynthText only.

### PARSeq-S\* (trained by us)

PARSeq-S\* is PARSeq-S retrained without ArT with the official PARSeq training script at the commit above, on 2,883,640 real word crops
(COCO-Text v2, LSVT, MLT19, OpenVINO, RCTW17, ReCTS, TextOCR, Uber-Text). The data are the real-data sets distributed by the PARSeq authors; `training/parseq_star/download_data.py`
lists the Google Drive files it downloads, and `training/parseq_star/run_train.sh` is the exact command (one GPU, batch size 384, two accumulation steps, 20 epochs, seed 20261106,
the averaged-weights checkpoint is saved as `final_swa.ckpt`).

## 4. Our own checkpoints

PARSeq-S\* (`final_swa.ckpt`, 381 MB) and the TiCLS CTW1500 fine-tune (`model_final.pth`) are not hosted anywhere yet. Their sha256 are given above so that a copy can be identified;
they are available from the corresponding author on request (see the Materials availability statement of the paper).
