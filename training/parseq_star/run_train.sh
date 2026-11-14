#!/bin/bash
# E33 PARSeq-S-CLEAN training: real minus ArT, one RTX 3090. Usage: run_train.sh
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
A=${SECONDLOOK_ARTIFACTS:?set SECONDLOOK_ARTIFACTS}/parseq-retrain
PY=${SECONDLOOK_READER_PYTHON:-python}
export PYTHONPATH=$A/pydeps2:$A/pydeps:$ROOT
cd $A
$PY train_wrapper.py +experiment=parseq dataset=real data.root_dir=$A/data trainer.accelerator=gpu trainer.devices=1 \
  model.batch_size=384 +trainer.accumulate_grad_batches=2 model.lr=4.949747468305833e-4 trainer.val_check_interval=500 trainer.max_epochs=20 data.num_workers=8 \
  hydra.run.dir=$A/outputs/parseq-clean-real-minus-art
