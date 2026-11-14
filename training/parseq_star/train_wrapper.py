"""E33: official PARSeq training script (commit 1902db04...) run on ONE GPU with the 2-GPU official effective setup (documented in the study log, E33).
Only three things are added to the unmodified official train.py, both asserted to apply exactly once:
 1. seed_everything(20261106, workers=True) before anything else;
 2. swa_lr is computed from the OFFICIAL base lr 7e-4 (the official run uses model.lr = 7e-4 for it); here model.lr is rescaled to give the official peak lr on one device;
 3. after trainer.fit the model (now holding the SWA-averaged weights) is saved as checkpoints/final_swa.ckpt."""
import atexit, os, runpy, sys
import torch
import pytorch_lightning as pl
from secondlook.paths import ARTIFACTS

SRC = str(ARTIFACTS / 'parseq/parseq-1902db043c029a7e03a3818c616c06600af574be')
sys.path.insert(0, SRC)
os.chdir(SRC)
pl.seed_everything(20261106, workers=True)
torch.set_float32_matmul_precision('high')
text = open('train.py').read()
old = 'swa_lr = config.model.lr * get_swa_lr_factor('
assert text.count(old) == 1
text = text.replace(old, 'swa_lr = 7e-4 * get_swa_lr_factor(')
fit = 'trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)'
assert text.count(fit) == 1
# 3. after fit, StochasticWeightAveraging has copied the averaged weights into the model: save them explicitly (the official script's 'last' checkpoint predates that copy)
text = text.replace(fit, fit + "\n    trainer.save_checkpoint(str(Path(cwd) / 'checkpoints' / 'final_swa.ckpt'))")
atexit.register(lambda: print('PEAK allocated MiB', torch.cuda.max_memory_allocated() / 2**20, 'reserved MiB', torch.cuda.max_memory_reserved() / 2**20, flush=True))
sys.argv = ['train.py'] + sys.argv[1:]
exec(compile(text, 'train.py', 'exec'), {'__name__': '__main__', '__file__': SRC + '/train.py'})
