"""E33 stage 2 (main env, CPU or GPU): PARSeq-S-CLEAN (PARSeq-S retrained by us on the official real set minus ArT, the study log E33) reads the cached crops of
stage 1. Same reading statistics and file format as research/p3_parseq.py, but the readings are written to --out-dir (never next to the PARSeq-S files, whose names are identical).

The qualified adapter (recognition/parseq.py) pins the released checkpoint's name and SHA256, so it cannot load another checkpoint and is left untouched. This script builds the same
model from the same pinned runtime source, loads OUR checkpoint with strict=True, records its SHA256 and borrows the adapter's `recognize_candidate_batch`.
Usage: python -m secondlook.second_stage.read_parseq_star --checkpoint <final_swa.ckpt> --out-dir <dir> --files <crops .npz>..."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from secondlook.second_stage.parseq import ParseqApproxRecognizer, ParseqApproxConfig, PARSEQ_ARCHIVE_SOURCE, verify_parseq_runtime_source
from secondlook.paths import ARTIFACTS

ART = (ARTIFACTS / 'parseq')
SOURCE = ART / 'parseq-1902db043c029a7e03a3818c616c06600af574be'
ARCHIVE = ART / 'parseq-1902db043c029a7e03a3818c616c06600af574be.tar.gz'


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_recognizer(checkpoint: Path, device: str):
    import torch
    identity = verify_parseq_runtime_source(SOURCE, source_type=PARSEQ_ARCHIVE_SOURCE, source_archive=ARCHIVE)
    sys.path.insert(0, str(SOURCE))
    try:
        from strhub.data.module import SceneTextDataModule
        from strhub.models.utils import create_model
        model = create_model('parseq', pretrained=False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = state.get('state_dict', state)                         # Lightning checkpoint -> the system's state dict
        inner = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
        loaded = model.model.load_state_dict(inner, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError('Strict PARSeq load was incomplete: %s' % (loaded,))
        transform = SceneTextDataModule.get_transform(tuple(model.hparams.img_size))
    finally:
        sys.path.remove(str(SOURCE))
    rec = ParseqApproxRecognizer._from_test_components(model.eval().to(device), transform)   # the adapter's methods, our weights
    rec.config = ParseqApproxConfig(runtime_source=SOURCE, checkpoint=checkpoint, device=device)
    rec._provenance = {'status': 'approximate reproduction', 'checkpoint': checkpoint.name, 'checkpoint_sha256': sha256_of(checkpoint), **identity,
                       'strict_state_dict_load': True, 'decode_ar': True, 'refine_iters': 1, 'trained_by_us': 'real minus ArT, E33'}
    return rec


def main(a):
    import torch
    torch.set_num_threads(a.threads)
    rec = load_recognizer(Path(a.checkpoint), a.device)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'provenance.json').write_text(json.dumps(rec._provenance, indent=1, default=str))
    for f in a.files:
        d = np.load(f)
        crops = d['crops']
        out = []
        for i in range(0, len(crops), a.batch):
            for r in rec.recognize_candidate_batch(list(crops[i:i + a.batch])):
                out.append(dict(text=r.raw_text, seq=r.sequence_confidence_native, mint=r.minimum_token_probability, meanlp=r.mean_token_log_probability, n=r.predicted_length))
            if (i // a.batch) % 20 == 0:
                print(Path(f).name, i, len(crops), flush=True)
        target = out_dir / Path(str(f).replace('.npz', '_parseq.json')).name
        target.write_text(json.dumps(dict(owner=d['owner'].tolist(), cidx=d['cidx'].tolist(), rows=out)))
        print('done', target.name, len(out), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--files', nargs='+', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch', type=int, default=64)
    main(p.parse_args())
