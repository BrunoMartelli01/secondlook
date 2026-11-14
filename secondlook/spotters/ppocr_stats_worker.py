"""E46 (approximate reproduction): the PP-OCR compact worker of secondlook.spotters.ppocr with ONE addition: per emitted character, the normalised entropy and the
margin (top1 - top2) of the CTC row are returned next to the unchanged fields (keys `char_entropy`, `char_margin`, aligned with `char_probs`).
`secondlook.spotters.ppocr.decode_ctc` is wrapped, never edited: every original field is produced by the original function and returned unchanged.
Run exactly like secondlook.spotters.ppocr:  python -m secondlook.spotters.ppocr_stats_worker --config <approximate config>  (same stdin/stdout JSON protocol)."""
import argparse
import math
from pathlib import Path

import numpy as np

import secondlook.spotters.ppocr as base

_original = base.decode_ctc


def decode_ctc_with_distribution_statistics(probabilities, characters, cutoffs):
    decoded = _original(probabilities, characters, cutoffs)
    p = np.asarray(probabilities, dtype=np.float64)
    tokens = p.argmax(1)
    selected = np.ones(len(tokens), bool)
    selected[1:] = tokens[1:] != tokens[:-1]
    selected &= tokens != 0                       # blank = class 0, exactly as in decode_ctc
    ix = np.flatnonzero(selected)
    if not len(ix):
        decoded['char_entropy'] = []
        decoded['char_margin'] = []
        return decoded
    rows = p[ix]
    values = rows[np.arange(len(ix)), tokens[ix]]
    second = np.partition(rows, -2, axis=1)[:, -2]
    entropy = -(rows * np.log(np.maximum(rows, 1e-15))).sum(1) / math.log(p.shape[1])
    if len(entropy) != len(decoded['char_probs']) or not np.allclose(values, decoded['char_probs']):
        raise ValueError('E46 worker: emitted characters differ from decode_ctc')
    decoded['char_entropy'] = entropy.tolist()
    decoded['char_margin'] = (values - second).tolist()
    return decoded


base.decode_ctc = decode_ctc_with_distribution_statistics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    base.worker(parser.parse_args().config)
