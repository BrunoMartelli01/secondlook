"""Versioned dump of DeepSolo per-point CTC posteriors for decoding-level methods (schema ``deepsolo-posteriors/2``).

READY, NOT RUN on real data (needs the GPU window after E17). The dump reuses the ``pred['instances']`` that
``research/ctw1500_official.infer`` already computes (``rec_scores`` = softmax [N,50,97], ``recs`` = argmax ids [N,50],
``ctrl_points`` = centre-curve coordinates in original-image pixels [N,100], ``bd`` = boundary polygon [N,200] or None); no model change,
no extra forward pass. Schema v2 follows the reviewer's sanity checks: floor 0.05 band included, LOG-probabilities (fp16 probabilities underflow
in beam scores), per point only the top-8 character classes + blank + the leftover tail mass, boundary polygon and image id per candidate.

Files per split: ``posteriors_<split>.npz`` + ``posteriors_<split>.manifest.json`` (schema, sha256 of the npz, counts, label table, rec strings, provenance).
npz arrays (all candidates with score >= ``floor``, in image-id/candidate order), M candidates, P=50 points:
  image_id int64 [M]; cand_index int32 [M] (position in that image's candidate list of cands_<split>.json); score float32 [M];
  ids int16 [M,P] (argmax class; labels = upstream CTLABELS (95 characters), ids >= 95 decode as blank);
  top_ids int16 [M,P,8] and top_logp float16 [M,P,8] (the 8 most probable CHARACTER classes, ids < 95, natural log);
  blank_logp float16 [M,P] (log of the blank mass = classes 95 and 96 together); tail_logp float16 [M,P] (log of the mass outside top-8 and blank);
  points float32 [M,P,2]; boundary float32 [M,200] (NaN rows when the model has no boundary head).
Size ~2.6 KB per candidate uncompressed: TEST 8830 (score >= 0.05) ~23 MB, TRAIN ~25k ~65 MB, within the 60/120 MB budget.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'deepsolo-posteriors/2'
POINTS, CLASSES, TOPK, CHARS, BD = 50, 97, 8, 95, 200
EPS = 1e-30


class Dump:
    def __init__(self, floor=0.05):
        self.floor = floor
        self.rows = {k: [] for k in ('image_id', 'cand_index', 'score', 'ids', 'top_ids', 'top_logp', 'blank_logp', 'tail_logp', 'points', 'boundary')}
        self.rec = []

    def add_image(self, image_id, scores, rec_scores, recs, ctrl_points, rec_strings, boundary=None):
        """Arrays for ALL instances of one image (same order as the candidate list written by the inference stage)."""
        scores = np.asarray(scores, np.float32)
        probs = np.asarray(rec_scores, np.float64)
        ids = np.asarray(recs)
        n = len(scores)
        pts = np.asarray(ctrl_points, np.float32).reshape(n, POINTS, 2)
        if probs.shape != (n, POINTS, CLASSES) or ids.shape != (n, POINTS) or len(rec_strings) != n:
            raise ValueError('Unexpected posterior shapes: %s %s' % (probs.shape, ids.shape))
        bd = np.full((n, BD), np.nan, np.float32) if boundary is None else np.asarray(boundary, np.float32).reshape(n, BD)
        chars = probs[..., :CHARS]
        blank = probs[..., CHARS:].sum(-1)
        order = np.argsort(-chars, axis=-1)[..., :TOPK]
        top = np.take_along_axis(chars, order, axis=-1)
        tail = np.clip(1.0 - top.sum(-1) - blank, 0.0, 1.0)
        for j in range(n):
            if scores[j] < self.floor:
                continue
            for k, v in (('image_id', int(image_id)), ('cand_index', j), ('score', scores[j]), ('ids', ids[j].astype(np.int16)), ('top_ids', order[j].astype(np.int16)),
                         ('top_logp', np.log(np.maximum(top[j], EPS)).astype(np.float16)), ('blank_logp', np.log(np.maximum(blank[j], EPS)).astype(np.float16)),
                         ('tail_logp', np.log(np.maximum(tail[j], EPS)).astype(np.float16)), ('points', pts[j]), ('boundary', bd[j])):
                self.rows[k].append(v)
            self.rec.append(rec_strings[j])

    def write(self, directory, split, labels, provenance):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        r = self.rows
        arrays = dict(image_id=np.asarray(r['image_id'], np.int64), cand_index=np.asarray(r['cand_index'], np.int32), score=np.asarray(r['score'], np.float32),
                      ids=np.asarray(r['ids'], np.int16).reshape(-1, POINTS), top_ids=np.asarray(r['top_ids'], np.int16).reshape(-1, POINTS, TOPK),
                      top_logp=np.asarray(r['top_logp'], np.float16).reshape(-1, POINTS, TOPK), blank_logp=np.asarray(r['blank_logp'], np.float16).reshape(-1, POINTS),
                      tail_logp=np.asarray(r['tail_logp'], np.float16).reshape(-1, POINTS), points=np.asarray(r['points'], np.float32).reshape(-1, POINTS, 2),
                      boundary=np.asarray(r['boundary'], np.float32).reshape(-1, BD))
        npz = directory / ('posteriors_%s.npz' % split)
        np.savez_compressed(npz, **arrays)
        manifest = dict(schema=SCHEMA, split=split, floor=self.floor, count=int(len(arrays['score'])), npz=npz.name, npz_sha256=sha256(npz), npz_bytes=npz.stat().st_size,
                        labels=list(labels), blank=len(labels), rec=self.rec, provenance=provenance)
        (directory / ('posteriors_%s.manifest.json' % split)).write_text(json.dumps(manifest))
        return manifest


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(directory, split):
    directory = Path(directory)
    manifest = json.loads((directory / ('posteriors_%s.manifest.json' % split)).read_text())
    if manifest['schema'] != SCHEMA:
        raise ValueError('Unsupported schema ' + manifest['schema'])
    npz = directory / manifest['npz']
    if sha256(npz) != manifest['npz_sha256']:
        raise ValueError('Corrupt posterior dump: sha256 mismatch')
    return manifest, dict(np.load(npz))


def to_dense(arrays, i):
    """[P, CHARS+1] probabilities of candidate i for the beam search: top-8 as stored, blank, and the tail mass spread uniformly over the remaining
    character classes (documented approximation; rows are renormalised to sum to 1)."""
    p = np.zeros((POINTS, CHARS + 1), np.float64)
    top = np.exp(arrays['top_logp'][i].astype(np.float64))
    for t in range(POINTS):
        p[t, arrays['top_ids'][i, t]] = top[t]
    rest = p[:, :CHARS] == 0
    tail = np.exp(arrays['tail_logp'][i].astype(np.float64))
    p[:, :CHARS] += rest * (tail / np.maximum(rest.sum(-1), 1))[:, None]
    p[:, CHARS] = np.exp(arrays['blank_logp'][i].astype(np.float64))
    return p / p.sum(-1, keepdims=True)


def verify_parity(manifest, arrays, greedy_decode):
    """Best-path decoding of the stored argmax ids must reproduce every stored rec string; also reports whether the stored log-probs agree with the ids."""
    labels = manifest['labels']
    bad = [i for i in range(len(manifest['rec'])) if greedy_decode(arrays['ids'][i], labels) != manifest['rec'][i]]
    m = len(arrays['ids'])
    if m:
        top1 = arrays['top_ids'][..., 0]
        char_pts = arrays['ids'] < CHARS
        agree = float((top1[char_pts] == arrays['ids'][char_pts]).mean()) if char_pts.any() else 1.0
    else:
        agree = 1.0
    return dict(candidates=len(manifest['rec']), id_decode_mismatches=bad, top1_agrees_with_argmax_on_character_points=agree)
