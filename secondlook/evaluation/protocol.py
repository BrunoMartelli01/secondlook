"""Auditable fixed-budget scheduling, exact image oracles and paired spotting metrics."""

from fractions import Fraction

import hashlib

import json

from pathlib import Path

import shutil

import numpy as np

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, value, exclusive=True):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

def load_config(path):
    p = Path(path).resolve()
    cfg = read_json(p)
    if p.parent.name != "approximate" or cfg.get("mode") != "approximate":
        raise ValueError("Explicit configs/approximate configuration required")
    if cfg.get("claim_label") != "approximate reproduction":
        raise ValueError("Explicit approximate reproduction claim label required")
    for name, expected in cfg["pinned_inputs"].items():
        if sha256(name) != expected:
            raise ValueError("Changed pinned study input: " + name)
    return cfg

def require_space(path, minimum_free_bytes):
    p = Path(path)
    while not p.exists():
        p = p.parent
    free = shutil.disk_usage(p).free
    if free < minimum_free_bytes:
        raise RuntimeError("Storage reserve unavailable: {} < {} bytes on {}".format(
            free, minimum_free_bytes, p))
    return free

def hmean_fraction(counts, detection=False):
    row = np.asarray(counts, dtype=np.int64)
    if row.ndim == 2:
        row = row.sum(axis=0)
    j = 4 if detection else 0
    return Fraction(2 * int(row[j]), int(row[j + 1] + row[j + 2])) if row[j + 1] + row[j + 2] else Fraction(0)
