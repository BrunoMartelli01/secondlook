"""Audit pinned TiCLS assets and runtime prerequisites without installing anything.

This is qualification preparation, never a model-performance result. It can run
under the standard-library Python environment; checkpoint inspection is optional
and uses the separately specified legacy interpreter."""

import hashlib

from pathlib import Path

import shutil

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def check_free_space(paths, minimum_free_bytes):
    if type(minimum_free_bytes) is not int or minimum_free_bytes <= 0:
        raise ValueError("A positive explicit storage reserve is required")
    records = []
    for path in paths:
        path = Path(path)
        usage = shutil.disk_usage(path)
        if usage.free < minimum_free_bytes:
            raise RuntimeError("Insufficient disk space at {}: {} free bytes; require {}. "
                               "Free old run artifacts or choose another qualified path before execution."
                               .format(path, usage.free, minimum_free_bytes))
        records.append(dict(path=str(path), free_bytes=usage.free, total_bytes=usage.total,
                            minimum_free_bytes=minimum_free_bytes))
    return records
