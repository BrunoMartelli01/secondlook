"""Lossless submission boundaries and isolated, pinned upstream text evaluation.

This module supports the existing Python 3.8 evaluator environment. It does not
implement a replacement matching metric. See OFFICIAL-BENCHMARK-001."""

import hashlib

import importlib

import math

from pathlib import Path

import re

import sys

import types

import zipfile

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prediction_polygon(row, dataset, image_wh):
    """Retain native DeepSolo boundaries; the light branch emits axis-aligned boxes.

    IC15 requires quadrilaterals: use the published DeepSolo minimum-area box
    conversion for its 50-point boundaries. Never manufacture a curved contour
    for a light prediction. Ignore flags from the old evaluator are not inputs.
    """
    boundary = row.get("boundary_original")
    if boundary is None:
        x0, y0, x1, y1 = row["bbox"]
        points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    else:
        points = boundary
        if dataset == "icdar2015":
            import cv2
            import numpy as np
            corners = cv2.boxPoints(cv2.minAreaRect(np.asarray(points, dtype=np.float32))).astype(int)
            ps = sorted(corners.tolist(), key=lambda p: p[0])
            left = sorted(ps[:2], key=lambda p: p[1])
            right = sorted(ps[2:], key=lambda p: p[1])
            points = [left[0], right[0], right[1], left[1]]
            w, h = image_wh
            points = [[min(max(x, 1), w - 1), min(max(y, 1), h - 1)] for x, y in points]
    if len(points) < 3 or any(len(p) != 2 or not all(math.isfinite(float(v)) for v in p) for p in points):
        raise ValueError("Missing or nonfinite predicted geometry")
    # Upstream Total-Text validator expects clockwise Cartesian polygons.
    if dataset == "totaltext":
        signed = sum(points[i][0] * points[(i + 1) % len(points)][1] -
                     points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points)))
        if signed > 0:
            points = list(reversed(points))
    if dataset == "icdar2015" and len(points) != 4:
        raise ValueError("ICDAR2015 submissions require four corners")
    return points

def submission_line(points, text, dataset):
    if any(c in text for c in "\r\n"):
        raise ValueError("A prediction cannot contain line breaks")
    if dataset == "icdar2015":
        numbers = [str(int(v)) for p in points for v in p]
    else:
        # Official variable-polygon parser accepts coordinates as floats.
        numbers = [format(float(v), ".9g") for p in points for v in p]
    if dataset == "totaltext" and ",####" in text:
        raise ValueError("Transcription contains the upstream field delimiter")
    return ",".join(numbers) + (",####" if dataset == "totaltext" else ",") + text

def write_zip(path, files):
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in files.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, body.encode("utf-8"))

def load_upstream_evaluator(source_root, dataset):
    """Import the actual evaluator without loading Detectron2 or neural models."""
    package = types.ModuleType("adet")
    package.__path__ = [str(Path(source_root) / "adet")]
    sys.modules["adet"] = package
    evaluation = types.ModuleType("adet.evaluation")
    evaluation.__path__ = [str(Path(source_root) / "adet/evaluation")]
    sys.modules["adet.evaluation"] = evaluation
    suffix = "_ic15" if dataset == "icdar2015" else ""
    return importlib.import_module("adet.evaluation.text_eval_script" + suffix)

def parse_upstream_metrics(value):
    match = re.fullmatch(r"\w+: precision: ([\d.eE+-]+), recall: ([\d.eE+-]+), hmean: ([\d.eE+-]+)", value)
    if match is None:
        raise ValueError("Unexpected upstream result: " + value)
    return dict(zip(("precision", "recall", "hmean"), map(float, match.groups())))
