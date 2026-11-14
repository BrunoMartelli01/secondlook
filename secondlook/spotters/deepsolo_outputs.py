"""GT-free DeepSolo Stage-1 router and routed TiCLS fallback/fusion primitives.

All functions in this module are strictly annotation-blind (GT-free) at feature
extraction and inference time, compatible with both Python 3.8 and Python 3.11,
and enforce strict isolation between Weak-lexicon and None (lexicon-free) modes."""

import math

import cv2

import numpy as np

from secondlook.spotters.deepsolo_boundaries import EmptyDetectorBoundaryError, boundary_bbox, detector_boundary_in_original

from secondlook.evaluation.official_benchmarks import prediction_polygon, submission_line

def _signed_double_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    return float(
        sum(
            points[i][0] * points[(i + 1) % n][1]
            - points[(i + 1) % n][0] * points[i][1]
            for i in range(n)
        )
    )

def _polygon_area(points):
    return abs(_signed_double_area(points)) * 0.5

def _segments_intersect_or_touch(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_seg(p, q, r):
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (
        o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0
    ):
        return True
    if o1 == 0 and on_seg(a, c, b):
        return True
    if o2 == 0 and on_seg(a, d, b):
        return True
    if o3 == 0 and on_seg(c, a, d):
        return True
    if o4 == 0 and on_seg(c, b, d):
        return True
    return False

def is_valid_simple_polygon(int_points):
    """Use Shapely Polygon.is_valid when installed; exact planar check otherwise."""
    pts = [(int(p[0]), int(p[1])) for p in int_points]
    if len(pts) < 3:
        return False
    try:
        from shapely.geometry import Polygon

        return bool(Polygon(pts).is_valid)
    except ImportError:
        pass
    dedup = []
    for p in pts:
        if not dedup or dedup[-1] != p:
            dedup.append(p)
    if len(dedup) > 1 and dedup[0] == dedup[-1]:
        dedup.pop()
    if len(dedup) < 3 or _polygon_area(dedup) <= 0.0:
        return False
    if len(set(dedup)) != len(dedup):
        return False
    n = len(dedup)
    for i in range(n):
        a1, a2 = dedup[i], dedup[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or j == (i + 1) % n or (j + 1) % n == i:
                continue
            b1, b2 = dedup[j], dedup[(j + 1) % n]
            if _segments_intersect_or_touch(a1, a2, b1, b2):
                return False
    return True

def is_ccw_ring(int_points):
    pts = [(int(p[0]), int(p[1])) for p in int_points]
    try:
        from shapely.geometry import LinearRing

        return bool(LinearRing(pts).is_ccw)
    except ImportError:
        return _signed_double_area(pts) > 0.0

def ctc_token_statistics(probabilities):
    """Extract greedy CTC character confidence, margin, and entropy statistics."""
    if not probabilities:
        raise ValueError("Nonempty 38-class DeepSolo CTC probability matrix required")
    log_denom = math.log(38.0)
    emitted_probs = []
    emitted_margins = []
    emitted_entropies = []
    blank_steps = 0
    previous = None
    for row in probabilities:
        if len(row) != 38:
            raise ValueError("Expected 38-class DeepSolo CTC probabilities")
        arr = [float(v) for v in row]
        token = max(range(38), key=arr.__getitem__)
        if token >= 36:
            blank_steps += 1
            previous = None
            continue
        if token != previous:
            p1 = max(min(arr[token], 1.0), 1e-12)
            second = max(arr[k] for k in range(38) if k != token)
            ent = -sum(p * math.log(p) for p in arr if p > 1e-15) / log_denom
            emitted_probs.append(float(arr[token]))
            emitted_margins.append(float(max(0.0, arr[token] - second)))
            emitted_entropies.append(float(min(max(ent, 0.0), 1.0)))
            log_p_check = math.log(p1)
            if not math.isfinite(log_p_check):
                raise ValueError("Nonfinite CTC log-probability")
        previous = token
    blank_fraction = float(blank_steps) / float(len(probabilities))
    if not emitted_probs:
        return dict(
            recognition_confidence=0.0,
            char_probs=[],
            min_char_prob=0.0,
            mean_char_log_prob=-12.0,
            min_char_margin=0.0,
            mean_char_margin=0.0,
            max_char_entropy=1.0,
            mean_char_entropy=1.0,
            blank_fraction=blank_fraction,
        )
    logs = [math.log(max(p, 1e-12)) for p in emitted_probs]
    mean_log = sum(logs) / len(logs)
    return dict(
        recognition_confidence=float(math.exp(mean_log)),
        char_probs=emitted_probs,
        min_char_prob=float(min(emitted_probs)),
        mean_char_log_prob=float(mean_log),
        min_char_margin=float(min(emitted_margins)),
        mean_char_margin=float(sum(emitted_margins) / len(emitted_margins)),
        max_char_entropy=float(max(emitted_entropies)),
        mean_char_entropy=float(sum(emitted_entropies) / len(emitted_entropies)),
        blank_fraction=blank_fraction,
    )

def extract_compact_deepsolo_sample(raw_payload, dataset, sample_id, image_wh, orientation, image_sha256):
    """Convert a raw 0.05-floor DeepSolo response into a compact GT-free record."""
    if raw_payload.get("ground_truth_used") is True:
        raise ValueError("DeepSolo raw payload must never consume ground truth")
    contract = raw_payload.get("contract", {})
    if contract.get("image_sha256") != image_sha256:
        raise ValueError("DeepSolo raw payload image_sha256 mismatch for " + str(sample_id))
    if not raw_payload.get("response", {}).get("success", False):
        raise ValueError("DeepSolo raw response did not succeed for " + str(sample_id))
    if "inference_floor" in raw_payload and float(raw_payload["inference_floor"]) != 0.05:
        raise ValueError("Expected 0.05-floor DeepSolo export for " + str(sample_id))
    seconds = raw_payload.get("seconds", raw_payload.get("inference_seconds"))
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) <= 0.0:
        raise ValueError("Missing positive DeepSolo per-image timing for " + str(sample_id))
    wh_tuple = (int(image_wh[0]), int(image_wh[1]))
    detections = []
    seen_indices = set()
    for det in raw_payload["response"]["detections"]:
        idx = int(det["source_index"])
        if idx in seen_indices:
            raise ValueError("Duplicate DeepSolo source_index in " + str(sample_id))
        seen_indices.add(idx)
        boundary_original = detector_boundary_in_original(
            det["geometry"]["boundary_points"], wh_tuple, int(orientation)
        )
        try:
            bbox = boundary_bbox(boundary_original, wh_tuple)
            invalid_envelope = None
        except EmptyDetectorBoundaryError as err:
            bbox = None
            invalid_envelope = dict(
                unclipped_bbox=err.unclipped_bbox,
                clipped_bbox=err.clipped_bbox,
            )
        stats = ctc_token_statistics(det["raw"]["recognition_class_probabilities"])
        proj_points = prediction_polygon(
            dict(boundary_original=boundary_original), dataset, wh_tuple
        )
        int_points = [(int(float(x)), int(float(y))) for x, y in proj_points]
        native_valid = is_valid_simple_polygon(int_points)
        if native_valid and dataset == "totaltext" and is_ccw_ring(int_points):
            int_points = list(reversed(int_points))
        hull_arr = cv2.convexHull(np.asarray(int_points, dtype=np.int32)).reshape(-1, 2)
        hull_points = [(int(p[0]), int(p[1])) for p in hull_arr.tolist()]
        hull_valid = len(hull_points) >= 3 and is_valid_simple_polygon(hull_points)
        if hull_valid and dataset == "totaltext" and is_ccw_ring(hull_points):
            hull_points = list(reversed(hull_points))
        detections.append(
            dict(
                instance_id=idx,
                text=str(det["text"]),
                score=float(det["score"]),
                bbox=bbox,
                invalid_envelope=invalid_envelope,
                int_points=[[p[0], p[1]] for p in int_points],
                native_polygon_valid=bool(native_valid),
                hull_points=[[p[0], p[1]] for p in hull_points] if hull_valid else None,
                hull_polygon_valid=bool(hull_valid),
                polygon_area=float(_polygon_area(int_points)) if native_valid else (
                    float(_polygon_area(hull_points)) if hull_valid else 0.0
                ),
                **stats,
            )
        )
    return dict(
        sample_id=sample_id,
        dataset=dataset,
        image_wh=[wh_tuple[0], wh_tuple[1]],
        orientation=int(orientation),
        image_sha256=image_sha256,
        seconds=float(seconds),
        ground_truth_used=False,
        detections=detections,
    )

def _enforce_lexicon_boundary(lexicon_mode, weak_lexicon):
    if lexicon_mode not in ("None", "Weak"):
        raise ValueError("Unsupported lexicon_mode: " + str(lexicon_mode))
    if lexicon_mode == "None" and weak_lexicon is not None:
        raise ValueError("Weak lexicon object is strictly forbidden in None lexicon mode")
    if lexicon_mode == "Weak" and weak_lexicon is None:
        raise ValueError("Weak lexicon object is required in Weak lexicon mode")

def format_deepsolo_variant(compact_sample, dataset, variant_spec, lexicon_mode, weak_lexicon=None):
    """Format a compact DeepSolo sample under a declared Stage-1 admission policy."""
    _enforce_lexicon_boundary(lexicon_mode, weak_lexicon)
    score_th = float(variant_spec["score_threshold"])
    rec_th = float(variant_spec["recognition_threshold"])
    poly_policy = variant_spec.get("invalid_polygon_policy", "drop")
    if poly_policy not in ("drop", "convex_hull"):
        raise ValueError("Unsupported invalid_polygon_policy: " + str(poly_policy))
    ordered = sorted(
        compact_sample["detections"], key=lambda d: (-d["score"], d["instance_id"])
    )
    n_rec_rejected = 0
    score_passed = []
    for det in ordered:
        if det["score"] < score_th:
            continue
        if det["recognition_confidence"] < rec_th:
            n_rec_rejected += 1
            continue
        raw_text = "" if det["invalid_envelope"] is not None else det["text"]
        raw_text = raw_text.replace("\r", "").replace("\n", "")
        if dataset == "totaltext" and ",####" in raw_text:
            raw_text = raw_text.replace(",####", "")
        score_passed.append((det, raw_text))

    n_weak_matched = 0
    n_weak_exact = 0
    n_weak_rejected = 0
    lexicon_filtered = []
    if lexicon_mode == "Weak":
        for det, raw_text in score_passed:
            match = weak_lexicon.nearest(raw_text)
            if match is None:
                n_weak_rejected += 1
                continue
            n_weak_matched += 1
            if match["distance"] == 0:
                n_weak_exact += 1
            lexicon_filtered.append((det, raw_text, match["text"], match["distance"]))
    else:
        for det, raw_text in score_passed:
            lexicon_filtered.append((det, raw_text, raw_text, None))

    lines = []
    admitted_rows = []
    n_invalid_polygons = 0
    for det, raw_text, final_text, weak_dist in lexicon_filtered:
        if det["native_polygon_valid"]:
            chosen_points = det["int_points"]
        elif poly_policy == "convex_hull" and det["hull_polygon_valid"]:
            chosen_points = det["hull_points"]
        else:
            n_invalid_polygons += 1
            continue
        xs = [p[0] for p in chosen_points]
        ys = [p[1] for p in chosen_points]
        poly_bbox = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
        lines.append(submission_line(chosen_points, final_text, dataset))
        admitted_rows.append(
            dict(
                instance_id=det["instance_id"],
                score=det["score"],
                raw_text=raw_text,
                submission_text=final_text,
                weak_distance=weak_dist,
                points=chosen_points,
                bbox=det["bbox"] if det["bbox"] is not None else poly_bbox,
                polygon_area=float(_polygon_area(chosen_points)),
                recognition_confidence=det["recognition_confidence"],
                char_probs=det["char_probs"],
                min_char_prob=det["min_char_prob"],
                mean_char_log_prob=det["mean_char_log_prob"],
                min_char_margin=det["min_char_margin"],
                mean_char_margin=det["mean_char_margin"],
                max_char_entropy=det["max_char_entropy"],
                mean_char_entropy=det["mean_char_entropy"],
                blank_fraction=det["blank_fraction"],
            )
        )
    body = "\n".join(lines) + ("\n" if lines else "")
    return dict(
        body=body,
        admitted_rows=admitted_rows,
        rejected_by_rec_threshold=n_rec_rejected,
        rejected_invalid_polygons=n_invalid_polygons,
        score_passed_count=len(score_passed),
        weak_stats=dict(
            matched=n_weak_matched,
            exact=n_weak_exact,
            rejected=n_weak_rejected,
        ),
    )
