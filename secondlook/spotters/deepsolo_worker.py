"""DeepSolo-environment worker. This module is kept parseable by Python 3.8."""

from __future__ import annotations

import pickle

from pathlib import Path

from typing import Any, Dict, Tuple

LABELS_37 = list("abcdefghijklmnopqrstuvwxyz0123456789")

LABELS_96 = list(
    " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz{|}~"
)

VITAE_PAD_DIVISOR = 32

def _decode_ctc(recognition: Any, vocabulary_size: int, custom_dictionary: str) -> str:
    if vocabulary_size == 37:
        labels = LABELS_37
    elif vocabulary_size == 96:
        labels = LABELS_96
    else:
        if not custom_dictionary:
            raise RuntimeError("DeepSolo custom vocabulary has no dictionary path")
        with Path(custom_dictionary).open("rb") as stream:
            labels = pickle.load(stream)
    if len(labels) != vocabulary_size - 1:
        raise RuntimeError("DeepSolo vocabulary size does not match its dictionary")
    output, previous = "", None
    for raw_value in recognition:
        value = int(raw_value)
        if value < vocabulary_size - 1:
            if value != previous:
                label = labels[value]
                output += str(label) if vocabulary_size in (37, 96) else chr(label)
                previous = value
        else:
            previous = None
    # Preserve the frozen evaluator's raw CTC decode. Its visualization path
    # uppercases 37-character output for display only; this contract does not.
    return output

def _as_points(value: Any) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.size == 0:
        return []
    return array.reshape(-1, 2).astype(float).tolist()

def _boundary_polygon(value: Any) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.size == 0:
        return []
    if array.ndim == 2 and array.shape[1] % 2 == 0:
        first, second = np.hsplit(array, 2)
        return np.vstack([first, second[::-1]]).reshape(-1, 2).astype(float).tolist()
    return array.reshape(-1, 2).astype(float).tolist()

def _load_model_and_validate(cfg: Any, checkpoint_path: str) -> Tuple[Any, Dict[str, Any]]:
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model

    # Registration imports are required by the frozen upstream source.
    from adet.modeling import swin, vitae_v2  # noqa: F401

    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpoint = checkpointer._load_file(checkpoint_path)
    incompatible = checkpointer._load_model(checkpoint)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    incorrect = [
        {"key": item[0], "checkpoint_shape": list(item[1]), "model_shape": list(item[2])}
        for item in incompatible.incorrect_shapes
    ]
    if missing or unexpected or incorrect:
        raise RuntimeError(
            "DeepSolo checkpoint load was incompatible: missing={}, unexpected={}, incorrect_shapes={}".format(
                missing, unexpected, incorrect
            )
        )
    total_parameters = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable_parameters = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    return model, {
        "strict_compatibility_check": True,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "incorrect_shapes": incorrect,
        "model_class": type(model).__name__,
        "backbone_builder": str(cfg.MODEL.BACKBONE.NAME),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }

def _apply_frozen_vitae_padding(image: Any, transforms: Any) -> Any:
    """Apply frozen upstream Pad semantics without importing ``adet.data``."""
    height, width = image.shape[:2]
    height_remainder = height % VITAE_PAD_DIVISOR
    width_remainder = width % VITAE_PAD_DIVISOR
    pad_h = 0 if height_remainder == 0 else VITAE_PAD_DIVISOR - height_remainder
    pad_w = 0 if width_remainder == 0 else VITAE_PAD_DIVISOR - width_remainder
    transform = transforms.PadTransform(0, 0, pad_w, pad_h, pad_value=0)
    return transform.apply_image(image)

def _predict(model: Any, cfg: Any, image_bgr: Any, is_vitae: bool) -> Any:
    import numpy as np
    import torch
    import detectron2.data.transforms as transforms

    height, width = image_bgr.shape[:2]
    image = image_bgr
    if cfg.INPUT.FORMAT == "RGB":
        image = image[:, :, ::-1]
    augmentation = transforms.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )
    image = augmentation.get_transform(image).apply_image(image)
    if is_vitae:
        image = _apply_frozen_vitae_padding(image, transforms)
    tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    with torch.no_grad():
        return model([{"image": tensor, "height": height, "width": width}])[0]

def _serialize_instances(instances: Any, cfg: Any) -> Any:
    instances = instances.to("cpu")
    required_fields = (
        "scores", "pred_classes", "recs", "rec_scores", "bd", "ctrl_points",
    )
    missing = [field for field in required_fields if not instances.has(field)]
    if missing:
        raise RuntimeError("DeepSolo output is missing expected fields: {}".format(missing))
    scores = instances.scores.tolist()
    predicted_classes = instances.pred_classes.tolist()
    recognitions = instances.recs.numpy()
    recognition_scores = instances.rec_scores.numpy()
    boundaries = instances.bd.numpy()
    control_points = instances.ctrl_points.numpy()
    vocabulary_size = int(cfg.MODEL.TRANSFORMER.VOC_SIZE)
    custom_dictionary = str(cfg.MODEL.TRANSFORMER.CUSTOM_DICT)
    detections = []
    for index, (score, predicted_class, rec, rec_score, boundary, controls) in enumerate(
        zip(
            scores,
            predicted_classes,
            recognitions,
            recognition_scores,
            boundaries,
            control_points,
        )
    ):
        text = _decode_ctc(rec, vocabulary_size, custom_dictionary)
        detections.append({
            "source_index": index,
            "text": text,
            "score": float(score),
            "score_type": "DeepSolo instance score",
            "geometry": {
                "type": "DeepSolo boundary polygon and center-line control points",
                "boundary_points": _boundary_polygon(boundary),
                "control_points": _as_points(controls),
            },
            "raw": {
                "predicted_class": int(predicted_class),
                "recognition_token_ids": [int(value) for value in rec.tolist()],
                "recognition_token_confidences": [
                    float(value) for value in rec_score.max(axis=-1).tolist()
                ],
                "recognition_class_probabilities": rec_score.tolist(),
                "boundary": boundary.tolist(),
                "control_points": controls.tolist(),
            },
        })
    return detections
