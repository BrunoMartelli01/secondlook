"""APPROXIMATE low-floor/resolution study, using separate existing Python runtimes.

Run export in legacy DeepSolo Python; evaluate in main Python. Source datasets,
checkpoint/runtime, native preprocessing and denominators are verified explicitly."""

from __future__ import annotations

def configure_model_resize(model, inference):
    """Synchronize ViTAE's cached inverse resize with actual input augmentation.

    Frozen TransformerPureDetector stores these at construction, rather than
    reading the supplied inference cfg in forward(). Both must change together.
    """
    if not hasattr(model, 'min_size_test') or not hasattr(model, 'max_size_test'):
        raise ValueError('Expected frozen ViTAE postprocessing resize attributes')
    model.min_size_test = inference.INPUT.MIN_SIZE_TEST
    model.max_size_test = inference.INPUT.MAX_SIZE_TEST
    return [int(model.min_size_test), int(model.max_size_test)]
