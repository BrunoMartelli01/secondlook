"""E17 evaluation: ONE evaluation of a fine-tuned TiCLS CTW1500 checkpoint on all 500 official TEST images with the UNMODIFIED TiCLS evaluator.

Frozen plan (the study log, E17/E24): only ``model_final.pth`` is evaluated, once; no checkpoint is chosen on TEST. Lexicon None and the official
"Full" (weak) lexicon are reported separately; the paper reference is 66.2 / 88.1. "Near" = None >= 64.7 AND Full >= 86.1, otherwise the result is labelled
'own fine-tune, not comparable' and no head is calibrated.

Same machinery as ``research/ticls_ctw_finetune.py`` (reuses its runtime ``initialize``: legacy venv, pinned deformable-attention extension, strict PLM decoder load)
and as ``research/ctw1500_official.Official`` (TiCLS source: its own ``to_eval_format`` / ``sort_detection`` / ``text_eval_script``, per-image counters).
Run on the GPU only after 'GPU RELEASED (E17)':
  PYTHONPATH=. python -m secondlook.spotters.ticls_ctw1500_eval --runtime-config <totaltext resolved_configuration.json> \
      --checkpoint <E17>/full/train/model_final.pth --annotations <E12 assets> --images <E17>/full/data/ctw1500/ctw1500/test_images \
      --assets <E11 assets> --out <dir>
Result class: approximate reproduction (own fine-tune from the published Total-Text pretrain; not the authors' CTW checkpoint). FP32, batch 1.
"""
import argparse
import hashlib
import importlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def resolve_image(images_dir, name):
    path = Path(images_dir) / name
    if not path.exists():
        raise FileNotFoundError('Staged test image missing: ' + str(path))
    return path


def read_image(path, height, width):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None and image.shape[:2] != (height, width):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None or image.shape[:2] != (height, width):
        raise ValueError('Unreadable raster or size differs from the annotation: ' + str(path))
    return image


def main(args):
    from secondlook.spotters.ticls_ctw1500_finetune import initialize
    runtime_cfg = json.load(open(args.runtime_config))
    source = Path(runtime_cfg['expert']['asset_root']) / ('TiCLS-' + runtime_cfg['expert']['source_revision'])
    args.seed = 20261105
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    observed = initialize(runtime_cfg, source, args)
    import torch
    from detectron2.modeling import build_model
    from adet.config import get_cfg
    ckpt = Path(args.checkpoint)
    if ckpt.name != 'model_final.pth':
        raise ValueError('The frozen plan evaluates model_final.pth only (got %s)' % ckpt.name)
    cfg = get_cfg()
    cfg.merge_from_file(str(source / 'configs/ViTAEv2_S/finetune/finetunw_ctw.yaml'))
    cfg.MODEL.WEIGHTS = str(ckpt)
    cfg.MODEL.DEVICE = 'cuda'
    cfg.TEST.LEXICON_TYPE = None
    cfg.freeze()
    model = build_model(cfg)
    saved = torch.load(str(ckpt), map_location='cpu')
    state = saved['model']
    loaded = model.load_state_dict(state, strict=True)
    actual = model.state_dict()
    for name, tensor in state.items():
        if actual[name].dtype != tensor.dtype or actual[name].shape != tensor.shape or not torch.equal(actual[name].detach().cpu(), tensor):
            raise ValueError('Loaded tensor differs from the checkpoint: ' + name)
    model.eval()
    parameters = sum(p.numel() for p in model.parameters())
    import contextlib
    import io
    from detectron2.data import transforms as T   # noqa: F401  (augmentation classes used by build_augmentation)
    augmentation = importlib.import_module('adet.data.detection_utils').build_augmentation(cfg, False)
    package = types.ModuleType('adet.evaluation')
    package.__path__ = [str(source / 'adet/evaluation')]
    sys.modules['adet.evaluation'] = package
    evaluator_class = importlib.import_module('adet.evaluation.text_evaluation_all').TextEvaluator
    exporter = object.__new__(evaluator_class)
    from transformers import BartTokenizer
    root = Path(runtime_cfg['expert']['asset_root'])
    exporter.bert_tokenizer = BartTokenizer(vocab_file=str(root / 'tokenizer/vocab.json'), merges_file=str(root / 'tokenizer/merge.txt'))
    exporter.dataset_name = 'ctw1500_test'
    images = json.load(open(Path(args.annotations) / 'ctw_test_bart_base.json'))['images']
    if len(images) != 500:
        raise ValueError('Expected the 500 official TEST images')
    if args.limit:
        images = images[:args.limit]
    rows_by_image, seconds = {}, {}
    tick = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), open(out / 'author_decode_stdout.txt', 'w') as log:
        for index, im in enumerate(images, 1):
            image = read_image(resolve_image(args.images, im['file_name']), im['height'], im['width'])
            if cfg.INPUT.FORMAT == 'RGB':
                image = image[:, :, ::-1]
            transformed = image
            for aug in augmentation:
                transformed = aug.get_transform(transformed).apply_image(transformed)
            if transformed.shape[0] % 32 or transformed.shape[1] % 32:
                raise ValueError('Author ViTAE padding is missing')
            tensor = torch.as_tensor(np.ascontiguousarray(transformed).astype('float32').transpose(2, 0, 1))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = model([dict(image=tensor, height=im['height'], width=im['width'])])[0]
            torch.cuda.synchronize()
            seconds[str(im['id'])] = time.perf_counter() - t0
            with contextlib.redirect_stdout(log):
                rows = exporter.instances_to_coco_json(result['instances'].to('cpu'), dict(image_id=im['id'], height=im['height'], width=im['width']))
            rows_by_image[str(im['id'])] = rows
            if index % 50 == 0:
                print('ticls ctw', index, len(images), flush=True)
    entries = [r for iid in sorted(rows_by_image, key=int) for r in rows_by_image[iid]]
    (out / 'ticls_ctw_test_rows.json').write_text(json.dumps(dict(images=rows_by_image, seconds=seconds)))
    # evaluation: TiCLS's own evaluator functions through the per-image counter wrapper (secondlook.spotters.deepsolo_ctw1500.Official)
    from secondlook.spotters.deepsolo_ctw1500 import Official, counts_array, f1_of, parse
    ids = sorted(rows_by_image, key=int)
    off = Official(source, args.assets, out / 'scratch', Path(args.assets) / 'evaluation' / 'gt_ctw1500.zip')
    result_table = {}
    for name, lex in (('None', None), ('Full', 1)):
        native, counters, official = off.run(entries, lex)
        counts = counts_array(counters, ids)
        result_table[name] = dict(e2e=parse(native['e2e_method']), det_only=parse(native['det_only_method']), official_string=official['e2e_method'],
                                  pooled_f1_pct=100 * f1_of(counts), correct=int(counts[:, 0].sum()), predictions=int(counts[:, 1].sum()), gt=int(counts[:, 2].sum()), counts=counts.tolist())
        print('TiCLS CTW1500', name, result_table[name]['e2e'], flush=True)
    paper = {'None': 66.2, 'Full': 88.1}
    verdict = dict(none_threshold=64.7, full_threshold=86.1,
                   near_paper=bool(result_table['None']['e2e']['hmean'] * 100 >= 64.7 and result_table['Full']['e2e']['hmean'] * 100 >= 86.1))
    report = dict(status='COMPLETE' if len(images) == 500 else 'SMOKE', mode='approximate',
                  claim_label='approximate reproduction (own CTW1500 fine-tune from the published Total-Text pretrain; not the authors CTW checkpoint)',
                  images=len(images), ticls_images='%d/%d (100%%)' % (len(images), len(images)), deepsolo_images='0/%d (0%%)' % len(images),
                  checkpoint=str(ckpt), checkpoint_sha256=sha256(ckpt), parameters=parameters, strict_load=dict(missing=list(loaded.missing_keys), unexpected=list(loaded.unexpected_keys)),
                  paper_reference=paper, results=result_table, verdict=verdict, package_versions=observed, wall_seconds=time.perf_counter() - tick,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=torch.cuda.get_device_name(),
                  precision='float32', batch_size=1, source_sha256=sha256(__file__), augmentation=[type(a).__name__ for a in augmentation])
    (out / 'results.json').write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ('status', 'ticls_images', 'verdict')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--runtime-config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--annotations', required=True, help='directory with ctw_test_bart_base.json')
    p.add_argument('--images', required=True, help='staged test_images directory (file names of the annotation file)')
    p.add_argument('--assets', required=True, help='E11 official assets directory (ctw1500/weak_voc*, evaluation/gt_ctw1500.zip)')
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, help='smoke only (result labelled SMOKE)')
    main(p.parse_args())
