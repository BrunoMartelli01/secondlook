"""E27: candidate bank of the own TiCLS CTW1500 fine-tune (author threshold pool) with the 13 deployment word features.

Purpose: the word-admission head needs, per candidate, the same 13 features used for Total-Text / ICDAR2015 TiCLS
(``secondlook.features.word_trust.word_features``: logit score, log relative area, log relative height, log aspect ratio, six BART local
language statistics, log1p length, digit fraction, alpha fraction). The six statistics come from ``scale_features.decode_statistics``
(token probabilities of the greedy BART decode, computed from the author's own prefix logits; the recognised text is the author's decode, untouched). No ground truth is read. ``model_final.pth`` only.

Fidelity checks (hard failures): (1) the rec strings and scores of every candidate must equal those of the single TEST evaluation
(``ticls_ctw_test_rows.json``, ``--reference-rows``) when evaluating TEST; (2) rows and language statistics must be aligned.
Candidate pool = the author's CTW1500 inference threshold of the config (``INFERENCE_TH_TEST``); the evaluator additionally drops score <= 0.1.
Result class: approximate reproduction (own fine-tune; TRAIN images are in-sample for this model - see the study log E27).
"""
import argparse
import contextlib
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


def main(args):
    from secondlook.spotters.ticls_ctw1500_eval import read_image, resolve_image
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
    from secondlook.features.decode_statistics import decode_statistics
    from secondlook.features.word_trust import word_features
    ckpt = Path(args.checkpoint)
    if ckpt.name != 'model_final.pth':
        raise ValueError('The frozen plan uses model_final.pth only (got %s)' % ckpt.name)
    cfg = get_cfg()
    cfg.merge_from_file(str(source / 'configs/ViTAEv2_S/finetune/finetunw_ctw.yaml'))
    cfg.MODEL.WEIGHTS = str(ckpt)
    cfg.MODEL.DEVICE = 'cuda'
    cfg.TEST.LEXICON_TYPE = None
    cfg.freeze()
    model = build_model(cfg)
    state = torch.load(str(ckpt), map_location='cpu')['model']
    loaded = model.load_state_dict(state, strict=True)
    actual = model.state_dict()
    for name, tensor in state.items():
        if actual[name].dtype != tensor.dtype or actual[name].shape != tensor.shape or not torch.equal(actual[name].detach().cpu(), tensor):
            raise ValueError('Loaded tensor differs from the checkpoint: ' + name)
    model.eval()
    saved = {}

    author_decode = model.run_inference_w_BART   # the author's decode produces the recognised text and the logits of the generated prefix (unchanged)
    cross, finals = [], []

    def stats_one(language, logits):
        """The six BART statistics of ``decode_statistics`` for one instance from the author's own final prefix logits [T, V]."""
        pad, eos = language.config.pad_token_id, language.config.eos_token_id
        lp = logits.float().log_softmax(-1)
        p = lp.exp()
        top, indices = p.topk(2, dim=-1)
        top1, top2, nxt = top[:, 0], top[:, 1], indices[:, 0]
        stop = (nxt == pad) | (nxt == eos)
        before = (stop.long().cumsum(-1) - stop.long()) > 0          # a stop token occurred at an earlier step
        active = ~before & ~stop
        count = active.float().sum()
        den = count.clamp_min(1)
        mass = torch.where(active, top1.log(), torch.zeros_like(top1)).sum()
        minimum = torch.where(active, top1, torch.ones_like(top1)).min()
        entropy = torch.where(active, -(p * lp).sum(-1) / np.log(p.shape[-1]), torch.zeros_like(top1)).sum()
        margin = torch.where(active, top1 - top2, torch.ones_like(top1)).min()
        nonempty = count > 0
        zero = torch.zeros_like(count)
        return torch.stack([torch.log1p(count), torch.where(nonempty, (mass / den).exp(), zero), torch.where(nonempty, minimum, zero), entropy / den,
                            torch.where(nonempty, margin, zero), stop.any().float()])

    def decode(self, language, vis_embeds, max_len=27):
        original_forward = language.forward

        def recording_forward(*a, **k):
            out = original_forward(*a, **k)
            if k['input_ids'].shape[1] == 1:
                finals.append(None)
            finals[-1] = out[0]          # the author's loop keeps the logits of the LAST step of every instance
            return out
        language.forward = recording_forward
        finals.clear()
        try:
            result = author_decode(language, vis_embeds, max_len)
        finally:
            del language.forward
        if len(finals) != len(vis_embeds):
            raise ValueError('Recorded %d instance decodes for %d instances' % (len(finals), len(vis_embeds)))
        saved['language'] = torch.stack([stats_one(language, f) for f in finals]).cpu().numpy()
        finals.clear()
        if args.cross_check_statistics:
            reference_stats = decode_statistics(language, vis_embeds, max_len)[1]
            cross.append(float(np.abs(reference_stats - saved['language']).max()))
        return result
    model.run_inference_w_BART = types.MethodType(decode, model)
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
    annotation = 'ctw_%s_bart_base.json' % args.split
    images = json.load(open(Path(args.annotations) / annotation))['images']
    expected = {'train': 1000, 'test': 500}[args.split]
    if len(images) != expected:
        raise ValueError('Expected %d %s images' % (expected, args.split))
    if args.limit:
        images = images[:args.limit]
    reference = None
    if args.reference_rows:
        reference = json.load(open(args.reference_rows))['images']
    result_images, tick, mismatches = {}, time.perf_counter(), []
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
            saved['language'] = np.zeros((0, 6))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            instances = model([dict(image=tensor, height=im['height'], width=im['width'])])[0]['instances']
            torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            with contextlib.redirect_stdout(log):
                rows = exporter.instances_to_coco_json(instances.to('cpu'), dict(image_id=im['id'], height=im['height'], width=im['width']))
            statistics = saved['language']
            if len(rows) != len(statistics):
                raise ValueError('Language statistics not aligned with candidates for image %s (%d rows, %d statistics)' % (im['id'], len(rows), len(statistics)))
            input_wh = [transformed.shape[1], transformed.shape[0]]
            cands = []
            for row, values in zip(rows, statistics):
                row = dict(row)
                row['local_lm_statistics'] = [float(v) for v in values]
                row['x13'] = [float(v) for v in word_features(row, [im['width'], im['height']], input_wh, True)]
                cands.append(row)
            if reference is not None:
                ref = reference[str(im['id'])]
                got, want = [(c['rec'], round(c['score'], 6)) for c in cands], [(c['rec'], round(c['score'], 6)) for c in ref]
                if got != want:
                    if not args.report_mismatch:
                        raise ValueError('Bank candidates differ from the evaluated rows for image %s' % im['id'])
                    mismatches.append(dict(image=im['id'], got=got, want=want))
            result_images[str(im['id'])] = dict(width=im['width'], height=im['height'], file=im['file_name'], seconds=seconds, cands=cands)
            if index % 50 == 0:
                print('ticls ctw bank', args.split, index, len(images), flush=True)
    status = 'COMPLETE' if len(images) == expected else 'SMOKE'
    (out / ('cands_%s_ticls.json' % args.split)).write_text(json.dumps(dict(
        status=status, split=args.split, images=result_images, feature_names=['logit_score', 'log_rel_area', 'log_rel_height', 'log_aspect', 'lm_log1p_count', 'lm_geo_mean_prob',
                                                                              'lm_min_prob', 'lm_mean_entropy', 'lm_min_margin', 'lm_finished', 'log1p_len', 'digit_frac', 'alpha_frac'],
        threshold_pool=float(cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST), checkpoint=str(ckpt), checkpoint_sha256=sha256(ckpt), strict_load=dict(missing=list(loaded.missing_keys), unexpected=list(loaded.unexpected_keys)),
        no_gt_used=True, reference_rows_checked=bool(reference is not None), package_versions=observed, wall_seconds=time.perf_counter() - tick,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=torch.cuda.get_device_name(),
        precision='float32', batch_size=1, source_sha256=sha256(__file__))))
    if mismatches:
        (out / 'reference_mismatches.json').write_text(json.dumps(mismatches, indent=1))
        print('MISMATCHES', len(mismatches), flush=True)
    if cross:
        print('STATISTICS max abs difference vs decode_statistics:', max(cross), 'over', len(cross), 'images', flush=True)
    print('done', args.split, status, len(images), time.perf_counter() - tick, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--runtime-config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--split', choices=['train', 'test'], required=True)
    p.add_argument('--reference-rows', help='ticls_ctw_test_rows.json of the single TEST evaluation (checked row by row)')
    p.add_argument('--out', required=True)
    p.add_argument('--report-mismatch', action='store_true', help='diagnosis only: write the differing images instead of failing')
    p.add_argument('--cross-check-statistics', action='store_true', help='diagnosis: also run decode_statistics and report the maximal absolute difference')
    p.add_argument('--limit', type=int)
    main(p.parse_args())
