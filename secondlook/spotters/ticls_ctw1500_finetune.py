"""TiCLS fine-tuning on CTW1500 TRAIN - APPROXIMATE reproduction, NOT the authors' checkpoint.

Run with the legacy DeepSolo Python (torch 1.9 + detectron2 0.6) from the repository root:
``PYTHONPATH=. python -m secondlook.spotters.ticls_ctw1500_finetune ...``

Re-uses the qualified TiCLS runtime of ``scripts/run_ticls_inference.py`` (author source tree stubbed as the
``adet`` package, pinned native deformable-attention operator, strict PLM decoder, author tokenizer) and then runs
the AUTHORS' ``tools/train_net.py`` Trainer unchanged, with:
  * initial weights = the published ``ticls_totaltext_pretrain.pth`` (the CTW pretrain ``pretrained_ctw1500.pth`` named in
    ``finetunw_ctw.yaml`` is NOT published)                                                        -> ASSUMPTION
  * data = authors' CTW1500 line-level token annotations (``ctw_train_bart_base.json``)            -> REPOSITORY_SPECIFIED
  * schedule = paper Sec. 4.3: batch 2, lr 1e-5 -> 1e-6 @10K -> 1e-7 @20K; total iterations not stated  -> ASSUMPTION
No evaluation on TEST is performed by this script.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import random
import shutil
import sys
import tarfile
import types
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def stage_data(args, data_root):
    """Symlink CTW1500 rasters under the names used by the authors' annotation files (no pixels changed, except six EXIF-rotated copies)."""
    import cv2
    ann_dir = Path(args.annotations)
    for split, ann, folder in [('train', 'ctw_train_bart_base.json', 'train_images'), ('test', 'ctw_test_bart_base.json', 'test_images')]:
        d = Path(data_root) / 'ctw1500/ctw1500' / folder
        d.mkdir(parents=True, exist_ok=True)
        images = json.load(open(ann_dir / ann))['images']
        src = Path(args.images) / split / 'images'
        for im in images:
            target = d / im['file_name']
            if target.exists() or target.is_symlink():
                continue
            cand = src / im['file_name']
            if not cand.exists():
                cand = src / im['file_name'].lstrip('0')
            if not cand.exists():
                matches = [p for p in src.iterdir() if p.stem == Path(im['file_name']).stem.lstrip('0')]
                if len(matches) != 1:
                    raise FileNotFoundError(im['file_name'])
                cand = matches[0]
            override = Path(args.override) / im['file_name'].lstrip('0') if args.override else None
            if override is not None and override.exists():
                # original JPEG whose EXIF orientation would rotate it away from the annotation frame: store raw-frame pixels without EXIF
                raw = cv2.imread(str(override), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if raw is None or raw.shape[:2] != (im['height'], im['width']):
                    raise ValueError('Raw-frame size mismatch ' + str(override))
                ok, buf = cv2.imencode('.png', raw)
                target.write_bytes(buf.tobytes())
            else:
                os.symlink(cand.resolve(), target)
    return data_root


def initialize(runtime_cfg, source, args):
    runtime = runtime_cfg['runtime']
    scope = runtime['reuse_existing_packages']
    from secondlook.evaluation.existing_packages import reuse_existing_packages
    reuse_existing_packages(scope['site_packages'], scope['imports'])
    import importlib.metadata
    import numpy as np
    import torch
    from transformers import BartConfig, BartTokenizer
    observed = {name: importlib.metadata.version(name) for name in runtime['upstream_versions']}
    if observed != runtime['upstream_versions']:
        raise ValueError('Runtime differs from declared versions: ' + repr(observed))
    ex = runtime['execution']
    torch.cuda.set_device(ex['device'])
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    native = runtime['native_extension']
    if sha256(native['path']) != native['sha256'] or sha256(source / 'adet/layers/ms_deform_attn.py') != native['interface_sha256']:
        raise ValueError('Changed native extension or operator interface')
    for name, path in [('adet', source / 'adet'), ('adet.modeling', source / 'adet/modeling'), ('adet.data', source / 'adet/data')]:   # same stubs as scripts/run_ticls_inference.py (import order: patch language before losses)
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    spec = importlib.util.spec_from_file_location('adet._C', native['path'])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules['adet._C'] = mod
    root = Path(runtime_cfg['expert']['asset_root'])
    arts = runtime_cfg['expert']['artifacts']
    plm_path = root / 'weights' / arts['plm_decoder']['filename']
    if sha256(plm_path) != arts['plm_decoder']['sha256']:
        raise ValueError('Changed PLM decoder')
    for name, expected in runtime_cfg['expert']['tokenizer_sha256'].items():
        if sha256(root / 'tokenizer' / name) != expected:
            raise ValueError('Changed tokenizer ' + name)
    tokenizer = BartTokenizer(vocab_file=str(root / 'tokenizer/vocab.json'), merges_file=str(root / 'tokenizer/merge.txt'))
    language = importlib.import_module('adet.modeling.model.language')

    def strict_language_model():
        cfg = BartConfig(vocab_size=194, d_model=768, decoder_attention_heads=12, decoder_ffn_dim=3072, decoder_layers=6, max_position_embeddings=1024)
        module = language.DecoderWithLMHead(cfg)
        state = torch.load(str(plm_path), map_location='cpu')
        mapped = {k if k.startswith('lm_head') else 'decoder.' + k: v for k, v in state.items()}
        module.load_state_dict(mapped, strict=True)
        return module

    language.get_language_model_BART_dec = strict_language_model
    language.get_language_tokenizer_BART = lambda: tokenizer
    importlib.import_module('adet.modeling.vitae_v2')
    importlib.import_module('adet.modeling.text_spotter')
    return observed


def main(args):
    runtime_cfg = json.load(open(args.runtime_config))
    source = Path(runtime_cfg['expert']['asset_root']) / ('TiCLS-' + runtime_cfg['expert']['source_revision'])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    observed = initialize(runtime_cfg, source, args)
    import torch
    data_root = stage_data(args, out / 'data')
    # the author's registration uses placeholders: re-register the two CTW sets on the staged rasters/annotations (no source edit)
    importlib.import_module('adet.data.builtin')
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from adet.data.datasets.text import register_text_instances
    for name, ann, folder in [('ctw1500_word_poly_train', 'ctw_train_bart_base.json', 'train_images'), ('ctw1500_word_poly_test', 'ctw_test_bart_base.json', 'test_images')]:
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)
        register_text_instances(name, {'thing_classes': ['text']}, str(Path(args.annotations) / ann), str(data_root / 'ctw1500/ctw1500' / folder), 194, 25)
    spec = importlib.util.spec_from_file_location('ticls_train_net', source / 'tools/train_net.py')
    tn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tn)
    from adet.config import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file(str(source / 'configs/ViTAEv2_S/finetune/finetunw_ctw.yaml'))
    cfg.MODEL.WEIGHTS = str(args.pretrain)
    cfg.MODEL.DEVICE = 'cuda'
    cfg.SOLVER.IMS_PER_BATCH = 2 // args.micro_batches   # per micro-batch; effective batch stays 2 (see AccumTrainer)
    cfg.SOLVER.BASE_LR = 1e-5
    cfg.SOLVER.LR_BACKBONE = 1e-5
    cfg.SOLVER.STEPS = (10000, 20000)
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.TEST.EVAL_PERIOD = 0
    cfg.DATASETS.TEST = ()   # no evaluation inside training (EvalHook would score TEST at the end); evaluation is a separate frozen step
    cfg.DATALOADER.NUM_WORKERS = args.workers
    cfg.SEED = args.seed
    cfg.OUTPUT_DIR = str(out / 'train')
    cfg.freeze()
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    (out / 'resolved_train_config.yaml').write_text(cfg.dump())
    # strict architecture/weights compatibility check before any update
    from detectron2.modeling import build_model
    model = build_model(cfg)
    saved = torch.load(str(args.pretrain), map_location='cpu')
    state = saved['model'] if 'model' in saved else saved
    loaded = model.load_state_dict(state, strict=True)
    print('strict pretrain load OK: missing', list(loaded.missing_keys), 'unexpected', list(loaded.unexpected_keys), 'params', sum(p.numel() for p in model.parameters()), flush=True)
    del model, saved, state
    torch.cuda.empty_cache()
    class A:
        config_file = str(source / 'configs/ViTAEv2_S/finetune/finetunw_ctw.yaml')
        eval_only = False
        resume = False
        opts = []
    from detectron2.engine import default_setup
    from detectron2.utils.logger import setup_logger
    default_setup(cfg, A)
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=0, name='adet')
    json.dump(dict(pretrain=str(args.pretrain), pretrain_sha256=sha256(args.pretrain), max_iter=args.max_iter, steps=[10000, 20000], seed=args.seed, versions=observed,
                   annotations_sha256={n: sha256(Path(args.annotations) / n) for n in ['ctw_train_bart_base.json', 'ctw_test_bart_base.json']}, source_sha256=sha256(__file__),
                   resume=args.resume, chunked_plm=args.chunked_plm, claim_label='approximate reproduction (own CTW1500 fine-tune from the published Total-Text pretrain; not the authors CTW checkpoint)'), open(out / ('train_identity_resume.json' if args.resume else 'train_identity.json'), 'w'))
    micro = args.micro_batches
    if args.chunked_plm:
        import torch.nn.functional as F
        from torch.utils.checkpoint import checkpoint as _ckpt
        crit = sys.modules['adet.modeling.model.losses'].SetCriterion
        chunk = args.chunked_plm

        def train_plm_chunked(self, lang_embs, target_texts):
            """HARDWARE_DEVIATION, mathematically identical to the authors train_plm: token-mean cross entropy over all
            instances computed as sum over instance chunks / total tokens, each chunk under activation checkpointing
            (RNG state preserved) so the 6-layer BART decoder activations of images with many text lines are not all
            kept for backward (the E17 first attempt ran out of 23.6 GiB at iteration ~16880)."""
            bos = torch.full((target_texts.size(0), 1), self.lang_tokenizer.bos_token_id, device=lang_embs.device)
            dec_in = torch.cat([bos, target_texts[:, :-1]], dim=1)
            total = float(target_texts.numel())

            def part(emb, ids, tgt):
                logits = torch.nan_to_num(self.lang_model(ids, emb), nan=0.0, posinf=1e4, neginf=-1e4)
                return F.cross_entropy(logits.transpose(1, 2), tgt.long(), reduction='sum')
            acc = 0.0
            for i in range(0, target_texts.size(0), chunk):
                emb = lang_embs[i:i + chunk]
                if torch.is_grad_enabled() and emb.requires_grad:
                    acc = acc + _ckpt(part, emb, dec_in[i:i + chunk], target_texts[i:i + chunk])
                else:
                    acc = acc + part(emb, dec_in[i:i + chunk], target_texts[i:i + chunk])
            return acc / total
        crit.train_plm = train_plm_chunked

    class AccumTrainer(tn.Trainer):
        """HARDWARE_DEVIATION: effective batch 2 = 2 sequential micro-batches of 1 image (24 GB GPU; the paper used two GPUs for batch 2)."""

        def run_step(self):
            st = self._trainer
            assert st.model.training
            import time as _t
            start = _t.perf_counter()
            data = [next(st._data_loader_iter) for _ in range(micro)]
            data_time = _t.perf_counter() - start
            st.optimizer.zero_grad()
            summed = {}
            for batch in data:
                loss_dict = st.model(batch)
                losses = sum(loss_dict.values()) / micro
                losses.backward()
                for k, v in loss_dict.items():
                    summed[k] = summed.get(k, 0) + v.detach() / micro
            st._write_metrics(summed, data_time)
            st.optimizer.step()

    trainer = (AccumTrainer if micro > 1 else tn.Trainer)(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--runtime-config', required=True, help='existing TiCLS resolved_configuration.json (runtime, expert pins)')
    p.add_argument('--pretrain', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True, help='directory with ctw_train_bart_base.json / ctw_test_bart_base.json')
    p.add_argument('--images', type=Path, required=True, help='CTW1500 image root with train/images and test/images')
    p.add_argument('--override', type=Path, help='original JPEGs for six rotated rasters')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-iter', type=int, required=True)
    p.add_argument('--checkpoint-period', type=int, default=10000)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--micro-batches', type=int, default=2, help='gradient accumulation factor (HARDWARE_DEVIATION)')
    p.add_argument('--resume', action='store_true', help='resume from last_checkpoint in <out>/train')
    p.add_argument('--chunked-plm', type=int, default=0, help='instances per checkpointed PLM-loss chunk (0 = authors loss)')
    p.add_argument('--seed', type=int, default=20261105)
    main(p.parse_args())
