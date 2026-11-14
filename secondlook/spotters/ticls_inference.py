"""Source-pinned TiCLS RGB inference using already installed dependencies only.

Qualified runtime reuse is local to this subprocess; no package installation or
environment creation occurs. Run a separate smoke output before a full batch.
"""

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import sys
import tarfile
import time
import types

from secondlook.evaluation.existing_packages import reuse_existing_packages
from secondlook.spotters.ticls_export import export_author_words
from secondlook.spotters.ticls_assets_check import check_free_space, digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_input_image(path, policy):
    import cv2
    if policy == "raw_raster_ignore_exif":
        return cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if policy == "historical_opencv_default":
        return cv2.imread(str(path), cv2.IMREAD_COLOR)
    raise ValueError("Explicit supported benchmark raster-orientation policy required")


def initialize(config, dataset):
    runtime = config["runtime"]
    scope = runtime["reuse_existing_packages"]
    reuse_existing_packages(scope["site_packages"], scope["imports"])
    import numpy as np
    import torch
    from transformers import BartConfig, BartTokenizer

    observed = {name: importlib.metadata.version(name) for name in runtime["upstream_versions"]}
    if observed != runtime["upstream_versions"]:
        raise ValueError("Reused runtime differs from declared upstream versions: " + repr(observed))
    if not str(Path(torch.__file__).resolve()).startswith(str(Path(sys.prefix).resolve()) + os.sep):
        raise ValueError("Borrowed packages replaced the qualified runtime Torch")
    execution = runtime["execution"]
    if execution["precision"] != "float32" or execution["batch_size"] != 1:
        raise ValueError("Only the declared FP32 single-image author path is qualified")
    torch.cuda.set_device(execution["device"])
    if "memory_fraction" in execution:
        fraction = execution["memory_fraction"]
        if not 0 < fraction <= 1:
            raise ValueError("Valid explicit engineering memory cap required")
        torch.cuda.set_per_process_memory_fraction(fraction)
    torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(execution["cpu_threads"])
    random.seed(execution["seed"]); np.random.seed(execution["seed"]); torch.manual_seed(execution["seed"])
    torch.backends.cudnn.benchmark = execution["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = execution["cudnn_deterministic"]
    torch.backends.cuda.matmul.allow_tf32 = execution["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = execution["cudnn_allow_tf32"]
    root = Path(config["expert"]["asset_root"])
    source = root / ("TiCLS-" + config["expert"]["source_revision"])
    if digest(root / "ticls-source.tar.gz") != config["expert"]["source_archive_sha256"]:
        raise ValueError("Changed author source archive")
    with tarfile.open(root / "ticls-source.tar.gz") as archive:
        for member in archive.getmembers():
            if member.isfile():
                path = (root / member.name).resolve()
                if root.resolve() not in path.parents or digest(path) != hashlib.sha256(archive.extractfile(member).read()).hexdigest():
                    raise ValueError("Changed extracted author source: " + member.name)
    native = runtime["native_extension"]
    if digest(native["path"]) != native["sha256"] or digest(source / "adet/layers/ms_deform_attn.py") != native["interface_sha256"]:
        raise ValueError("Changed native extension or author operator interface")
    # Only the native operator is reused. Every architecture module is TiCLS.
    for name, path in [("adet", source / "adet"), ("adet.modeling", source / "adet/modeling"),
                       ("adet.data", source / "adet/data")]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    spec = importlib.util.spec_from_file_location("adet._C", native["path"])
    native_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native_module)
    sys.modules["adet._C"] = native_module
    language = importlib.import_module("adet.modeling.model.language")
    artifacts = config["expert"]["artifacts"]
    plm_path = root / "weights" / artifacts["plm_decoder"]["filename"]
    checkpoint = root / "weights" / artifacts[dataset]["filename"]
    for name, path in [("plm_decoder", plm_path), (dataset, checkpoint)]:
        if digest(path) != artifacts[name]["sha256"]:
            raise ValueError("Changed author model weights: " + name)
    tokenizer_files = config["expert"]["tokenizer_sha256"]
    for name, expected in tokenizer_files.items():
        if digest(root / "tokenizer" / name) != expected:
            raise ValueError("Changed author tokenizer: " + name)
    tokenizer = BartTokenizer(vocab_file=str(root / "tokenizer/vocab.json"),
                              merges_file=str(root / "tokenizer/merge.txt"))
    compatibility = {}

    def strict_language_model():
        # Same architecture/config as author language.py; all keys must load.
        cfg = BartConfig(vocab_size=194, d_model=768, decoder_attention_heads=12,
                         decoder_ffn_dim=3072, decoder_layers=6, max_position_embeddings=1024)
        module = language.DecoderWithLMHead(cfg)
        state = torch.load(str(plm_path), map_location="cpu")
        mapped = {key if key.startswith("lm_head") else "decoder." + key: value
                  for key, value in state.items()}
        loaded = module.load_state_dict(mapped, strict=True)
        compatibility["plm"] = dict(missing_keys=list(loaded.missing_keys),
                                    unexpected_keys=list(loaded.unexpected_keys), keys=len(mapped))
        return module

    language.get_language_model_BART_dec = strict_language_model
    language.get_language_tokenizer_BART = lambda: tokenizer
    importlib.import_module("adet.modeling.vitae_v2")
    importlib.import_module("adet.modeling.text_spotter")
    from adet.config import get_cfg
    from detectron2.modeling import build_model
    cfg = get_cfg()
    cfg.merge_from_file(str(source / config["expert"]["configuration_files"][dataset]))
    cfg.MODEL.DEVICE = execution["device"]
    cfg.MODEL.WEIGHTS = str(checkpoint)
    lexicon = config["evaluation"]["datasets"][dataset]["lexicon"]
    if lexicon not in {"Weak", "None"}:
        raise ValueError("Unsupported declared lexicon")
    cfg.TEST.LEXICON_TYPE = 2 if lexicon == "Weak" else None
    cfg.freeze()
    model = build_model(cfg)
    saved = torch.load(str(checkpoint), map_location="cpu")
    state = saved["model"]
    loaded = model.load_state_dict(state, strict=True)
    actual = model.state_dict()
    for name, tensor in state.items():
        if (actual[name].dtype != tensor.dtype or actual[name].shape != tensor.shape
                or not torch.equal(actual[name].detach().cpu(), tensor)):
            raise ValueError("Loaded tensor differs from the full checkpoint: " + name)
    compatibility["complete_model"] = dict(missing_keys=list(loaded.missing_keys),
        unexpected_keys=list(loaded.unexpected_keys), keys=len(state), all_loaded_tensors_bit_identical=True)
    model.eval()
    return model, cfg, tokenizer, compatibility, source, observed


def run(config_path, dataset, input_path, output, limit):
    config = json.loads(config_path.read_text())
    initial_config_sha256, initial_source_sha256 = digest(config_path), digest(__file__)
    source_pins = {str(Path(__file__).parents[1] / path): digest(Path(__file__).parents[1] / path)
                  for path in ["recognition/existing_packages.py", "recognition/ticls_export.py", "scripts/check_sota_expert_assets.py"]}
    if config["mode"] != "approximate" or config_path.resolve().parent.name != "approximate":
        raise ValueError("Explicit approximate profile required")
    execution = config["runtime"]["execution"]
    source_inputs = json.loads(input_path.read_text())
    partition = source_inputs["partition"]
    input_specs = config["evaluation"]["rgb_only_inputs"] if partition == "TEST" else config["evaluation"]["train_rgb_only_inputs"] if partition == "TRAIN" else None
    if input_specs is None:
        raise ValueError("Explicit TRAIN or TEST partition required")
    input_spec = input_specs[dataset]
    if digest(input_path) != input_spec["sha256"]:
        raise ValueError("Changed pinned RGB input manifest")
    records = source_inputs["records"]
    expected = input_spec["images"]
    if source_inputs["dataset"] != dataset or len(records) != expected or len({r["sample_id"] for r in records}) != expected:
        raise ValueError("Incomplete or duplicate RGB input population")
    if any(set(row) != {"sample_id", "image_path", "image_sha256", "image_wh"} for row in records):
        raise ValueError("Inference accepts only the pinned RGB fields, without annotations")
    storage = config["storage"]
    check_free_space([config["expert"]["asset_root"], storage["output_root"], storage["temporary_root"]], storage["minimum_free_bytes"])
    if limit is not None:
        if limit <= 0:
            raise ValueError("Positive smoke population required")
        records = records[:limit]
    output.mkdir(parents=True, exist_ok=False)
    (output / "samples").mkdir()
    write(output / "resolved_configuration.json", config)
    write(output / "run_identity.json", dict(pid=os.getpid(), parent_pid=os.getppid(),
        argv=sys.argv, config_sha256=initial_config_sha256, source_sha256=initial_source_sha256,
        helper_source_sha256=source_pins, input_sha256=digest(input_path), partition=partition))
    started = time.perf_counter()
    torch = None
    try:
        model, cfg, tokenizer, compatibility, source, versions = initialize(config, dataset)
        import cv2
        import numpy as np
        import torch
        from detectron2.data import transforms as T
        parameters = sum(p.numel() for p in model.parameters())
        keys_before = set(model.state_dict())
        # Use the actual author augmentation list, including ViTAE's Pad32.
        augmentation = importlib.import_module("adet.data.detection_utils").build_augmentation(cfg, False)
        # Serialization uses original TiCLS functions, without a COCO GT object.
        evaluation_package = types.ModuleType("adet.evaluation")
        evaluation_package.__path__ = [str(source / "adet/evaluation")]
        sys.modules["adet.evaluation"] = evaluation_package
        evaluator_class = importlib.import_module("adet.evaluation.text_evaluation_all").TextEvaluator
        exporter = object.__new__(evaluator_class)
        exporter.bert_tokenizer = tokenizer
        exporter.dataset_name = "ic15_test" if dataset == "icdar2015" else "totaltext_test"
        results, flattened = {}, []
        loaded_seconds = time.perf_counter() - started
        write(output / "load_audit.json", dict(strict_load_audit=compatibility,
            parameters=parameters, package_versions=versions, cfg=cfg.dump(),
            augmentation=[type(item).__name__ for item in augmentation],
            loaded_package_paths={name: str(sys.modules[name].__file__) for name in config["runtime"]["reuse_existing_packages"]["imports"] if name in sys.modules}))
        with torch.no_grad(), (output / "author_decode_stdout.txt").open("w") as log:
            for index, record in enumerate(records, 1):
                tick = time.perf_counter()
                path = Path(record["image_path"])
                if digest(path) != record["image_sha256"]:
                    raise ValueError("Changed RGB input: " + record["sample_id"])
                image = read_input_image(path, config.get("input_decoding", {}).get("policy", "historical_opencv_default"))
                if image is None or list(image.shape[1::-1]) != record["image_wh"]:
                    raise ValueError("Unreadable or changed image dimensions")
                if cfg.INPUT.FORMAT == "RGB":
                    image = image[:, :, ::-1]
                height, width = image.shape[:2]
                transformed = image
                for aug in augmentation:
                    transformed = aug.get_transform(transformed).apply_image(transformed)
                if transformed.shape[0] % 32 or transformed.shape[1] % 32:
                    raise ValueError("Author ViTAE padding is missing")
                tensor = torch.as_tensor(transformed.astype("float32").transpose(2, 0, 1))
                torch.cuda.synchronize()
                forward_started = time.perf_counter()
                result = model([dict(image=tensor, height=height, width=width)])[0]
                torch.cuda.synchronize()
                forward_seconds = time.perf_counter() - forward_started
                # ID mapping for official TT export is applied later; retain the
                # full original sample ID and unsimplified native polygons here.
                with contextlib.redirect_stdout(log):
                    rows = exporter.instances_to_coco_json(result["instances"].to("cpu"),
                        dict(image_id=index, height=height, width=width))
                if set(model.state_dict()) != keys_before or sum(p.numel() for p in model.parameters()) != parameters:
                    raise ValueError("Inference created uninitialized model state")
                sample = dict(sample_id=record["sample_id"], rows=rows, forward_seconds=forward_seconds,
                              full_image_seconds=time.perf_counter() - tick,
                              transformed_shape=list(tensor.shape), image_sha256=record["image_sha256"],
                              transformed_rgb_sha256=hashlib.sha256(transformed.tobytes()).hexdigest())
                results[record["sample_id"]] = sample
                write(output / "samples" / (record["sample_id"].replace(":", "__") + ".json"), sample)
                flattened.extend(rows)
                write(output / "progress.json", dict(images=index, expected_images=len(records)))
                print(dataset, index, "/", len(records), "predictions", len(rows), "forward_s", round(forward_seconds, 4), flush=True)
        torch.cuda.synchronize()
        final_export = export_author_words(config, dataset, results, output / "author_export", exporter) if partition == "TEST" else None
        if digest(config_path) != initial_config_sha256 or digest(__file__) != initial_source_sha256:
            raise ValueError("Resolved configuration or worker source changed during execution")
        if any(digest(path) != expected for path, expected in source_pins.items()):
            raise ValueError("Runtime/export helper source changed during execution")
        report = dict(status="COMPLETE", mode="approximate", claim_label=config["claim_label"],
            result_class="smoke-test result" if limit else "partial-dataset result" if source_inputs.get("partial_official_train") else "approximate reproduction",
            expert="TiCLS", dataset=dataset, partition=partition, total_images=len(records), expert_images=len(records),
            expert_percentage=100., deepsolo_images=0, deepsolo_percentage=0.,
            actual_new_ticls_calls=len(records), actual_new_deepsolo_calls=0,
            strict_load_audit=compatibility, measured_parameters=parameters, package_versions=versions,
            input_sha256=digest(input_path), config_sha256=digest(config_path), source_sha256=digest(__file__),
            initialization_seconds=loaded_seconds, total_seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu=torch.cuda.get_device_name(), execution=execution, out_of_memory=False,
            official_quality_evaluated=False, router_result=False, python=platform.python_version(),
            environment_installation_performed=False)
        report["helper_source_sha256"] = source_pins
        if final_export is not None:
            report["final_export"] = final_export
        report["export_source_sha256"] = digest(Path(__file__).parents[1] / "recognition/ticls_export.py")
        write(output / "per_image.json", results)
        write(output / "author_instances.json", flattened)
        write(output / "results.json", report)
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        if torch is None:
            torch = sys.modules.get("torch")
        message = str(error)
        oom = "out of memory" in message.lower()
        write(output / "failure.json", dict(error_type=type(error).__name__, error=message,
            out_of_memory=oom, peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch is not None and torch.cuda.is_initialized() else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch is not None and torch.cuda.is_initialized() else None))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=["icdar2015", "totaltext"], required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.config, args.dataset, args.inputs, args.output, args.limit)
