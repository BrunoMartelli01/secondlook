"""Run the unchanged TiCLS output/lexicon converter in an isolated workdir.

Geometry repair, rounding, score ordering, text cleanup and lexicon rejection
come from the source-pinned author methods. Only filenames and missing empty
submissions are canonicalized after conversion; prediction bodies are preserved."""

import contextlib

import hashlib

import importlib

import json

import os

from pathlib import Path

import sys

import tarfile

import time

import types

import zipfile

from secondlook.evaluation.existing_packages import reuse_existing_packages

from secondlook.evaluation.official_benchmarks import sha256, write_zip

def submission_identity(dataset, sid, mapping):
    if dataset == "icdar2015":
        number = int(sid.split(":")[-1])
        return number, "res_img_" + str(number) + ".txt"
    if dataset != "totaltext":
        raise ValueError("Unknown official dataset")
    filename = mapping[sid]
    if Path(filename).name != filename or Path(filename).suffix != ".txt" or not Path(filename).stem.isdecimal():
        raise ValueError("Invalid author ground-truth filename")
    return int(Path(filename).stem), filename

def export_author_words(config, dataset, per_image, output, exporter=None):
    started = time.perf_counter()
    benchmark = json.loads(Path(config["evaluation"]["benchmark_config"]).read_text())
    spec = benchmark["datasets"][dataset]
    mapping = json.loads(Path(spec["spotting_id_map"]).read_text()) if dataset == "totaltext" else None
    source = Path(config["expert"]["asset_root"]) / ("TiCLS-" + config["expert"]["source_revision"])
    archive_path = source.parent / "ticls-source.tar.gz"
    if sha256(archive_path) != config["expert"]["source_archive_sha256"]:
        raise ValueError("Changed author source archive")
    relative = "adet/evaluation/text_evaluation_all.py"
    with tarfile.open(archive_path) as archive:
        expected_source = hashlib.sha256(archive.extractfile(source.name + "/" + relative).read()).hexdigest()
    if sha256(source / relative) != expected_source:
        raise ValueError("Changed author output converter")
    if exporter is None:
        scope = config["runtime"]["reuse_existing_packages"]
        reuse_existing_packages(scope["site_packages"], scope["imports"])
        for name, path in [("adet", source / "adet"), ("adet.modeling", source / "adet/modeling"),
                           ("adet.evaluation", source / "adet/evaluation")]:
            if name not in sys.modules:
                package = types.ModuleType(name); package.__path__ = [str(path)]; sys.modules[name] = package
        cls = importlib.import_module("adet.evaluation.text_evaluation_all").TextEvaluator
        exporter = object.__new__(cls)
    exporter.dataset_name = "ic15_test" if dataset == "icdar2015" else "totaltext_test"
    if spec["lexicon"] not in {"Weak", "None"} or (dataset == "totaltext" and spec["lexicon"] != "None"):
        raise ValueError("Unsupported declared author lexicon")
    exporter.lexicon_type = 2 if spec["lexicon"] == "Weak" else None
    expected = {}
    records = []
    for sid, sample in per_image.items():
        number, filename = submission_identity(dataset, sid, mapping)
        if number in expected:
            raise ValueError("Duplicate official image mapping")
        expected[number] = filename
        records.extend(dict(row, image_id=number) for row in sample["rows"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "author_instances.json").write_text(json.dumps(records, ensure_ascii=False) + "\n")
    if dataset == "icdar2015":
        links = output / "datasets/ic15"; links.mkdir(parents=True)
        for name, resource in config["evaluation"]["author_weak_lexicon"].items():
            if sha256(resource["path"]) != resource["sha256"]:
                raise ValueError("Changed author Weak lexicon")
            (links / name).symlink_to(resource["path"])
    old_cwd, old_system = Path.cwd(), os.system

    def checked_system(command):
        status = old_system(command)
        if status:
            raise RuntimeError("Author output ZIP command failed with status " + str(status))
        return status

    try:
        os.chdir(output)
        with Path("author_converter_stdout.txt").open("w") as log, contextlib.redirect_stdout(log):
            exporter.to_eval_format("author_instances.json", "raw")
            # The author converter emits only nonempty images. Empty files keep
            # the full image population and make the author's zip command valid.
            for number in expected:
                path = Path("raw") / ("{:07d}.txt".format(number))
                if not path.exists():
                    path.write_bytes(b"")
            os.system = checked_system
            raw_name, lexicon_name = exporter.sort_detection("raw/")
    finally:
        os.system = old_system
        os.chdir(old_cwd)
    selected = output / (lexicon_name if exporter.lexicon_type is not None else raw_name)
    canonical, counts = {}, 0
    with zipfile.ZipFile(selected) as archive:
        for name in archive.namelist():
            base = Path(name).name
            number = int(Path(base).stem.replace("res_img_", ""))
            if number not in expected or expected[number] in canonical:
                raise ValueError("Unexpected or duplicate author submission image")
            body = archive.read(name)
            canonical[expected[number]] = body.decode("utf-8")
            counts += len(body.splitlines())
    for filename in expected.values():
        canonical.setdefault(filename, "")
    final = output / "official_submission.zip"
    write_zip(final, canonical)
    report = dict(dataset=dataset, images=len(per_image), native_predictions=len(records),
                  submitted_predictions=counts, lexicon=spec["lexicon"],
                  submission=str(final), sha256=sha256(final), seconds=time.perf_counter()-started,
                  none_submission=str(output / raw_name), none_submission_sha256=sha256(output / raw_name),
                  author_methods="to_eval_format and sort_detection", bodies_modified=False,
                  explicit_empty_images=sum(not body for body in canonical.values()))
    (output / "export_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
