"""Locations of inputs and outputs. Everything that is machine specific is read here, from the environment.

SECONDLOOK_REPO       root of this checkout (default: the directory that contains the `secondlook` package)
SECONDLOOK_RUNS       working directory with the stored candidate banks, readings and run outputs (default: <repo>/runs)
SECONDLOOK_ARTIFACTS  pinned third-party sources and evaluators, checkpoints (default: <repo>/artifacts)
SECONDLOOK_DATASETS   benchmark images and annotations (default: <repo>/datasets)
See docs/DATA_AND_ASSETS.md for the layout of each directory.
"""
import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get('SECONDLOOK_REPO', Path(__file__).resolve().parents[1]))
RUNS = Path(os.environ.get('SECONDLOOK_RUNS', REPO_ROOT / 'runs'))
ARTIFACTS = Path(os.environ.get('SECONDLOOK_ARTIFACTS', REPO_ROOT / 'artifacts'))
DATASETS = Path(os.environ.get('SECONDLOOK_DATASETS', REPO_ROOT / 'datasets'))
