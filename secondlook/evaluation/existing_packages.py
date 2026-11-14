"""Reuse explicitly named, already installed Python packages in one subprocess.

No site-packages directory is appended to sys.path. The original interpreter's
imports and distributions retain priority; unrelated optional packages remain
unavailable. No virtual environment is created and no package is installed."""

import importlib.abc

import importlib.machinery

import importlib.metadata

from pathlib import Path

import sys

def normalized_distribution(name):
    return name.lower().replace("-", "_").replace(".", "_")

class ExistingPackages(importlib.abc.MetaPathFinder):
    def __init__(self, site_packages, imports):
        self.site_packages = str(Path(site_packages).resolve())
        if not Path(self.site_packages).is_dir():
            raise ValueError("Missing explicitly supplied package source")
        self.imports = dict(imports)
        if not self.imports or any(not name.isidentifier() for name in self.imports):
            raise ValueError("Explicit top-level import names required")
        self.distributions = {normalized_distribution(name) for name in self.imports.values()}

    def find_spec(self, fullname, path=None, target=None):
        if "." not in fullname and fullname in self.imports:
            return importlib.machinery.PathFinder.find_spec(fullname, [self.site_packages])
        return None

    def find_distributions(self, context=importlib.metadata.DistributionFinder.Context()):
        if normalized_distribution(context.name or "") not in self.distributions:
            return iter(())
        scoped = importlib.metadata.DistributionFinder.Context(name=context.name, path=[self.site_packages])
        return importlib.metadata.MetadataPathFinder.find_distributions(scoped)

def reuse_existing_packages(site_packages, imports):
    finder = ExistingPackages(site_packages, imports)
    # Append after the original finders: never override the qualified runtime.
    sys.meta_path.append(finder)
    return finder
