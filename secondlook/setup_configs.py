"""Write configs/approximate/*.json for this machine from configs/templates/*.json.

The templates use five placeholders, resolved from the environment (see secondlook.paths):
  $RUNS $ARTIFACTS $DATASETS $REPO   the four roots of secondlook.paths
  $PYTHON                            interpreter of the analysis environment (SECONDLOOK_PYTHON, default: this interpreter)
  $READER_PYTHON                     interpreter of the reader / PP-OCR environment (SECONDLOOK_READER_PYTHON, default: $PYTHON)
Each template pins the sha256 of the inputs it was written against ("pinned_inputs"). A pinned file that does not exist on this
machine is dropped from the written config and listed; with --strict-pins it is an error instead. A pinned file that exists but
has a different hash is always an error when the config is loaded (secondlook.evaluation.protocol.load_config).

    python -m secondlook.setup_configs [--strict-pins]
"""
import argparse
import json
import os
import sys
from pathlib import Path

from secondlook.paths import ARTIFACTS, DATASETS, REPO_ROOT, RUNS


def materialise(strict=False):
    py = os.environ.get('SECONDLOOK_PYTHON', sys.executable)
    sub = {'$RUNS': str(RUNS), '$ARTIFACTS': str(ARTIFACTS), '$DATASETS': str(DATASETS), '$REPO': str(REPO_ROOT),
           '$READER_PYTHON': os.environ.get('SECONDLOOK_READER_PYTHON', py), '$PYTHON': py}
    out_dir = REPO_ROOT / 'configs' / 'approximate'
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for template in sorted((REPO_ROOT / 'configs' / 'templates').glob('*.json')):
        text = template.read_text()
        for key in sorted(sub, key=len, reverse=True):
            text = text.replace(key, sub[key])
        cfg = json.loads(text)
        pins = cfg.get('pinned_inputs', {})
        missing = [k for k in pins if not Path(k).exists()]
        if missing and strict:
            raise SystemExit('%s: pinned inputs missing: %s' % (template.name, missing[:5]))
        cfg['pinned_inputs'] = {k: v for k, v in pins.items() if k not in missing}
        (out_dir / template.name).write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n')
        report[template.name] = dict(pinned=len(cfg['pinned_inputs']), dropped_missing=len(missing))
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--strict-pins', action='store_true')
    for name, r in materialise(ap.parse_args().strict_pins).items():
        print('%-34s pinned inputs verified at load: %d, dropped (file absent on this machine): %d' % (name, r['pinned'], r['dropped_missing']))
