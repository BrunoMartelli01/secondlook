"""E42 reader runs for ViTSTR-S and ABINet (legacy env, CPU). These two readers do NOT read the cached 32x128 crops: production (research/ideas_vitstr_read.py,
research/ideas_abinet_read.py) cuts the NATIVE-resolution patch (same polygon / perspective / rotation rule as p3_crops.crop, without the final 128x32 resize) from the image.
This driver imports the production functions unchanged (build, native_patch, to_tensor, read_batch) and only changes where polygons / images come from:
  mode gate  : first 128 patches of candidate-banks deepsolo/icdar2015 TRAIN (old Task, same image paths as production), first 100 compared with the stored readings
  mode extra : the EXTRA TRAIN images of <expert>/<task> exactly as research/fulltrain_crops.py enumerates them (T.new_ids, T.tab[s]['lines'], expanded_source_v3 rasters,
               sha256 checked); owner / cidx must equal the crop npz."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from secondlook.paths import RUNS
a = argparse.ArgumentParser()
a.add_argument('--reader', required=True, choices=['vitstr', 'abinet'])
a.add_argument('--mode', required=True, choices=['gate', 'extra'])
a.add_argument('--out', required=True); a.add_argument('--work', required=True)
a.add_argument('--expert'); a.add_argument('--task'); a.add_argument('--threads', type=int, default=4); a.add_argument('--batch', type=int, default=64)
a = a.parse_args()
import torch
torch.set_num_threads(a.threads)
from PIL import Image
import cv2
from secondlook.second_stage.read_vitstr import native_patch
from secondlook.second_stage.crops import B, DIRS, polygon
from secondlook.evaluation.protocol import read_json, sha256
E18 = (RUNS / 'candidate-banks')
if a.reader == 'vitstr':
    import secondlook.second_stage.read_vitstr as M
    model = M.build(); read_batch = lambda buf: M.read_batch(model, buf); to_tensor = M.to_tensor
    load_img = lambda p: np.asarray(Image.open(p).convert('RGB'))
    stored_dir = E18 / 'vitstr_run/p3'
else:
    import secondlook.second_stage.read_abinet as M
    sha = (M.ART / 'sha256.txt').read_text().split()[0]
    model, charset, _ = M.build(Path(M.SRC), Path(M.WEIGHTS), sha); read_batch = lambda buf: M.read_batch(model, charset, buf); to_tensor = M.to_tensor
    load_img = lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)          # as production (cv2.imread)
    stored_dir = E18 / 'abinet_run/p3'

def items_gate(work):
    from secondlook.data.tasks import Task
    T = Task('icdar2015', work)
    for s in T.ids:
        if not len(T.tab[s]['lines']):
            continue
        path = str(B / DIRS['icdar2015'] / 'train' / 'images' / ('%s.jpg' % s.split(':')[-1]))
        yield s, path, T.tab[s]['lines'], None

def items_extra(work):
    from secondlook.data.tasks_fulltrain import install
    install()
    from secondlook.selection.operating_point import load
    EXPANDED = (RUNS / 'ticls-source-banks/expanded_source_v3')
    T = load(a.expert, a.task, work)
    rec = {r['sample_id']: r for r in read_json(EXPANDED / ('%s_rgb_only.json' % a.task))['records']}
    for s in T.new_ids:
        if not len(T.tab[s]['lines']):
            continue
        r = rec[s]
        if sha256(r['image_path']) != r['image_sha256']:
            raise ValueError('Changed raster ' + s)
        yield s, r['image_path'], T.tab[s]['lines'], None

def run(items, limit=None, batches=None):
    owner, cidx, rows, buf = [], [], [], []
    def flush():
        if buf:
            rows.extend(read_batch(buf)); buf.clear()
    for s, path, lines, _ in items:
        img = load_img(path)
        for i, line in enumerate(lines):
            buf.append(to_tensor(native_patch(img, polygon(line)))); owner.append(s); cidx.append(i)
            if len(buf) >= a.batch:
                flush()
            if limit and len(owner) >= limit:
                flush(); return owner, cidx, rows
    flush()
    return owner, cidx, rows

work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
t0 = time.time()
if a.mode == 'gate':
    owner, cidx, rows = run(items_gate(work), limit=128)
    st = json.load(open(stored_dir / 'deepsolo_icdar2015_train_parseq.json'))
    assert owner[:100] == st['owner'][:100] and cidx[:100] == st['cidx'][:100], 'owner/cidx order differs from stored readings'
    new, ref = rows[:100], st['rows'][:100]
    res = dict(reader=a.reader, stored=str(stored_dir / 'deepsolo_icdar2015_train_parseq.json'), threads=a.threads,
               batching_A_production_64=dict(n_compared=100, exact_equal=bool(all(n == r for n, r in zip(new, ref))), texts_equal=all(n['text'] == r['text'] and n['n'] == r['n'] for n, r in zip(new, ref)),
                                             max_abs_diff=max(abs(n[k] - r[k]) for n, r in zip(new, ref) for k in ('seq', 'mint', 'meanlp')), n_rows_not_exact=sum(n != r for n, r in zip(new, ref))))
    res['gate_pass'] = res['batching_A_production_64']['exact_equal']; res['wall_s'] = time.time() - t0
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)); sys.exit(0 if res['gate_pass'] else 3)
else:
    owner, cidx, rows = run(items_extra(work))
    crops = np.load(str(RUNS / 'fulltrain/crops/%s_%s_train_extra.npz') % (a.expert, a.task))
    if owner != crops['owner'].tolist() or cidx != crops['cidx'].tolist():
        raise SystemExit('owner/cidx differ from the crop file')
    Path(a.out).write_text(json.dumps(dict(owner=owner, cidx=cidx, rows=rows)))
    print('done', a.reader, a.expert, a.task, len(rows), '%.0fs' % (time.time() - t0), flush=True)
