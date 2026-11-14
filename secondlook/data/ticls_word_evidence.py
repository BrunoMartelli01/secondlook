"""Join qualified original and expanded source word evidence without relabeling.

Native evaluator counters supply every new word target. Existing candidate
bytes, labels/features and GT are retained and independently reproduced.
"""

import argparse
from pathlib import Path
import zipfile
import numpy as np
from secondlook.evaluation.protocol import load_config, read_json, write_json, sha256, require_space
from secondlook.evaluation.declared_metric import DeclaredMetric, counter_key, word_labels
from secondlook.features.word_trust import word_features, FEATURE_NAMES
from secondlook.features.word_identity import export_with_identities


def merge_word_tables(old, extra):
    names = {'ids', 'features', 'target', 'image_index', 'raw_index', 'detector_score', 'counts'}
    if set(old) != names or set(extra) != names:
        raise ValueError('Explicit complete word-table schema required')
    for data in [old, extra]:
        n, rows = len(data['ids']), len(data['target'])
        if len(set(data['ids'].tolist())) != n or data['features'].shape != (rows, len(FEATURE_NAMES)) or data['counts'].shape != (n, 7) or any(data[k].shape != (rows,) for k in ['image_index', 'raw_index', 'detector_score']) or not np.isfinite(data['features']).all() or np.any((data['image_index'] < 0) | (data['image_index'] >= n)) or data['image_index'].dtype.kind not in 'iu' or data['raw_index'].dtype.kind not in 'iu' or np.any(data['raw_index'] < 0) or not set(data['target'].tolist()) <= {-1, 0, 1} or not np.isfinite(data['detector_score']).all() or np.any((data['detector_score'] < 0) | (data['detector_score'] > 1)) or data['counts'].dtype.kind not in 'iu' or np.any(data['counts'] < 0):
            raise ValueError('Invalid aligned source word table')
    if set(old['ids'].tolist()) & set(extra['ids'].tolist()):
        raise ValueError('Expanded word table reuses an old physical sample')
    result = {k:np.concatenate([old[k], extra[k]]) for k in names}
    result['image_index'] = np.concatenate([old['image_index'], extra['image_index']+len(old['ids'])])
    return result


def merge_archives(paths, destination):
    seen = set()
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as out:
        for path in paths:
            with zipfile.ZipFile(path) as source:
                for name in source.namelist():
                    if name in seen or name.endswith('/'):
                        raise ValueError('Duplicate/unsupported source archive entry')
                    seen.add(name)
                    out.writestr(name, source.read(name))
    return seen


def run(config_path, task, action):
    cfg = load_config(config_path)
    s = cfg['expanded_words']
    state = read_json(s['capture_completion'])
    if state['status'] != s['capture_completion_status']:
        raise ValueError('Complete qualified expanded source capture before labeling')
    profile = cfg['matched_readers'][task]
    domain = profile['domain']
    spec = profile[action]
    if sha256(spec['path']) != spec['sha256'] or sha256(spec['source_config']) != spec['source_config_sha256']:
        raise ValueError('Changed original admission source profile')
    old_frozen = read_json(spec['path'])
    source = load_config(spec['source_config'])
    metric = DeclaredMetric(source, source['protocol_calibration']['experiments'][old_frozen['experiment']]['metric'])
    if metric.record()['signature'] != profile['quality_contract']['signature']:
        raise ValueError('Expanded source task differs from deployment contract')
    root = Path(cfg['output_root'])
    out = root/s['directory']/task/action
    out.mkdir(parents=True, exist_ok=False)
    require_space(out, cfg['storage']['minimum_free_bytes'])
    inventory = read_json(s['inventory']['path'])
    if sha256(s['inventory']['path']) != s['inventory']['sha256']:
        raise ValueError('Changed source expansion population/roles')
    evidence = inventory['domains'][domain]['files']
    for item in evidence.values():
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Changed expanded source annotation/role/RGB evidence')
    collection = cfg['expanded_words']['collections'][domain][action]
    if sha256(collection['config']) != collection['config_sha256']:
        raise ValueError('Changed source capture action configuration')
    capture_cfg = load_config(collection['config'])
    bank_path = root/capture_cfg['scale_router']['local_bank_directory']/domain/'predictions.json'
    bank = read_json(bank_path)
    if bank['config_sha256'] != collection['config_sha256'] or bank['compatibility']['complete_model']['all_loaded_tensors_bit_identical'] is not True:
        raise ValueError('Expanded source expert identity is not qualified')
    records = read_json(evidence['rgb.json']['path'])['records']
    ids = [r['sample_id'] for r in records]
    excluded = set(inventory['domains'][domain].get('excluded_native_invalid_source_ids', []))
    if set(ids) & excluded or set(ids) | excluded != set(bank['rows']):
        raise ValueError('Expanded source collection omitted/added images')
    image_wh = {r['sample_id']:r['image_wh'] for r in records}
    mapping = read_json(evidence['id_map.json']['path'])
    local = read_json(cfg['online']['datasets'][domain]['ticls_config'])
    benchmark = read_json(local['evaluation']['benchmark_config'])
    benchmark['datasets'][domain]['lexicon'] = 'None'
    if domain == 'totaltext':
        benchmark['datasets'][domain]['spotting_id_map'] = evidence['id_map.json']['path']
    write_json(out/'benchmark.json', benchmark)
    local['evaluation']['benchmark_config'] = str(out/'benchmark.json')
    raw = {sid:dict(rows=bank['rows'][sid]['low']['rows']) for sid in ids}
    exported, identity = export_with_identities(local, domain, raw, out/'export')
    full_manifest = read_json(root/capture_cfg['scale_router']['inference_directory']/'train/manifest.json')
    reference_zip = full_manifest['datasets'][domain]['export']['submission']
    reference_map_spec = collection['id_map']
    if sha256(reference_map_spec['path']) != reference_map_spec['sha256']:
        raise ValueError('Changed historical source export identity map')
    reference_mapping = read_json(reference_map_spec['path'])
    with zipfile.ZipFile(reference_zip) as reference, zipfile.ZipFile(exported['submission']) as observed:
        if set(reference.namelist()) != set(reference_mapping.values()) or set(observed.namelist()) != {mapping[sid] for sid in ids} or any(reference.read(reference_mapping[sid]) != observed.read(mapping[sid]) for sid in ids):
            raise ValueError('Expanded labeling export changed collected author bodies')
    features, image_indices, raw_indices, scores, bodies = [], [], [], [], {}
    for i, sid in enumerate(ids):
        state = bank['rows'][sid]['low']
        if bank['rows'][sid]['image_sha256'] != records[i]['image_sha256']:
            raise ValueError('Expanded source raster identity changed')
        bb = dict(filename=mapping[sid], lines=identity['canonical_lines'][sid], word_indices=[])
        for raw_index in identity['raw_indices'][sid]:
            row = state['rows'][raw_index]
            bb['word_indices'].append(len(features))
            features.append(word_features(row, image_wh[sid], state['input_wh'], True))
            image_indices.append(i)
            raw_indices.append(raw_index)
            scores.append(row['score'])
        bodies[sid] = bb
    counts, native, counters = metric.evaluate(evidence['gt.zip']['path'], exported['submission'], ids, bodies)
    target = np.full(len(features), -128, np.int8)
    for sid in ids:
        key = counter_key(mapping[sid], metric.key_mode)
        indices = bodies[sid]['word_indices']
        if len(indices) != len(native['per_sample'][key]['detTrans']):
            raise ValueError('Expanded raw/native word index alignment failed')
        target[indices] = word_labels(indices, counters[key]['e2e_correct_indices'], native['per_sample'][key]['detDontCare'], len(target))
    if np.any(target == -128) or sum(target == 1) != int(counts[:, 0].sum()) or sum(target >= 0) != int(counts[:, 1].sum()):
        raise ValueError('Expanded source labels do not reproduce official TP/care counts')
    extra = dict(ids=np.array(ids), features=np.asarray(features, np.float32).reshape(-1, len(FEATURE_NAMES)), target=target, image_index=np.asarray(image_indices, np.int32), raw_index=np.asarray(raw_indices, np.int32), detector_score=np.asarray(scores), counts=counts)
    old_directory = Path(spec['path']).parents[1]
    old_meta = read_json(old_directory/'prepared.json')['views']['low']
    if sha256(old_directory/'low/words.npz') != old_meta['words_sha256'] or sha256(old_directory/'low/bodies.json') != old_meta['bodies_sha256']:
        raise ValueError('Changed original word evidence')
    with np.load(old_directory/'low/words.npz') as archive:
        old = {k:archive[k] for k in archive.files}
    merged = merge_word_tables(old, extra)
    all_bodies = read_json(old_directory/'low/bodies.json')
    for sid, bb in bodies.items():
        all_bodies[sid] = dict(bb, word_indices=[j+len(old['target']) for j in bb['word_indices']])
    merge_archives([old_directory/'low/source_gt.zip', evidence['gt.zip']['path']], out/'source_gt.zip')
    merge_archives([old_directory/'low/original_candidates.zip', exported['submission']], out/'original_candidates.zip')
    verified, _, _ = metric.evaluate(out/'source_gt.zip', out/'original_candidates.zip', merged['ids'].tolist(), all_bodies)
    if not np.array_equal(verified, merged['counts']):
        raise ValueError('Merged source populations changed original/new per-image counts')
    np.savez_compressed(out/'words.npz', **merged)
    write_json(out/'bodies.json', all_bodies)
    roles = read_json(evidence['roles.json']['path'])
    if set(roles['fit_ids']+roles['selection_ids']+roles['audit_ids']) != set(merged['ids'].tolist()) or any(old_frozen[k] != [sid for sid in roles[k] if sid in set(old_frozen[k])] for k in ['fit_ids', 'selection_ids', 'audit_ids']):
        raise ValueError('Expanded roles changed original physical membership/order')
    write_json(out/'partitions.json', roles)
    write_json(out/'prepared.json', dict(status='COMPLETE_QUALIFIED_EXPANDED_WORD_EVIDENCE', mode='approximate', claim_label=cfg['claim_label'], task=task, action=action, total_images=len(merged['ids']), deepsolo_images=0, deepsolo_percentage=0., ticls_images=len(merged['ids']), ticls_percentage=100., new_expert_forward_calls=0, result_class='partial-dataset source development' if excluded else 'complete source development', excluded_native_invalid_source_ids=sorted(excluded), original_additional_source_images=len(ids)+len(excluded), old_images=len(old['ids']), added_images=len(ids), canonical_words=len(merged['target']), ignored_words=int((merged['target']<0).sum()), quality_contract=metric.record(), original_profile=spec, source_bank_sha256=sha256(bank_path), words_sha256=sha256(out/'words.npz'), bodies_sha256=sha256(out/'bodies.json'), source_gt_sha256=sha256(out/'source_gt.zip'), candidates_sha256=sha256(out/'original_candidates.zip'), partitions_sha256=sha256(out/'partitions.json'), config_sha256=sha256(config_path), source_sha256=sha256(__file__), scope='Qualified source evidence only. Original728 bodies/features/labels retained; appended original TRAIN candidates get actual deployment-evaluator labels. All combined original/new per-image counts independently reproduced. Explicit TT native-invalid source cohort exclusions are not missing target images; complete original predictions/GT preserved. Historical extra-source metrics already inspected; no untouched-source claim. Source roles pinned, targets/GT unchanged. No fitted head or quality improvement claim.'))
    print('EXPANDED WORD EVIDENCE QUALIFIED', task, action, len(merged['ids']), len(merged['target']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--task', required=True, choices=['totaltext', 'icdar2015', 'ctw1500_word'])
    p.add_argument('--action', required=True, choices=['economical', 'native'])
    a = p.parse_args()
    run(a.config, a.task, a.action)
