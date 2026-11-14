"""Explicit evaluator/quality contract for source calibration and deployment."""

import hashlib

import importlib

import json

from pathlib import Path

import sys

import types

import numpy as np

from secondlook.evaluation.quality_contract import QualityContract

from secondlook.evaluation.protocol import read_json, sha256, hmean_fraction

FIELDS=['e2e_correct','e2e_predictions','e2e_gt_words','e2e_localized','correct','predictions','gt_words']

def counter_key(filename,mode):
    if mode=='filename':return filename
    if mode=='native_stem':return Path(filename).stem.replace('res_img_','')
    raise ValueError('Explicit supported counter-key convention required')

def word_labels(word_indices,correct,ignored,total_words):
    indices=np.asarray(word_indices,np.int64);correct=set(correct);ignored=set(ignored)
    if len(set(indices.tolist()))!=len(indices) or np.any(indices<0) or np.any(indices>=total_words) or not correct.isdisjoint(ignored) or not (correct|ignored)<=set(range(len(indices))):
        raise ValueError('Native canonical word identity/label indices are inconsistent')
    return np.asarray([-1 if j in ignored else int(j in correct) for j in range(len(indices))],np.int8)

class DeclaredMetric:
    def __init__(self,cfg,spec):
        from secondlook.evaluation.official_benchmarks import load_upstream_evaluator
        if type(spec['word_spotting'])is not bool:raise ValueError('Explicit native word-care boolean required')
        self.key_mode=spec['counter_key_mode'];kind=spec['kind']
        if kind=='native':
            runtime=read_json(cfg['online']['datasets'][spec['source_domain']]['ticls_config']);source=Path(runtime['expert']['asset_root'])/('TiCLS-'+runtime['expert']['source_revision']);module=load_upstream_evaluator(str(source),spec['source_domain']);paths=[Path(module.__file__),Path(module.rrc_evaluation_funcs.__file__)]
        elif kind=='vimts_word':
            v=cfg['vimts_word'];identity=read_json(v['source_identity'])
            if sha256(v['source_identity'])!=v['source_identity_sha256']:raise ValueError('Changed deployed word-evaluator identity')
            paths=[]
            for name,value in identity['files'].items():
                p=Path(v['source_directory'])/name
                if sha256(p)!=value['sha256']:raise ValueError('Changed deployed evaluation source '+name)
                paths.append(p)
            package=types.ModuleType('declared_vimts_eval');package.__path__=[v['source_directory']];sys.modules['declared_vimts_eval']=package;module=importlib.import_module('declared_vimts_eval.text_eval_script')
        else:raise ValueError('Unsupported explicit evaluation implementation')
        self.module=module;self.parameters=module.default_evaluation_params();self.parameters['WORD_SPOTTING']=spec['word_spotting'];self.files={p.name:sha256(p) for p in paths};digest=hashlib.sha256(json.dumps(self.files,sort_keys=True).encode()).hexdigest();self.contract=QualityContract.create(spec['prediction_unit'],digest,self.parameters)

    def evaluate(self,gt,submission,ids,bodies):
        from secondlook.evaluation.official_benchmarks import parse_upstream_metrics
        from secondlook.evaluation.detection_counters import evaluate_with_detection_counts
        self.module.validate_data(str(gt),str(submission),self.parameters);native,counters=evaluate_with_detection_counts(self.module,str(gt),str(submission),self.parameters);keys=[counter_key(bodies[sid]['filename'],self.key_mode) for sid in ids]
        if len(set(keys))!=len(ids) or set(keys)!=set(counters):raise ValueError('Native evaluator omitted/added source images')
        counts=np.asarray([[counters[k][name] for name in FIELDS] for k in keys],np.int64)
        if abs(float(hmean_fraction(counts))-parse_upstream_metrics(native['e2e_method'])['hmean'])>1e-12:raise ValueError('Native aggregate/count mismatch')
        return counts,native,counters

    def record(self):
        return dict(signature=self.contract.signature,prediction_unit=self.contract.prediction_unit,aggregation=self.contract.aggregation,evaluator_sha256=self.contract.evaluator_sha256,parameters=self.parameters,files=self.files,counter_key_mode=self.key_mode)
