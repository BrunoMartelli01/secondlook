"""Source-only calibrated admission from detector and existing language uncertainty.

A cheap control for adaptive readers, not by itself a novel spotting architecture.
All final cut metrics use the native polygon/text evaluator. Exact raw row
identity comes from an observation-only author-export trace, not bbox guessing."""

from pathlib import Path

import re

import zipfile

import numpy as np

FEATURE_NAMES=['detector_logit','log_relative_area','log_native_height','log_aspect','lm_log_length','lm_geometric_probability','lm_min_probability','lm_entropy','lm_min_margin','lm_finished','log_transcript_length','digit_fraction','letter_fraction']

def word_features(row,image_wh,input_wh,language_required=True):
    p=np.asarray(row['polys'],float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all():raise ValueError('Finite polygon required')
    wh=np.asarray(image_wh,float);span=np.maximum(p.max(0)-p.min(0),1.);score=float(row['score']);text=str(row['rec']);stats=row.get('local_lm_statistics')
    if not 0<=score<=1 or wh.shape!=(2,) or not np.isfinite(wh).all() or np.any(wh<=0) or np.shape(input_wh)!=(2,) or np.any(np.asarray(input_wh)<=0):raise ValueError('Valid probability and positive input/image dimensions required')
    if stats is None:
        if language_required:raise ValueError('Required native language statistics missing')
        stats=[0.]*6
    if np.shape(stats)!=(6,):raise ValueError('Exactly six aligned word language statistics required')
    x=[np.log(np.clip(score,1e-6,1-1e-6)/(1-np.clip(score,1e-6,1-1e-6))),np.log(np.prod(span)/np.prod(wh)),np.log(span[1]*input_wh[1]/wh[1]),np.log(span[0]/span[1])]+list(stats)+[np.log1p(len(text)),sum(c.isdigit() for c in text)/max(1,len(text)),sum(c.isalpha() for c in text)/max(1,len(text))]
    result=np.asarray(x,np.float32)
    if result.shape!=(len(FEATURE_NAMES),) or not np.isfinite(result).all():raise ValueError('Nonfinite/invalid word feature vector')
    return result

def subset_gt(original,destination,module,mapping,dataset):
    pattern=re.compile(module.default_evaluation_params()['GT_SAMPLE_NAME_2_ID']);keys={Path(v).stem.replace('res_img_','') for v in mapping.values()};found=set()
    with zipfile.ZipFile(original) as source,zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED) as target:
        for name in source.namelist():
            match=pattern.search(name)
            if match and match.group(1) in keys:
                key=match.group(1)
                if key in found:raise ValueError('Duplicate GT sample')
                found.add(key);target.writestr(name,source.read(name))
    if found!=keys:raise ValueError('Incomplete source GT population')
