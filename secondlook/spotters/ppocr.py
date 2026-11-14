"""Pinned PP-OCRv5 Mobile ONNX adapter; annotation-blind, isolated CUDA worker.

The author's DB postprocessor and perspective crop function are loaded from
hash-verified source definitions without importing the Paddle framework.
"""

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np


FEATURE_NAMES = (
    'count_log1p','empty','image_log_aspect','image_log_megapixels',
    'det_mean','det_min','det_q25','det_std',
    'rec_mean','rec_min','rec_q25','rec_std',
    'min_char_mean','min_char_min','margin_mean','margin_min',
    'entropy_mean','entropy_max','char_low_t1_fraction','char_low_t2_fraction',
    'rec_uncertainty_sum','characters_log1p','length_mean','length_min','length_max',
    'short_or_symbol_fraction','punctuation_fraction','coverage_sum','area_min','area_mean',
    'short_side_min_log1p','short_side_mean_log1p','maximum_box_iou',
    'rejected_proposals_log1p','rejected_proposal_max','rejected_proposal_mass',
    'recognition_rejected_log1p','rejected_t1_log1p','rejected_t2_log1p','foreground_fraction',
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_raster(path,policy):
    if policy!='raw_raster_ignore_exif':raise ValueError('Explicit qualified raw-raster input policy required')
    return cv2.imread(str(path),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)


def decode_ctc(probabilities, characters, cutoffs):
    p=np.asarray(probabilities,dtype=np.float64)
    if (p.ndim!=2 or not p.shape[0] or p.shape[1]!=len(characters)
            or not np.isfinite(p).all() or np.any(p<0.)
            or not np.allclose(p.sum(1),1.,atol=1e-4,rtol=0.)):
        raise ValueError('Normalized finite CTC probabilities and exact charset required')
    tokens=p.argmax(1)
    selected=np.ones(len(tokens),bool)
    selected[1:]=tokens[1:]!=tokens[:-1]
    selected &= tokens!=0
    ix=np.flatnonzero(selected)
    text=''.join(characters[int(tokens[i])] for i in ix)
    if not len(ix):
        return dict(text='',score=0.,char_probs=[],min_char=0.,margin=0.,entropy=1.,
                    low_t1=1.,low_t2=1.)
    rows=p[ix];values=rows[np.arange(len(ix)),tokens[ix]]
    second=np.partition(rows,-2,axis=1)[:,-2]
    ent=-(rows*np.log(np.maximum(rows,1e-15))).sum(1)/math.log(p.shape[1])
    return dict(text=text,score=float(values.mean()),char_probs=values.tolist(),
                min_char=float(values.min()),margin=float((values-second).min()),
                entropy=float(ent.max()),low_t1=float(np.mean(values<cutoffs[0])),
                low_t2=float(np.mean(values<cutoffs[1])))


def word_quad(points,left,right,vertical):
    """Map an observed CTC interval through the exact rectified quad axes."""
    p=np.asarray(points,np.float32)
    if p.shape!=(4,2) or not 0.<=left<right<=1.:
        raise ValueError('Valid quad and nonempty normalized word interval required')
    if vertical:
        value=[p[1]*(1-left)+p[2]*left,p[1]*(1-right)+p[2]*right,
               p[0]*(1-right)+p[3]*right,p[0]*(1-left)+p[3]*left]
    else:
        value=[p[0]*(1-left)+p[1]*left,p[0]*(1-right)+p[1]*right,
               p[3]*(1-right)+p[2]*right,p[3]*(1-left)+p[2]*left]
    return np.rint(value).astype(int).tolist()


def box_iou(a,b):
    x=max(0.,min(a[2],b[2])-max(a[0],b[0]));y=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    inter=x*y
    area=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/area if area>0 else 0.


def features(rows,rejected,rec_rejected,image_wh,foreground,proposal_cuts):
    w,h=image_wh
    if min(w,h)<=0:raise ValueError('Positive original image dimensions required')
    n=len(rows);values=[math.log1p(n),float(n==0),math.log(w/h),math.log(w*h/1e6)]
    def stats(key,empty=0.):
        a=np.asarray([r[key] for r in rows],float)
        return [float(a.mean()),float(a.min()),float(np.quantile(a,.25)),float(a.std())] if n else [empty]*4
    values+=stats('det_score')+stats('score')
    probs=[r['min_char'] for r in rows];margins=[r['margin'] for r in rows];ent=[r['entropy'] for r in rows]
    values += [float(np.mean(probs)) if n else 0.,min(probs,default=0.),
               float(np.mean(margins)) if n else 0.,min(margins,default=0.),
               float(np.mean(ent)) if n else 1.,max(ent,default=1.),
               float(np.mean([r['low_t1'] for r in rows])) if n else 1.,
               float(np.mean([r['low_t2'] for r in rows])) if n else 1.,
               sum(1.-r['score'] for r in rows)]
    lengths=[len(r['text']) for r in rows];chars=''.join(r['text'] for r in rows)
    values += [math.log1p(sum(lengths)),float(np.mean(lengths)) if n else 0.,
               min(lengths,default=0),max(lengths,default=0),
               sum(sum(c.isalnum() for c in r['text'])<=1 for r in rows)/n if n else 1.,
               sum(not c.isalnum() for c in chars)/max(1,len(chars))]
    boxes=[];areas=[];short=[]
    for r in rows:
        p=np.asarray(r['points']);a=[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]
        boxes.append(a);areas.append((a[2]-a[0])*(a[3]-a[1])/(w*h));short.append(min(a[2]-a[0],a[3]-a[1]))
    values += [sum(areas),min(areas,default=0.),float(np.mean(areas)) if n else 0.,
               math.log1p(min(short,default=0.)),math.log1p(float(np.mean(short))) if n else 0.,
               max((box_iou(a,b) for i,a in enumerate(boxes) for b in boxes[i+1:]),default=0.),
               math.log1p(len(rejected)),max(rejected,default=0.),sum(rejected),math.log1p(rec_rejected),
               math.log1p(sum(s>=proposal_cuts[0] for s in rejected)),
               math.log1p(sum(s>=proposal_cuts[1] for s in rejected)),float(foreground)]
    result=np.asarray(values,float)
    if result.shape!=(len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError('Compact feature contract failed')
    return result.tolist()


def source_definition(spec,name,namespace):
    path=Path(spec['path'])
    if digest(path)!=spec['sha256']:raise ValueError('Pinned author source changed: '+str(path))
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name]
    if len(nodes)!=1:raise ValueError('Unique author definition required: '+name)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


class CompactOCR:
    def __init__(self,cfg):
        import onnxruntime as ort
        import pyclipper
        import yaml
        from shapely.geometry import Polygon
        self.cfg=cfg
        cv2.setNumThreads(cfg['runtime']['cpu_threads'])
        self.det_cfg=cfg['detector'];self.rec_cfg=cfg['recognizer']
        self.sessions=[]
        for spec in [self.det_cfg,self.rec_cfg]:
            if digest(spec['model_path'])!=spec['model_sha256']:raise ValueError('Model bytes changed')
            options=ort.SessionOptions();options.intra_op_num_threads=cfg['runtime']['cpu_threads']
            options.enable_mem_pattern=False
            session=ort.InferenceSession(spec['model_path'],sess_options=options,
                providers=[('CUDAExecutionProvider',cfg['runtime']['cuda_options'])])
            session.disable_fallback()
            if session.get_providers()[0]!='CUDAExecutionProvider':
                raise RuntimeError('CUDAExecutionProvider required; CPU model fallback forbidden')
            self.sessions.append(session)
        if digest(self.rec_cfg['yaml_path'])!=self.rec_cfg['yaml_sha256']:raise ValueError('Charset recipe changed')
        rec_meta=yaml.safe_load(Path(self.rec_cfg['yaml_path']).read_text())
        self.characters=['blank']+rec_meta['PostProcess']['character_dict']+[' ']
        namespace=dict(np=np,cv2=cv2,Polygon=Polygon,pyclipper=pyclipper)
        post=source_definition(cfg['author_sources']['db_postprocess'],'DBPostProcess',namespace)
        self.post=post(**self.det_cfg['postprocess'])
        self.crop=source_definition(cfg['author_sources']['crop'],'get_rotate_crop_image',dict(np=np,cv2=cv2))
        decoder=source_definition(cfg['author_sources']['ctc_decoder'],'BaseRecLabelDecode',dict(np=np,re=re))
        self.word_groups=lambda text,selection:decoder.get_word_info(None,text,selection)
        self.ready=dict(providers=[s.get_providers() for s in self.sessions],
                        onnxruntime=ort.__version__,model_bytes=sum(Path(x['model_path']).stat().st_size for x in [self.det_cfg,self.rec_cfg]),
                        precision='float32',image_batch_size=1,recognition_batch_size=self.rec_cfg['batch_size'],
                        feature_names=list(FEATURE_NAMES),pid=__import__('os').getpid(),
                        allocated_peak_bytes=None,reserved_peak_bytes=None,
                        memory_scope='ORT exposes no public allocated/reserved counters; record sampled per-process GPU resident memory separately')

    def predict(self,record):
        started=time.perf_counter();path=Path(record['image_path'])
        if digest(path)!=record['image_sha256']:raise ValueError('RGB checksum changed')
        image=read_raster(path,self.cfg['input_policy'])
        if image is None:raise ValueError('Unreadable image')
        h,w=image.shape[:2]
        if [w,h]!=record['image_wh']:raise ValueError('Decoded image geometry differs')
        scale=self.det_cfg['resize_long']/max(h,w);stride=self.det_cfg['resize_stride']
        rh=(int(h*scale)+stride-1)//stride*stride;rw=(int(w*scale)+stride-1)//stride*stride
        resized=cv2.resize(image,(rw,rh)).astype(np.float32)/255.
        tensor=((resized-np.array(self.det_cfg['mean'],np.float32))/np.array(self.det_cfg['std'],np.float32)).transpose(2,0,1)[None]
        det=self.sessions[0];prob=det.run(None,{det.get_inputs()[0].name:np.ascontiguousarray(tensor)})[0][0,0]
        if not np.isfinite(prob).all():raise ValueError('Nonfinite detector probability')
        bitmap=prob>self.det_cfg['postprocess']['thresh']
        boxes,scores=self.post.boxes_from_bitmap(prob,bitmap,w,h)
        selected=[];rejected=[]
        for points,score in zip(boxes,scores):
            if score<self.det_cfg['admission_score']:
                rejected.append(float(score));continue
            p=np.asarray(points,dtype=np.float32)
            total=p.sum(1);ordered=np.zeros((4,2),np.float32)
            ordered[0]=p[np.argmin(total)];ordered[2]=p[np.argmax(total)]
            rest=np.delete(p,(np.argmin(total),np.argmax(total)),axis=0)
            delta=np.diff(rest,axis=1);ordered[1]=rest[np.argmin(delta)];ordered[3]=rest[np.argmax(delta)]
            ordered[:,0]=np.clip(ordered[:,0],0,w-1);ordered[:,1]=np.clip(ordered[:,1],0,h-1)
            if min(int(np.linalg.norm(ordered[0]-ordered[1])),int(np.linalg.norm(ordered[0]-ordered[3])))<=self.det_cfg['minimum_crop_side']:
                continue
            crop=self.crop(image,ordered.copy())
            selected.append(dict(points=ordered.astype(int).tolist(),det_score=float(score),crop=crop))
        result=[None]*len(selected);order=sorted(range(len(selected)),key=lambda i:selected[i]['crop'].shape[1]/selected[i]['crop'].shape[0])
        batch=self.rec_cfg['batch_size'];rec=self.sessions[1];ch,height,min_width=self.rec_cfg['image_shape']
        for begin in range(0,len(order),batch):
            ix=order[begin:begin+batch]
            ratio=max([min_width/height]+[selected[i]['crop'].shape[1]/selected[i]['crop'].shape[0] for i in ix])
            width=int(height*ratio)
            if width>self.rec_cfg['maximum_width']:raise ValueError('Recognition width exceeds explicit qualified limit')
            images=[]
            for i in ix:
                crop=selected[i]['crop'];cw=min(width,int(math.ceil(height*crop.shape[1]/crop.shape[0])))
                normalized=cv2.resize(crop,(cw,height)).astype(np.float32).transpose(2,0,1)/255.
                normalized=(normalized-.5)/.5
                padded=np.zeros((ch,height,width),np.float32);padded[:,:,:cw]=normalized;images.append(padded)
            probabilities=rec.run(None,{rec.get_inputs()[0].name:np.stack(images)})[0]
            for i,p in zip(ix,probabilities):
                decoded=decode_ctc(p,self.characters,self.cfg['feature_settings']['char_probability_cutoffs'])
                tokens=p.argmax(1);selection=np.ones(len(tokens),bool)
                selection[1:]=tokens[1:]!=tokens[:-1];selection &= tokens!=0
                groups,columns,states=self.word_groups(decoded['text'],selection)
                crop=selected[i]['crop'];crop_ratio=crop.shape[1]/crop.shape[0]
                valid_columns=len(tokens)*crop_ratio/(width/height)
                decoded['word_groups']=[''.join(g) for g in groups]
                decoded['word_columns']=[list(map(int,g)) for g in columns]
                decoded['valid_ctc_columns']=float(valid_columns)
                result[i]=dict(points=selected[i]['points'],det_score=selected[i]['det_score'],**decoded)
        rows=[r for r in result if r['score']>=self.rec_cfg['drop_score'] and r['text']]
        words=[];invalid_words=[]
        for line_id,r in enumerate(rows):
            points=np.asarray(r['points']);vertical=np.linalg.norm(points[0]-points[3])/np.linalg.norm(points[0]-points[1])>=self.rec_cfg['vertical_crop_ratio']
            for word,columns in zip(r['word_groups'],r['word_columns']):
                if len(r['word_groups'])==1:
                    polygon=r['points']
                else:
                    left=max(0.,min(1.,columns[0]/r['valid_ctc_columns']))
                    right=max(0.,min(1.,(columns[-1]+1)/r['valid_ctc_columns']))
                    if left>=right:
                        invalid_words.append(dict(line_id=line_id,text=word,reason='CTC interval outside valid crop'));continue
                    polygon=word_quad(r['points'],left,right,vertical)
                if abs(cv2.contourArea(np.asarray(polygon,np.float32)))<=0:
                    invalid_words.append(dict(line_id=line_id,text=word,reason='Rounded word polygon degenerated'));continue
                words.append(dict(points=polygon,text=word,score=r['score'],line_id=line_id))
        rec_rejected=len(result)-len(rows)
        vector=features(rows,rejected,rec_rejected,[w,h],float(bitmap.mean()),self.cfg['feature_settings']['proposal_cutoffs'])
        return dict(sample_id=record['sample_id'],image_sha256=record['image_sha256'],rows=rows,words=words,
                    rejected_word_geometries=invalid_words,
                    features=vector,rejected_detection_scores=rejected,recognition_rejected=rec_rejected,
                    resized_detector_shape=[1,3,rh,rw],worker_seconds=time.perf_counter()-started)


def worker(config_path):
    cfg=json.loads(Path(config_path).read_text())
    if Path(config_path).resolve().parent.name!='approximate' or cfg.get('mode')!='approximate':
        raise ValueError('Explicit approximate worker configuration required')
    for path,expected in cfg['pinned_inputs'].items():
        if digest(path)!=expected:raise ValueError('Pinned worker input changed: '+path)
    try:
        model=CompactOCR(cfg['compact_ocr'])
        print(json.dumps(dict(status='OK',**model.ready)),flush=True)
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('command')=='stop':break
            if request.get('command') in ['reset','release_idle_cache']:
                print(json.dumps(dict(status='OK',request_id=request['request_id'])),flush=True);continue
            output=model.predict(request)
            print(json.dumps(dict(status='OK',request_id=request['request_id'],**output),ensure_ascii=False,allow_nan=False),flush=True)
    except BaseException as error:
        print(json.dumps(dict(status='FAILED',message=str(error),traceback=traceback.format_exc(),out_of_memory='out of memory' in str(error).lower())),flush=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',type=Path,required=True)
    worker(parser.parse_args().config)
