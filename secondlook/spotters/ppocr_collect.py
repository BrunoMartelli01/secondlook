"""Compact OCR collection, source-only routing and unchanged TiCLS fallback."""

import json

import os

import subprocess

import uuid

from secondlook.spotters.ppocr import FEATURE_NAMES

from secondlook.evaluation.protocol import write_json

class CompactWorker:
    def __init__(self,cfg,config_path,directory):
        directory.mkdir(parents=True,exist_ok=False)
        env=dict(os.environ)
        paths=cfg['compact_runtime']['cuda_library_paths']
        env['LD_LIBRARY_PATH']=':'.join(paths)+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        self.log=(directory/'stderr.log').open('w')
        self.process=subprocess.Popen([cfg['compact_runtime']['python'],'-m','secondlook.spotters.ppocr',
            '--config',str(config_path)],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        self.ready=self._receive()
        if self.ready['feature_names']!=list(FEATURE_NAMES):raise ValueError('Worker feature contract mismatch')
        write_json(directory/'ready.json',self.ready)

    def _receive(self,request_id=None):
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Compact worker exited; inspect preserved stderr')
        value=json.loads(line)
        if value.get('status')!='OK':raise RuntimeError('Compact worker failed: '+json.dumps(value))
        if request_id is not None and value.get('request_id')!=request_id:raise ValueError('Out-of-order compact response')
        return value

    def request(self,record):
        request=dict(record,request_id=uuid.uuid4().hex)
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        return self._receive(request['request_id'])

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"command":"stop"}\n');self.process.stdin.flush();self.process.wait(timeout=60)
        self.log.close()

def valid_quad(points):
    """Simple nondegenerate quadrilateral; no external geometry runtime needed."""
    p=[tuple(x) for x in points]
    if len(p)!=4:raise ValueError('Compact detector must return four corners')
    # Adjacent repeated vertices describe a triangle and are permitted by the
    # native polygon validator. Nonadjacent repeats are self-touching.
    p=[v for i,v in enumerate(p) if v!=p[i-1]]
    if len(p)<3 or len(set(p))!=len(p):return False
    area=sum(p[i][0]*p[(i+1)%len(p)][1]-p[(i+1)%len(p)][0]*p[i][1] for i in range(len(p)))
    if area==0:return False
    if len(p)==3:return True
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def on(a,b,c):return min(a[0],b[0])<=c[0]<=max(a[0],b[0]) and min(a[1],b[1])<=c[1]<=max(a[1],b[1])
    def intersects(a,b,c,d):
        x,y,z,w=cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
        return (x*y<0 and z*w<0) or (x==0 and on(a,b,c)) or (y==0 and on(a,b,d)) or (z==0 and on(c,d,a)) or (w==0 and on(c,d,b))
    return not intersects(p[0],p[1],p[2],p[3]) and not intersects(p[1],p[2],p[3],p[0])

def format_rows(rows,dataset,lexicon=None):
    from secondlook.evaluation.official_benchmarks import submission_line
    lines=[];rejected=[]
    for index,row in enumerate(rows):
        points=[list(p) for p in row['points']]
        if not valid_quad(points):
            rejected.append(dict(index=index,reason='invalid predicted polygon',prediction=row));continue
        text=row['text']
        if lexicon is not None:
            matched=lexicon.nearest(text)
            if matched is None:continue
            text=matched['text']
        area=sum(points[i][0]*points[(i+1)%len(points)][1]-points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points)))
        if dataset=='totaltext' and area>0:points.reverse()
        lines.append(submission_line(points,text,dataset))
    return '\n'.join(lines)+('\n' if lines else ''),rejected
