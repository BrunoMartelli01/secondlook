"""Observe exact author export row identities without changing its decisions.

The original method/source remains untouched. A copied AST gets one observer
after each successful canonical prediction write. Native intermediate rows and
final ZIP bodies are checked, avoiding approximate text/bbox identity matching."""

import ast

from collections import defaultdict

import importlib

import inspect

from pathlib import Path

import sys

import textwrap

import types

import zipfile

from secondlook.evaluation.protocol import read_json, write_json, sha256

def observed_sort(original):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)));sites=[]
    class Observer(ast.NodeTransformer):
        def visit_Expr(self,node):
            self.generic_visit(node)
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='fout' and node.value.func.attr=='writelines':
                sites.append(node);return [node,ast.parse('self._row_identity_trace.append((i,iline,outstr))').body[0]]
            return node
    tree=Observer().visit(tree)
    if len(sites)!=1:raise ValueError('Expected exactly one native canonical write observer site')
    ast.fix_missing_locations(tree);namespace=original.__globals__.copy();exec(compile(tree,inspect.getsourcefile(original),'exec'),namespace);return namespace[original.__name__]

def exporter(config):
    from secondlook.evaluation.existing_packages import reuse_existing_packages
    scope=config['runtime']['reuse_existing_packages'];reuse_existing_packages(scope['site_packages'],scope['imports']);source=Path(config['expert']['asset_root'])/('TiCLS-'+config['expert']['source_revision'])
    for name,relative in [('adet','adet'),('adet.modeling','adet/modeling'),('adet.evaluation','adet/evaluation')]:
        if name not in sys.modules:
            package=types.ModuleType(name);package.__path__=[str(source/relative)];sys.modules[name]=package
    cls=importlib.import_module('adet.evaluation.text_evaluation_all').TextEvaluator
    if Path(inspect.getsourcefile(cls.sort_detection)).resolve()!=(source/'adet/evaluation/text_evaluation_all.py').resolve():raise ValueError('Another author converter is already imported')
    obj=object.__new__(cls);obj._row_identity_trace=[];observed=observed_sort(cls.sort_detection)
    def sort_with_snapshot(self,temp_dir):
        # Native sort removes intermediate files before returning. Observe them
        # while they exist, before executing the unchanged sorting decisions.
        self._intermediate_rows={p.name:p.read_text().splitlines() for p in Path(temp_dir).glob('*.txt')}
        return observed(self,temp_dir)
    obj.sort_detection=types.MethodType(sort_with_snapshot,obj);return obj,source/'adet/evaluation/text_evaluation_all.py'

def export_with_identities(config,dataset,rows,out):
    from secondlook.spotters.ticls_export import export_author_words,submission_identity
    benchmark=read_json(config['evaluation']['benchmark_config']);spec=benchmark['datasets'][dataset]
    if spec['lexicon']!='None':raise ValueError('Word trust labels require the explicit None protocol')
    mapping=read_json(spec['spotting_id_map']) if dataset=='totaltext' else None;obj,source=exporter(config);result=export_author_words(config,dataset,rows,out,obj);numbers={};ordered={};intermediate={}
    for sid,sample in rows.items():
        number,filename=submission_identity(dataset,sid,mapping);key='{:07d}.txt'.format(number);numbers[key]=(sid,filename);candidates=[]
        for index,row in enumerate(sample['rows']):
            if float(row['score'])<=.1:continue
            points=[(int(p[0]),int(p[1])) for p in row['polys']];coordinates=','.join(str(v) for p in points for v in p);text=str(row['rec'])
            # This qualification fails on delimiters/newlines instead of guessing
            # an identity after native string splitting or trimming.
            if any(token in text for token in ['\n','\r',',####']):raise ValueError('Unsupported raw text delimiter in identity trace')
            candidates.append(((-round(float(row['score']),3),float(points[0][0]),float(points[0][1])),index,coordinates+',####'+text.rstrip()))
        candidates.sort(key=lambda item:item[0]);ordered[key]=[v[1] for v in candidates];intermediate[key]=[v[2] for v in candidates]
        actual=obj._intermediate_rows[key]
        if actual!=intermediate[key]:raise ValueError('Native intermediate ordering/text differs from identity reconstruction '+sid)
    captured=defaultdict(list);identities={s:[] for s in rows};bodies={s:[] for s in rows}
    for file_name,line_index,body in obj._row_identity_trace:
        key=Path(file_name).name
        if key not in numbers or not 0<=line_index<len(ordered[key]):raise ValueError('Native trace image/line out of range')
        sid,_=numbers[key];captured[sid].append(line_index);identities[sid].append(ordered[key][line_index]);bodies[sid].append(body+'\n')
    with zipfile.ZipFile(result['submission']) as z:
        for key,(sid,filename) in numbers.items():
            body=''.join(bodies[sid])
            if z.read(filename).decode('utf-8')!=body:raise ValueError('Observed native writes do not exactly reproduce canonical submission '+sid)
            if len(identities[sid])!=len(set(identities[sid])):raise ValueError('Duplicate raw row identity')
    identity=dict(schema_version=1,raw_indices=identities,canonical_lines=bodies,source_sha256=sha256(source),observer_source_sha256=sha256(__file__),qualification='Exact native intermediate ordering and final body equality for every image; one AST observer after successful author writes',images=len(rows),raw_rows=sum(len(v['rows']) for v in rows.values()),canonical_rows=sum(map(len,identities.values())),dropped_by_author=sum(len(v['rows']) for v in rows.values())-sum(map(len,identities.values())))
    write_json(out/'row_identities.json',identity);return result,identity
