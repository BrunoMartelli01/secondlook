"""Read native RRC detection counters without changing its matching computation."""

import ast

import inspect

import sys

import textwrap

from secondlook.evaluation.official_benchmarks import parse_upstream_metrics

def evaluate_with_detection_counts(module,gt,submission,params):
    function=module.evaluate_method
    source,start=inspect.getsourcelines(function)
    tree=ast.parse(textwrap.dedent("".join(source)))
    points=[node for node in ast.walk(tree) if isinstance(node,ast.AugAssign)
            and isinstance(node.target,ast.Name) and node.target.id=="matchedSum"
            and isinstance(node.value,ast.Name) and node.value.id=="detCorrect"]
    if len(points)!=1:raise ValueError("Pinned upstream per-image counter boundary changed")
    target_line=start+points[0].lineno-1;captured={}
    if sys.gettrace() is not None:raise RuntimeError("Counter observation requires an otherwise untraced evaluator")
    def observe(frame,event,arg):
        if frame.f_code is not function.__code__:return None
        if event=="line" and frame.f_lineno==target_line:
            values=frame.f_locals;sid=values["resFile"]
            if sid in captured:raise ValueError("Duplicate native detection sample")
            captured[sid]=dict(correct=int(values["detOnlyCorrect"]),
                predictions=int(values["det_only_numDetCare"]),gt_words=int(values["det_only_numGtCare"]),
                e2e_correct=int(values["detCorrect"]),e2e_predictions=int(values["numDetCare"]),
                e2e_gt_words=int(values["numGtCare"]),
                e2e_localized=int(values["gtRectMat"].sum()) if values["gtPols"] and values["detPols"] else 0)
            both=bool(values["gtPols"]) and bool(values["detPols"])
            captured[sid].update(e2e_correct_indices=[int(i) for i in values["detMatchedNums"]],
                e2e_matched_indices=values["detRectMat"].nonzero()[0].tolist() if both else [],
                detection_matched_indices=values["det_only_detRectMat"].nonzero()[0].tolist() if both else [],
                e2e_ignored_indices=[int(i) for i in values["detDontCarePolsNum"]],
                detection_ignored_indices=[int(i) for i in values["det_only_detDontCarePolsNum"]])
            if len(captured[sid]["e2e_correct_indices"])!=captured[sid]["e2e_correct"]:
                raise ValueError("Native per-word correctness indices differ from native counts")
        return observe
    sys.settrace(observe)
    try:result=function(gt,submission,params)
    finally:sys.settrace(None)
    if set(captured)!=set(result["per_sample"]):raise ValueError("Native detection counters omitted an image")
    correct=sum(row["correct"] for row in captured.values())
    denominator=sum(row["predictions"]+row["gt_words"] for row in captured.values())
    hmean=2*correct/denominator if denominator else 0.
    if abs(hmean-parse_upstream_metrics(result["det_only_method"])["hmean"])>1e-12:
        raise ValueError("Native detection counter aggregate differs from unchanged official output")
    correct=sum(row["e2e_correct"] for row in captured.values())
    denominator=sum(row["e2e_predictions"]+row["e2e_gt_words"] for row in captured.values())
    hmean=2*correct/denominator if denominator else 0.
    if abs(hmean-parse_upstream_metrics(result["e2e_method"])["hmean"])>1e-12:
        raise ValueError("Native end-to-end aggregate differs from unchanged official output")
    return result,captured
