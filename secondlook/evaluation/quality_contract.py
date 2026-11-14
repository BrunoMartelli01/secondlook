"""Model-independent identity of the quality task used to train a calibrator.

Different images/models may share a contract. Different prediction units,
evaluation code or care/matching parameters may not silently share calibration.
Dataset paths and model identities deliberately do not enter the signature.
Matching contracts are necessary for reuse; they do not establish calibration
transfer across different model/feature distributions."""

from dataclasses import dataclass

import hashlib

import json

import re

@dataclass(frozen=True)
class QualityContract:
    prediction_unit: str
    aggregation: str
    evaluator_sha256: str
    parameters_json: str

    def __post_init__(self):
        if not isinstance(self.prediction_unit,str) or not self.prediction_unit.strip():raise ValueError('Explicit prediction unit required')
        if self.aggregation!='pooled_micro_f1':raise ValueError('Explicit pooled_micro_f1 quality contract required')
        if re.fullmatch('[0-9a-f]{64}',self.evaluator_sha256) is None:raise ValueError('Pinned evaluator SHA256 required')
        params=json.loads(self.parameters_json)
        if not isinstance(params,dict) or not params:raise ValueError('Explicit native evaluation parameters required')
        canonical=json.dumps(params,sort_keys=True,separators=(',',':'),allow_nan=False)
        object.__setattr__(self,'parameters_json',canonical)

    @classmethod
    def create(cls,prediction_unit,evaluator_sha256,parameters):
        return cls(prediction_unit,'pooled_micro_f1',evaluator_sha256,json.dumps(parameters,allow_nan=False))

    @property
    def parameters(self):return json.loads(self.parameters_json)

    @property
    def signature(self):
        payload=json.dumps(dict(prediction_unit=self.prediction_unit,aggregation=self.aggregation,evaluator_sha256=self.evaluator_sha256,parameters=self.parameters),sort_keys=True,separators=(',',':'),allow_nan=False)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def require_compatible(self,deployment):
        if self.signature!=deployment.signature:
            different=[name for name in ['prediction_unit','aggregation','evaluator_sha256','parameters_json'] if getattr(self,name)!=getattr(deployment,name)]
            raise ValueError('Calibration/deployment quality contracts differ ('+', '.join(different)+'). Relabel and calibrate source outputs with the deployment task, or report an explicitly separate transfer diagnostic.')
