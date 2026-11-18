"""Restore the parent's declared FP32 parameters without changing frozen inputs."""
import argparse
from collections import Counter
from pathlib import Path

from . import path_retrieval_pilot_v1 as parent
from .schema import fingerprint, write_json

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'research/path_retrieval_execution_v1'
RUN=ROOT/'runs/path_retrieval_execution_v1'
load,sha,immutable=parent.load,parent.sha,parent.immutable
CONFIG=dict(version='path-retrieval-execution-v1',scientific_config=parent.CONFIG,
    correction='explicit model.float() on CPU before moving to requested device; optimizer parameters/moments FP32 as originally declared',
    previous_failed_run='runs/path_retrieval_pilot_v1',restart='same pretrained weights/seed/order; no partial optimizer/checkpoint reuse',
    new_training_data=False,new_development_data=False,new_hyperparameters=False,
    original_frozen_inputs_modified=False,precision_remains='FP32 parameters and AdamW; BF16 autocast',automatic_followup=False)


def cast_declared_model(model,device):
    model.float()
    if any(str(p.dtype)!='torch.float32' for p in model.parameters()):
        raise ValueError('Declared FP32 parameter invariant failed')
    return model.to(device)


def freeze():
    signature=parent.verify()
    paths=['reader_memory/path_retrieval_execution_v1.py','scripts/path_retrieval_execution_v1_queue.py',
        'tests/test_path_retrieval_execution_v1.py','tools/diagnose_path_optimizer_v1.py',
        'research/path_retrieval_execution_v1/PROTOCOL.md','research/path_retrieval_pilot_v1/manifest.json']
    value=dict(config=CONFIG,parent_signature=signature,files={p:sha(ROOT/p) for p in paths})
    value['signature']=fingerprint(value);immutable(PLAN/'manifest.json',value)
    print('FROZEN_PATH_EXECUTION',value['signature'],flush=True)


def verify():
    value=load(PLAN/'manifest.json')
    if value['config']!=CONFIG or value['signature']!=fingerprint({k:v for k,v in value.items() if k!='signature'}):
        raise ValueError('Execution identity changed')
    if parent.verify()!=value['parent_signature']:raise ValueError('Frozen parent identity changed')
    for p,digest in value['files'].items():
        if sha(ROOT/p)!=digest:raise ValueError('Execution file changed: '+p)
    return value['signature']


def run(mode,endpoint='http://127.0.0.1:8024'):
    sid=verify()
    original=(parent.RUN,parent.verify,parent.make_model)
    original_make_model=parent.make_model
    def make_model(device):
        tok,model=original_make_model('cpu')
        model=cast_declared_model(model,device)
        write_json(RUN/('model_dtype_'+mode+'.json'),dict(signature=sid,
            parameter_tensors_by_dtype=dict(Counter(str(p.dtype) for p in model.parameters())),
            parameters=sum(p.numel() for p in model.parameters()),explicit_cpu_float_cast=True,
            dropout_seed_and_heads_unchanged=True))
        return tok,model
    parent.RUN,parent.verify,parent.make_model=RUN,lambda:sid,make_model
    try:
        if mode=='qa':parent.qa(endpoint)
        else:getattr(parent,mode)()
    finally:parent.RUN,parent.verify,parent.make_model=original


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['freeze','verify','preflight','train','retrieve','select','qa','score'])
    p.add_argument('--endpoint',default='http://127.0.0.1:8024');args=p.parse_args()
    if args.mode=='freeze':freeze()
    elif args.mode=='verify':print(verify(),flush=True)
    else:run(args.mode,args.endpoint)
