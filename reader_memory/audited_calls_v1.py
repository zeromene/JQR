"""Append-only inference ledger with exact replay and unknown-commit protection."""
import time
from . import evidence_set_diagnostic_v1 as base
from .schema import fingerprint,read_jsonl,write_json


class Store:
    def __init__(self,run,signature,endpoint,parser,read_only=False):
        self.run,self.signature,self.endpoint,self.parser=run,signature,endpoint,parser
        self.read_only=read_only;self.used=set();self.intent=run/'inflight.json'
        rows=list(read_jsonl(run/'calls.jsonl')) if (run/'calls.jsonl').exists() else []
        self.by={r['key']:r for r in rows}
        if len(self.by)!=len(rows) or any(r['signature']!=signature for r in rows):
            raise ValueError('Invalid existing ledger')
        if self.intent.exists():
            p=base.load(self.intent)
            if p['signature']!=signature or p['key'] not in self.by:
                raise ValueError('Uncommitted inference intent; refusing hidden retry')
            if not read_only:self.intent.unlink()

    def call(self,key,phase,messages,limit,contexts=()):
        self.used.add(key);digest=fingerprint([phase,messages,limit])
        if key in self.by:
            r=self.by[key];result=r.get('result')
            if r['request_sha256']!=digest or r['messages']!=messages or r['max_tokens']!=limit or r['phase']!=phase:
                raise ValueError('Committed request changed')
            if result and len(result['completion_token_ids'])!=result['usage']['completion_tokens']:
                raise ValueError('Committed token log changed')
            if r['status']=='ok' and (not result or result['finish_reason']!='stop' or result['eos_seen'] is not True or
                                      self.parser(result['text'],phase,contexts)!=r['parsed']):
                raise ValueError('Committed output changed')
            return r
        if self.read_only:raise ValueError('Missing committed call during read-only audit')
        write_json(self.intent,{'signature':self.signature,'key':key,'request_sha256':digest})
        result=parsed=error=None;transport=False;started=time.perf_counter()
        try:
            result=base.rpc(self.endpoint,'/complete',{'messages':messages,'max_tokens':limit})
            if result.get('finish_reason')!='stop' or result.get('eos_seen') is not True:
                raise ValueError('Generation did not terminate normally')
            parsed=self.parser(result['text'],phase,contexts)
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}';transport=isinstance(exc,OSError)
        r={'signature':self.signature,'key':key,'phase':phase,'messages':messages,'max_tokens':limit,
           'request_sha256':digest,'result':result,'parsed':parsed,'status':'ok' if error is None else 'failed',
           'error':error,'wall_seconds':time.perf_counter()-started}
        base.append(self.run/'calls.jsonl',r);self.intent.unlink();self.by[key]=r
        write_json(self.run/'status.json',{'status':'running','completed_calls':len(self.by),'phase':phase})
        print('PROOF_CHECKPOINT',len(self.by),phase,r['status'],flush=True)
        if transport:raise RuntimeError('Transport failure recorded; no hidden retry')
        return r
