"""Placement-only adapters and public-input checks for the bounded screen."""
import hashlib, importlib.metadata as md
from pathlib import Path
from experiments.recent_baselines_v1.engines import GenerationEngine
from . import paper_competition_v1 as p


class CompactOffloadEngine(GenerationEngine):
    # Inherit generate_one verbatim: tokenizer BOS, native generation config,
    # output parser, precision and attention remain those of the prior adapter.
    def __init__(self,cache):
        import torch
        from accelerate import dispatch_model
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.torch=torch;self.cache=Path(cache);self.identity='CompAct-7b';self.adapters=None
        folder=p.ROOT/'models'/self.identity
        self.tokenizer=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
        self.tokenizer.padding_side='left'
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        model=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,attn_implementation='sdpa').eval()
        assert model.config.model_type=='mistral'
        assert all(x.dtype==torch.bfloat16 and x.device.type=='cpu' for x in model.parameters())
        size=lambda m:sum(x.numel()*x.element_size() for x in m.parameters())
        mapping={'lm_head':0};resident=size(model.lm_head);gpu_layers=[]
        for name,module in model.model.named_children():
            if name!='layers':mapping['model.'+name]=0;resident+=size(module)
        for i,layer in enumerate(model.model.layers):
            name='model.layers.'+str(i)
            if resident+size(layer)<=7800*2**20:
                mapping[name]=0;resident+=size(layer);gpu_layers.append(i)
            else:mapping[name]='cpu'
        assert 0<len(gpu_layers)<len(model.model.layers)
        assert all(any(n==k or n.startswith(k+'.') for k in mapping) for n,_ in model.named_parameters())
        preload=sorted({type(layer).__name__ for layer in model.model.layers})
        self.model=dispatch_model(model,device_map=mapping,main_device=0,preload_module_classes=preload).eval()
        self.placement=dict(device_map=mapping,gpu_layers=gpu_layers,resident_weights_mib=resident/2**20,
            preload_classes=preload,precision='bfloat16',quantization=False,all_compute='cuda:0')
        p.u.immutable(p.RUN/'compact_placement.json',self.placement)


def verify_compressor_models():
    expected=p.u.load(p.BASE/'manifest.json')['compressor_models']
    receipts={}
    for name,spec in expected.items():
        folder=p.ROOT/'models'/name
        actual=p.u.load(folder/'recent_baselines_manifest.json')
        assert actual['repository']==spec['repository'] and actual['revision']==spec['revision'],name
        by={x['name']:x for x in actual['files']}
        assert set(by)=={x['name'] for x in spec['files']}
        for item in spec['files']:
            path=folder/item['name'];stored=by[item['name']]
            assert path.stat().st_size==item['size']
            digest=p.u.sha(path);assert digest==stored['sha256']
            if item['sha256']:assert digest==item['sha256']
            if item['git_blob_sha1']:
                h=hashlib.sha1(('blob '+str(path.stat().st_size)+'\0').encode())
                with path.open('rb') as f:
                    for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
                assert h.hexdigest()==item['git_blob_sha1']
        receipts[name]=p.u.sha(folder/'recent_baselines_manifest.json')
    return receipts


def preflight():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(p.ROOT/'models/Qwen3.5-9B',local_files_only=True,trust_remote_code=False)
    manifest=p.u.load(p.BASE/'manifest.json')
    runtime={n:md.version(n) for n in manifest['runtime']}
    assert runtime==manifest['runtime'],(runtime,manifest['runtime'])
    cpu={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    assert cpu['MemAvailable']/1024>=32000,'CPU offload needs available RAM'
    model_receipts=verify_compressor_models()
    rows=p.sources();assert len(rows)==96 and len({r['source']['question_id'] for r in rows})==96
    assets=p.inputs();lengths=[];full=[]
    count=lambda text:len(tok.encode(text,add_special_tokens=False))
    for x in rows:
        s=x['source'];assert p.u.fp(s)==x['source_hash']
        for value in assets['inputs'][s['question_id']].values():
            assert value.get('source_hash',value.get('source_sha256'))==x['source_hash']
        units=p.it.units(s)
        for kind in ('role','direct'):
            msg=p.proposal_messages(s,kind)
            if x['dataset']=='hotpot':
                original=p.hp.messages(s,'role') if kind=='role' else p.direct.messages(s,'direct')
                assert msg==original,'Proposal prompt drift'
            n=count(tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False))
            assert n<=p.CONFIG['proposer_max_input'];lengths.append(n)
        full.append(count(p.it.render(units,list(range(len(units))))))
    plans=p.u.load(p.BASE/'public/fixed_plans.json');assert len(plans)==1008
    by={r['source']['question_id']:r['source'] for r in rows}
    replay=p.u.load(p.BASE/'public/replay.json')
    for item in plans:
        assert count(item['evidence_text'])==item['evidence_tokens']
        assert item['budget']==0 or item['evidence_tokens']<=item['budget']
        req=p.qa_request(item,by[item['question_id']]);key=req['key']
        if key in replay:
            r=replay[key];assert r['request_identity']==key
            n=count(tok.apply_chat_template(req['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False))
            assert r['result']['usage']['prompt_tokens']==n,'Replayed tokenizer/request mismatch'
    p.u.immutable(p.RUN/'preflight.json',dict(signature=p.u.sha(p.BASE/'manifest.json'),passed=True,
        runtime=runtime,model_receipts=model_receipts,source_count=96,fixed_plans=len(plans),
        max_proposer_input=max(lengths),max_full_evidence=max(full),private_references_read=False))
