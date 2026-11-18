"""A small LoRA adapter on the pinned semantic ranker, scoring complete evidence sets."""
import math
import re

from .mainline_reranker_v1 import PREFIX,SUFFIX

INSTRUCTION=('Given the question and the complete evidence context, judge whether a fixed question-answering '
    'model can produce a correct concise answer grounded in this context. Consider direct evidence, '
    'required relations, entity bindings, conflicting summaries and distractions together.')
CONFIG=dict(seed=29,epochs=2,learning_rate=1e-4,weight_decay=.01,accumulation=8,
    rank=8,alpha=16,last_layers=8,max_tokens=8192,max_pairs_per_question=12,
    gradient_clip=1.,regression_weight=.25,difference_temperature=4.,checkpoint='fixed last epoch')


def input_ids(tokenizer,question,context):
    body=f'<Instruct>: {INSTRUCTION}\n<Query>: {question}\n<Document>: {context}'
    ids=tokenizer.encode(PREFIX,add_special_tokens=False)+tokenizer.encode(body,add_special_tokens=False)+tokenizer.encode(SUFFIX,add_special_tokens=False)
    if len(ids)>CONFIG['max_tokens']:raise ValueError('Complete set exceeds8192 ranker tokens; no truncation')
    return ids


def add_adapters(model,torch):
    class LoRALinear(torch.nn.Module):
        def __init__(self,base):
            super().__init__();self.base=base
            self.a=torch.nn.Parameter(torch.empty(CONFIG['rank'],base.in_features,device=base.weight.device,dtype=torch.float32))
            self.b=torch.nn.Parameter(torch.zeros(base.out_features,CONFIG['rank'],device=base.weight.device,dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
        def forward(self,x):
            base=self.base(x)
            value=torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.a),self.b)
            return base+value.to(base.dtype)*(CONFIG['alpha']/CONFIG['rank'])
    for p in model.parameters():p.requires_grad_(False)
    paths=[]
    total=model.config.num_hidden_layers
    for name,module in list(model.named_modules()):
        match=re.search(r'(?:^|\.)layers\.(\d+)\.self_attn\.(q_proj|v_proj)$',name)
        if match and int(match.group(1))>=total-CONFIG['last_layers']:
            parent_name,attribute=name.rsplit('.',1)
            setattr(model.get_submodule(parent_name),attribute,LoRALinear(module));paths.append(name)
    if len(paths)!=CONFIG['last_layers']*2:raise ValueError('Unexpected ranker architecture for adapter')
    return paths


def load(model_path,training=False):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(8);torch.manual_seed(CONFIG['seed']);torch.cuda.manual_seed_all(CONFIG['seed'])
    tok=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True,padding_side='left')
    model=AutoModelForCausalLM.from_pretrained(str(model_path),local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    paths=add_adapters(model,torch)
    model.config.use_cache=False
    if training:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.train()
    else:model.eval()
    return torch,tok,model,paths


def logit(torch,tokenizer,model,ids):
    value=torch.tensor([ids],device='cuda',dtype=torch.long)
    logits=model(input_ids=value,attention_mask=torch.ones_like(value),logits_to_keep=1,use_cache=False).logits[0,-1].float()
    return logits[tokenizer.convert_tokens_to_ids('yes')]-logits[tokenizer.convert_tokens_to_ids('no')]


def adapter_state(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items() if k.endswith('.a') or k.endswith('.b')}


def load_adapter(model,state):
    expected={k for k,p in model.named_parameters() if p.requires_grad}
    if set(state)!=expected:raise ValueError('Adapter checkpoint keys differ from architecture')
    own=dict(model.named_parameters())
    for key,value in state.items():
        if own[key].shape!=value.shape:raise ValueError('Adapter checkpoint shape differs')
        own[key].data.copy_(value.to(own[key].device))
