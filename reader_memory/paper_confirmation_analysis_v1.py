"""One frozen evaluation after every request completes; honest scope and full costs."""
from collections import Counter,defaultdict
import statistics,json
from . import paper_confirmation_v1 as n
from . import full_comparison_v1 as mu
from .evidence_acquisition_v2 import parse_answer
u=n.u;ROOT=n.ROOT;RUN=n.RUN;BASE=n.BASE

def paired(values,clusters):
    import numpy as np
    grouped=defaultdict(list)
    for q,value in values.items():grouped[clusters[q]].append(value)
    groups=[grouped[k] for k in sorted(grouped)];sums=np.array([sum(v) for v in groups]);sizes=np.array([len(v) for v in groups])
    rng=np.random.default_rng(n.CONFIG['seed']);sample=rng.integers(len(groups),size=(n.CONFIG['bootstrap'],len(groups)))
    means=sums[sample].sum(1)/sizes[sample].sum(1);delta=list(values.values())
    return dict(n=len(delta),clusters=len(groups),delta_points=100*statistics.mean(delta),
                ci95_points=(100*np.quantile(means,[.025,.975])).tolist(),
                wins=sum(v>1e-9 for v in delta),ties=sum(abs(v)<=1e-9 for v in delta),losses=sum(v< -1e-9 for v in delta))

def references():
    # Do not call until the complete request matrix has been verified.
    import pyarrow.dataset as ds
    alloc=u.load(BASE/'allocation.json')['selected'];refs={}
    hp={v['native_id'] for v in alloc if v['dataset']=='hotpot'}
    manifest=u.load(ROOT/'data/hotpot_train_pinned_v1/source_manifest.json')
    for file in manifest['files']:
        assert u.sha(ROOT/file['path'])==file['sha256']
        table=ds.dataset(ROOT/file['path'],format='parquet').to_table(columns=['id','answer'],filter=ds.field('id').isin(sorted(hp)))
        for row in table.to_pylist():refs['hotpot:'+row['id']]=dict(answer=row['answer'],aliases=[])
    assert len(refs)==1400
    wanted={v['native_id'] for v in alloc if v['dataset']=='musique'}
    path=ROOT/'data/musique/musique_ans_v1.0_train.jsonl'
    assert u.sha(path)=='83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a'
    for line in path.open(encoding='utf-8'):
        row=json.loads(line)
        if row['id'] in wanted:refs[row['id']+':multi']=dict(answer=row['answer'],aliases=row.get('answer_aliases',[]))
    assert len(refs)==1935
    return refs

def analyze():
    from .paper_confirmation_runtime_v1 import put,state
    sid=n.verify();assert u.load(RUN/'completion.json')['signature']==sid
    plans=u.load(RUN/'plans.json');mapping=u.load(RUN/'mapping.json');requests=u.load(RUN/'requests.json')
    assert len(plans)==len(mapping)==n.CONFIG['logical_conditions']
    assert {p.stem for p in (RUN/'outputs').glob('*.json')}==set(requests)
    outputs={k:u.load(RUN/'outputs'/(k+'.json')) for k in requests};lengths=u.load(RUN/'prompt_lengths.json')
    for k,v in outputs.items():
        r=requests[k];assert v['signature']==sid and k==v['key']==v['request_identity']==u.fp([r['reader'],r['messages'],r['max_output']])
        assert v['result']['finish_reason'] in ('stop','length') and isinstance(v['result']['text'],str)
        assert v['result']['usage']['prompt_tokens']==lengths[k]
    for p,m in zip(plans,mapping):assert m['plan_hash']==u.fp(p) and (m['key'] in outputs if m['key'] else p['status']=='selection_failed')
    refs=references();metric=n.old.hp.official_metrics();records=[];sources=n.sources()
    metadata={('hotpot:'+v['native_id'] if v['dataset']=='hotpot' else v['native_id']+':multi'):v for v in u.load(BASE/'allocation.json')['selected']}
    for p,m in zip(plans,mapping):
        d,q=p['dataset'],p['question_id'];key=m['key'];hypothesis='';em=f1=0.;status=p['status'];usage={};finish=None
        if key:
            result=outputs[key]['result'];usage=result['usage'];finish=result['finish_reason']
            if d=='hotpot':
                hypothesis=result['text'];status='ok' if hypothesis.strip() else 'empty'
                em=float(metric['exact_match_score'](hypothesis,refs[q]['answer']));f1=float(metric['f1_score'](hypothesis,refs[q]['answer'])[0])
            elif finish!='stop':status='output_limit'
            else:
                try:hypothesis=parse_answer(result['text'])
                except (ValueError,KeyError,TypeError):status='parse_failure'
                else:em,f1=mu.answer_scores(hypothesis,refs[q]);status='ok'
        records.append(dict(dataset=d,question_id=q,study=p['study'],reader=p['reader'],extractor=p['extractor'],method=p['method'],
            f1=f1,em=em,hypothesis=hypothesis,status=status,finish_reason=finish,key=key,usage=usage,
            qwen_evidence_tokens=p['evidence_tokens'],actual_reader_evidence_tokens=p.get('actual_reader_evidence_tokens',0),
            proposal_fallback=p.get('proposal_fallback'),core_count=p.get('core_count'),core_retained=p.get('core_retained')))
    groups=[];comparisons=[];hopgroups=[]
    ix={(r['dataset'],r['question_id'],r['study'],r['reader'],r['extractor'],r['method']):r for r in records};assert len(ix)==len(records)
    for study in ('complete','retention'):
        readers=n.CONFIG['readers'] if study=='complete' else ['Qwen3.5-9B']
        extractors=[None] if study=='complete' else ['role','direct']
        methods=n.CONFIG['methods'] if study=='complete' else list(n.retention.ARMS)
        for d in ('hotpot','musique'):
            ids=sorted(s['source']['question_id'] for s in sources if s['dataset']==d and (study=='complete' or s['diagnostic']))
            clusters={q:metadata[q]['component_id'] if d=='hotpot' else metadata[q]['support_document_component'] for q in ids}
            for reader in readers:
                for extractor in extractors:
                    for method in methods:
                        rr=[ix[d,q,study,reader,extractor,method] for q in ids]
                        group=dict(dataset=d,study=study,reader=reader,extractor=extractor,method=method,n=len(rr),
                            f1=100*statistics.mean(r['f1'] for r in rr),em=100*statistics.mean(r['em'] for r in rr),
                            qwen_evidence_tokens=statistics.mean(r['qwen_evidence_tokens'] for r in rr),
                            actual_reader_evidence_tokens=statistics.mean(r['actual_reader_evidence_tokens'] for r in rr),
                            prompt_tokens=statistics.mean(r['usage'].get('prompt_tokens',0) for r in rr),
                            output_tokens=statistics.mean(r['usage'].get('completion_tokens',0) for r in rr),statuses=dict(Counter(r['status'] for r in rr)))
                        groups.append(group)
                        if d=='musique' and study=='complete':
                            for hops in (2,3,4):
                                hh=[r for r in rr if metadata[r['question_id']]['hops']==hops]
                                hopgroups.append(dict(reader=reader,method=method,hops=hops,n=len(hh),f1=100*statistics.mean(r['f1'] for r in hh)))
                    first=methods[0]
                    for control in methods[1:]:
                        values={q:ix[d,q,study,reader,extractor,first]['f1']-ix[d,q,study,reader,extractor,control]['f1'] for q in ids}
                        comparisons.append(dict(dataset=d,study=study,reader=reader,extractor=extractor,method=first,control=control,**paired(values,clusters)))
                    if study=='retention':
                        for q in ids:
                            a,b=(ix[d,q,study,reader,extractor,m] for m in ('retain_fill','knapsack_matched'))
                            assert a['qwen_evidence_tokens']==b['qwen_evidence_tokens'] and a['usage']['prompt_tokens']==b['usage']['prompt_tokens']
    primary=next(v for v in comparisons if v['study']=='complete' and v['dataset']=='hotpot' and v['reader']=='Qwen3.5-9B' and v['control']=='rank_add')
    primary_pass=primary['delta_points']>=2 and primary['ci95_points'][0]>0
    transfers=[v for v in comparisons if v['study']=='complete' and v['control']=='rank_add' and v is not primary]
    transfer_pass=all(v['delta_points']>0 and v['ci95_points'][0]>0 for v in transfers)
    diagnostic=[v for v in comparisons if v['study']=='retention' and v['control']=='knapsack_matched']
    design_pass=all(v['delta_points']>=(2 if v['extractor']=='role' else 0) and v['ci95_points'][0]>0 for v in diagnostic)
    design_pass=design_pass and all(v['delta_points']>=-1 for v in comparisons if v['study']=='retention' and v['control']!='knapsack_matched')
    # Every deployed strategy pays its proposer/retriever/ranker even when computation was shared during evaluation.
    deployment=[]
    for x in sources:
        s=x['source'];q=s['question_id'];d=x['dataset'];base={}
        for phase in ('retrieve','rank'):
            v=u.load(RUN/'cache'/phase/(u.fp(q)+'.json'));base[phase]=dict(seconds=v['seconds'],**v['value']['costs'])
        for method,kind in [('quote_anchor','role'),('rank_add','direct'),('ap3',None),('strong_rerank',None)]:
            proposal=[] if kind is None else u.load(RUN/'cache/proposal'/(u.fp([q,kind])+'.json'))['value']['attempts']
            deployment.append(dict(dataset=d,question_id=q,method=method,reranker=base['rank'],
                retrieval=base['retrieve'] if method!='strong_rerank' else None,proposer_model='Qwen3.5-9B' if kind else None,
                proposer_calls=len(proposal),proposer_prompt_tokens=sum(a['result']['usage']['prompt_tokens'] for a in proposal),
                proposer_output_tokens=sum(a['result']['usage']['completion_tokens'] for a in proposal),
                proposer_seconds=sum(a['result']['usage']['elapsed_seconds'] for a in proposal)))
    decision=('independent_confirmation_passed_draft_requires_full_review' if primary_pass and transfer_pass and design_pass else
              'stop_current_quality_mechanism_claim_no_adaptive_expansion')
    result=dict(signature=sid,complete=True,logical_conditions=len(records),unique_requests=len(outputs),groups=groups,
        comparisons=comparisons,primary=primary,primary_pass=primary_pass,transfer_pass=transfer_pass,design_pass=design_pass,
        hop_groups=hopgroups,records=records,deployment_stage_costs=deployment,decision=decision,publication_ready=False,
        limitations=['MuSiQue is question/atomic holdout, not unseen-fact holdout; 491/535 are 2-hop.',
                    'Phi receives Qwen-budgeted strings; actual lengths differ, no Phi length matching claim.',
                    'Only the Hotpot Qwen comparison is primary; other individual intervals descriptive. All designated gates must pass.',
                    'Shared GPU time is descriptive, not controlled speed benchmarking; tokenizer counts are model-specific.',
                    'No claim of role causality, recovered complete chains, or globally new SOTA.'])
    put(RUN/'analysis.json',result)
    lines=['# 独立题目确认结果','',decision,'',
           '|数据|读者|方法|F1|EM|Qwen证据token|实际读者证据token|',
           '|---|---|---|---:|---:|---:|---:|']
    for g in groups:
        if g['study']=='complete':lines.append(f"|{g['dataset']}|{g['reader']}|{g['method']}|{g['f1']:.2f}|{g['em']:.2f}|{g['qwen_evidence_tokens']:.1f}|{g['actual_reader_evidence_tokens']:.1f}|")
    lines+=['',f"Hotpot/Qwen 主比较：{primary['delta_points']:+.2f} F1，95% {primary['ci95_points']}。",'',
            'MuSiQue 是新题/新子问题验证，不是新事实验证；Phi 为相同材料迁移。此状态不是自动达到投稿标准。']
    path=RUN/'REPORT.md'
    if not path.exists():path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    state('analyzed',decision=decision,primary_pass=primary_pass,transfer_pass=transfer_pass,design_pass=design_pass)

if __name__=='__main__':analyze()
