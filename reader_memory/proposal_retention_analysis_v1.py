"""Full-matrix, source-paired scoring; references read only after completion."""
from collections import Counter
import statistics
from . import proposal_retention_v1 as n
from . import paper_competition_v1 as p
from . import full_comparison_v1 as mu
from .evidence_acquisition_v2 import parse_answer
u=n.u


def analyze():
    sid=n.verify();assert u.load(n.RUN/'completion.json')['signature']==sid
    plans=u.load(n.BASE/'public/plans.json');mapping=u.load(n.BASE/'public/mapping.json');requests=u.load(n.BASE/'public/requests.json')
    outputs={k:u.load(n.RUN/'outputs'/(k+'.json')) for k in requests}
    ix={(v['dataset'],v['question_id'],v['extractor'],v['arm']):v for v in mapping}
    assert len(plans)==len(mapping)==768 and len(ix)==768
    for v in plans:
        m=ix[v['dataset'],v['question_id'],v['extractor'],v['arm']];assert m['plan_hash']==u.fp(v)
        k=m['key'];o=outputs[k];assert o['signature']==sid and o['key']==k==o['request_identity']
    ss=u.load(n.BASE/'public/sources.json');ids={d:{x['source']['question_id'] for x in ss if x['dataset']==d} for d in ('hotpot','musique')}
    refs={}
    for d,name in [('hotpot','hotpot_confirmation_v1'),('musique','full_comparison_v1_repair1')]:
        allrefs=u.load(n.ROOT/'research'/name/'private/references.json')
        if isinstance(allrefs,list):allrefs={v['question_id']:v for v in allrefs}
        refs[d]={q:allrefs[q] for q in ids[d]}
    metric=p.hp.official_metrics();records=[]
    for v in plans:
        d,q,e,a=(v[k] for k in ('dataset','question_id','extractor','arm'))
        key=ix[d,q,e,a]['key'];o=outputs[key];result=o['result'];hypothesis='';em=f1=0.;status='ok'
        if d=='hotpot':
            hypothesis=result['text'];status='ok' if hypothesis.strip() else 'empty'
            em=float(metric['exact_match_score'](hypothesis,refs[d][q]['answer']))
            f1=float(metric['f1_score'](hypothesis,refs[d][q]['answer'])[0])
        elif result['finish_reason']!='stop':status='output_limit'
        else:
            try:hypothesis=parse_answer(result['text'])
            except (ValueError,TypeError,KeyError):status='parse_failure'
            else:em,f1=mu.answer_scores(hypothesis,refs[d][q])
        records.append(dict(dataset=d,question_id=q,extractor=e,arm=a,f1=f1,em=em,status=status,hypothesis=hypothesis,
            key=key,evidence_tokens=v['evidence_tokens'],usage=result['usage'],proposal_fallback=v['proposal_fallback'],
            core_count=v['core_count'],core_retained=v['core_retained']))
    idx={(v['dataset'],v['question_id'],v['extractor'],v['arm']):v for v in records};groups=[];comparisons=[]
    for d in ids:
        clusters={q:q for q in ids[d]} if d=='hotpot' else mu.source_clusters(refs[d])
        for e in n.KINDS:
            for a in n.ARMS:
                rr=[idx[d,q,e,a] for q in sorted(ids[d])]
                groups.append(dict(dataset=d,extractor=e,arm=a,n=len(rr),f1=100*statistics.mean(v['f1'] for v in rr),
                    em=100*statistics.mean(v['em'] for v in rr),tokens=statistics.mean(v['evidence_tokens'] for v in rr),
                    reader_input=statistics.mean(v['usage']['prompt_tokens'] for v in rr),
                    reader_output=statistics.mean(v['usage']['completion_tokens'] for v in rr),statuses=dict(Counter(v['status'] for v in rr))))
            for control in n.ARMS[1:]:
                first={q:idx[d,q,e,'retain_fill']['f1'] for q in ids[d]};second={q:idx[d,q,e,control]['f1'] for q in ids[d]}
                comparisons.append(dict(dataset=d,extractor=e,control=control,**mu.paired_stats(first,second,clusters)))
            for q in ids[d]:
                x,y=idx[d,q,e,'retain_fill'],idx[d,q,e,'knapsack_matched']
                assert x['evidence_tokens']==y['evidence_tokens'] and x['usage']['prompt_tokens']==y['usage']['prompt_tokens']
    primary=[v for v in comparisons if v['control']=='knapsack_matched']
    # paired_stats uses percent points in delta_points and ci95_points.
    signal=all(v['delta_points']>=(2 if v['extractor']=='role' else 0) and v['delta_points']>0 for v in primary)
    signal=signal and all(v['delta_points']>=-1 for v in comparisons if v['control']!='knapsack_matched')
    costs={k:sum(v['result']['usage'].get(k,0) for v in outputs.values() if v['origin']=='new_inference')
           for k in ('prompt_tokens','completion_tokens','elapsed_seconds')}
    # Existing complete-system references, explicitly separate from controlled arms.
    assert u.sha(p.RUN/'analysis.json')==u.load(n.BASE/'manifest.json')['parent_analysis_hash']
    old=u.load(p.RUN/'analysis.json');oldgroups=[v for v in old['groups'] if v['budget']==512 or v['method']=='uncompressed']
    ap3path=p.BASE/'results/ap3_reuse_check.json'
    ap3=u.load(ap3path)['groups'] if ap3path.exists() else []
    result=dict(signature=sid,complete=True,logical_conditions=768,unique_requests=len(outputs),groups=groups,
        comparisons=comparisons,records=records,initial_preparation=u.load(n.BASE/'preparation.json'),
        preflight=u.load(n.RUN/'preflight.json'),new_qa_costs=costs,new_proposer_calls=0,
        cpu_preparation_costs=u.load(n.BASE/'cpu_preparation_costs.json'),
        inherited_proposer_costs=u.load(n.BASE/'public/proposals.json')['records'],
        inherited_reranker_costs={q:v['ranking']['costs'] for q,v in p.inputs()['inputs'].items()},
        existing_complete_system_references=oldgroups,posthoc_ap3_reference=ap3,
        decision='design_signal_requires_independent_confirmation' if signal else 'stop_current_postprocessing_mechanism_claim',
        publication_ready=False,exposed_development_only=True,automatic_followup=False,
        caveats=['Same 9B, two extraction protocols; not cross-architecture replication.',
                 'Sentence expansion and repeated-title rendering are shared diagnostics, not original quote_anchor.',
                 'Exact input length control does not equal total deployment cost or output length.',
                 'Sparse changed pairs; intervals descriptive and unadjusted; no sample expansion.'])
    u.immutable(n.RUN/'analysis.json',result)
    lines=['# 固定提案后处理验证结果','',result['decision'],'',
        '|数据|抽取协议|后处理|F1|EM|证据token|','|---|---|---|---:|---:|---:|']
    for g in groups:lines.append(f"|{g['dataset']}|{g['extractor']}|{g['arm']}|{g['f1']:.2f}|{g['em']:.2f}|{g['tokens']:.2f}|")
    lines+=['','全部主要比较逐题证据长度及完整读者输入长度相等。小型已暴露开发面板，仅作设计筛选；不是独立确认。',
        '角色/直接提案内容不变、无新增提案调用。全部题保留；格式失败按原协议评分。',
        '原始完整系统/AP3参考单列在analysis.json；当前句子化诊断不可替代原系统成绩。','']
    for v in primary:lines.append(f"- {v['dataset']}/{v['extractor']}: {v['delta_points']:+.2f}, 95% {v['ci95_points']}")
    with (n.RUN/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    from .proposal_retention_runtime_v1 import state
    state('analyzed',decision=result['decision'],logical_conditions=768)
