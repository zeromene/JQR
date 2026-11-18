"""Gold-blind BF16 controls with exactly matched call counts AND input-token totals."""
import argparse
from collections import Counter
import itertools
import statistics
from pathlib import Path

from . import replacement_precision_v2 as frozen
from . import replacement_ranker_v1 as ranker
from . import replacement_stage2_common as common
from .schema import fingerprint, read_jsonl, write_json

ROOT = common.ROOT
PLAN = ROOT / 'research/replacement_controls_v3'
RUN = ROOT / 'runs/replacement_controls_v3'
ARMS = ('wide_bf16','matched_single_bf16','matched_union_bf16')
CONFIG = dict(arms=list(ARMS),precision='frozen BF16',budget=2048,
    matching='all original single proposals mandatory; maximum common extra count and exact ranker-input-token sum',
    matching_tie='minimum sha256(seed,question_id,extra_count,token_sum); first subset in proposal order',
    seed='exact-compute-replacement-20261028-v1',heldout_accessed=False)


def freeze():
    paths = ['reader_memory/replacement_controls_v3.py','reader_memory/replacement_stage2_common.py',
             'research/replacement_controls_v3/PROTOCOL.md','tests/test_replacement_stage2.py',
             'runs/replacement_precision_v2/scored.json','runs/replacement_precision_v2/qa/answer_calls.jsonl',
             'runs/replacement_learning_v1_repair1/rewards/predictions.jsonl',
             'runs/replacement_learning_v1_repair1/rewards/answer_calls.jsonl']
    paths += [str(p.relative_to(ROOT)) for p in sorted((frozen.RUN/'sets').glob('*.json'))]
    sid = common.identity(PLAN,CONFIG,paths); print('FROZEN_CONTROLS',sid,flush=True)


def exact_match(mandatory, single, group, lengths, question_id):
    """Choose subsets without accessing model scores, answers, or observed outcomes."""
    required = {p['evidence_sha256'] for p in mandatory}
    if len(required)!=len(mandatory): raise ValueError('Duplicate mandatory proposal')
    for arm in [single,group]:
        hashes=[p['evidence_sha256'] for p in arm]
        if len(set(hashes))!=len(hashes) or not required<=set(hashes):
            raise ValueError('Control lost an original single path or duplicated a proposal')
    extras = [[p for p in arm if p['evidence_sha256'] not in required] for arm in [single,group]]
    if max(map(len,extras),default=0)>18: raise ValueError('Unexpected control size; review enumeration bound')
    tables = []
    for values in extras:
        table = {}
        for count in range(len(values)+1):
            for indices in itertools.combinations(range(len(values)),count):
                size = sum(lengths[values[i]['evidence_sha256']] for i in indices)
                table.setdefault((count,size),indices)
        tables.append(table)
    keys = set(tables[0]) & set(tables[1])
    count = max(k[0] for k in keys)
    key = min((k for k in keys if k[0]==count),key=lambda k:fingerprint([CONFIG['seed'],question_id,*k]))
    selected = [list(mandatory)+[extras[a][i] for i in tables[a][key]] for a in [0,1]]
    stats = dict(shared_sets=len(mandatory),extra_sets=count,calls=len(selected[0]),
                 input_tokens=sum(lengths[p['evidence_sha256']] for p in selected[0]),
                 informative={p['evidence_sha256'] for p in selected[0]}!={p['evidence_sha256'] for p in selected[1]})
    assert len(selected[0])==len(selected[1])
    assert stats['input_tokens']==sum(lengths[p['evidence_sha256']] for p in selected[1])
    return selected[0],selected[1],stats


def prepare():
    sid = common.verify(PLAN,CONFIG)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(frozen.parent.MODEL),local_files_only=True)
    questions = {q['question_id']:q for q in common.load(frozen.base.PLAN/'public/queries.json')}
    plans = []; audits = []
    for path in sorted((frozen.RUN/'sets').glob('*.json')):
        saved = common.load(path); qid = saved['plans'][0]['question_id']; q = questions[qid]
        alts = saved['alternatives']; wide = alts['wide_single_fp32']; union = alts['union_fp32']; mandatory = alts['single_fp32']
        allsets = frozen.ordered_union(union,wide)
        lengths = {p['evidence_sha256']:len(ranker.input_ids(tokenizer,q['question'],p['evidence_text'])) for p in allsets}
        single, group, audit = exact_match(mandatory,wide,union,lengths,qid)
        candidates = dict(wide_bf16=wide,matched_single_bf16=single,matched_union_bf16=group)
        choices = []
        for arm, values in candidates.items():
            best = max(values,key=lambda p:saved['scores'][p['evidence_sha256']]['bf16'])
            row = dict(**q,arm=arm,**best,scored_sets=len(values),
                       score_input_tokens=sum(lengths[p['evidence_sha256']] for p in values),private_inputs_read=False)
            plans.append(row); choices.append(row)
        value = dict(signature=sid,question_id=qid,audit=audit,alternatives=candidates,plans=choices,
                     input_lengths=lengths,source_sha256=frozen.base.sha(path),new_scorer_calls=0)
        common.immutable_json(RUN/'selected'/path.name,value); audits.append(dict(question_id=qid,**audit))
    common.immutable_json(RUN/'qa_plans.json',plans)
    common.immutable_json(RUN/'matching_audit.json',dict(signature=sid,questions=len(audits),
        informative_questions=sum(a['informative'] for a in audits),all_calls_exact=True,all_input_tokens_exact=True,rows=audits))
    print('PREPARED_CONTROLS',len(plans),'informative',sum(a['informative'] for a in audits),flush=True)


def qa(endpoint):
    sid = common.verify(PLAN,CONFIG)
    common.run_qa(RUN,common.load(RUN/'qa_plans.json'),sid,endpoint)


def score():
    sid = common.verify(PLAN,CONFIG)
    rows = common.official_rows(common.load(RUN/'qa_plans.json'),list(read_jsonl(RUN/'qa/predictions.jsonl')))
    previous = common.load(frozen.RUN/'scored.json')
    by_arm = {a:{r['question_id']:r['official_native_f1'] for r in rows if r['arm']==a} for a in ARMS}
    by_arm['full_union_bf16'] = {r['question_id']:r['official_native_f1'] for r in previous if r['arm']=='union_bf16'}
    matching=common.load(RUN/'matching_audit.json')
    informative={r['question_id'] for r in matching['rows'] if r['informative']}
    comparisons = []
    for a,b in [('full_union_bf16','wide_bf16'),('matched_union_bf16','matched_single_bf16')]:
        if set(by_arm[a])!=set(by_arm[b]): raise ValueError('Comparison coverage differs')
        ds = [by_arm[a][q]-by_arm[b][q] for q in by_arm[a]]
        comparisons.append(dict(arm=a,comparator=b,delta_points=100*statistics.mean(ds),
            wins=sum(d>1e-9 for d in ds),ties=sum(abs(d)<=1e-9 for d in ds),losses=sum(d<-1e-9 for d in ds)))
    if informative:
        ds=[by_arm['matched_union_bf16'][q]-by_arm['matched_single_bf16'][q] for q in sorted(informative)]
        comparisons.append(dict(arm='matched_union_bf16',comparator='matched_single_bf16',
            scope='candidate-subset hashes differ; fixed without outcomes',n=len(ds),
            delta_points=100*statistics.mean(ds),wins=sum(d>1e-9 for d in ds),
            ties=sum(abs(d)<=1e-9 for d in ds),losses=sum(d<-1e-9 for d in ds)))
    report = dict(signature=sid,complete=True,
        scores=[dict(arm=a,n=len(by_arm[a]),native_f1_points=100*statistics.mean(by_arm[a].values())) for a in by_arm],
        comparisons=comparisons,matching=matching,heldout_accessed=False)
    common.immutable_json(RUN/'scored.json',rows); common.immutable_json(RUN/'report.json',report)
    print('CONTROL_RESULTS',report['scores'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['freeze','prepare','qa','score'])
    parser.add_argument('--endpoint',default='http://127.0.0.1:8018'); args=parser.parse_args()
    if args.mode=='qa': qa(args.endpoint)
    else: globals()[args.mode]()
