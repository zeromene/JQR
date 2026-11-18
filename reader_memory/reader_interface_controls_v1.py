"""Frozen prompt controls over unchanged evidence; no selection or training."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random
import statistics

from . import coverage_reader_controls_v1 as parent
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'research/reader_interface_controls_v1'
RUN = ROOT / 'runs/reader_interface_controls_v1'
METHODS = parent.ARMS
VARIANTS = ('original', 'format_only', 'relation_aware')
CONFIG = dict(version='reader-interface-controls-v1', sources=16, questions=32,
    methods=list(METHODS), variants=list(VARIANTS), evidence_budget=2048,
    reader='Qwen3.5-9B', output_tokens=2048, decoding='greedy-thinking_false',
    max_conditions=480, max_new_reader_calls=320, retries=0,
    bootstrap_samples=2000, bootstrap_seed=29,
    unchanged='question, evidence bytes/order/membership, references, model, decoder, parser, F1',
    scope='previously exposed original16 dev sources; post-hoc motivated feasibility controls',
    primary_contrasts=['format_only-original', 'relation_aware-format_only'],
    support_diagnostic_only=True, no_training=True, no_search=True,
    protected_final_data_accessed=False, independent_confirmation=False,
    automatic_followup=False)

FORMAT_REPLACEMENTS = (
    ('2. Then provide a very CONCISE answer (short phrase about core information)',
     '2. Return only the minimal entity or value requested in the answer field. '
     'Do not add background, explanations, or information beyond what the question asks.'),
    ("4. All dates in the response must be formatted as 'DD Month YYYY' but you can output more or less details if needed",
     '4. Preserve the date precision supported by the context and requested by the question. '
     'If only a year is given, return the year; never invent missing month or day components.'))
RELATION_REPLACEMENT = (
    '1. First, think through the reasoning process',
    '1. Identify the type of entity or value the question requests. Check the relevant '
    'facts across all supplied contexts and connect references to the same entities. '
    'Distinguish the final requested value from intermediate entities and nearby facts '
    'about other subjects. Before declaring information unavailable, check whether '
    'the supplied facts jointly support the answer. Do not invent a missing relation. '
    'Use the reasoning field only for a brief evidence-based explanation.')

load = parent.load
immutable = parent.immutable
base = parent.base
ledger = parent.ledger
ORIGINAL_MESSAGES_FOR = ledger.messages_for


def arm(method, variant):
    if method not in METHODS or variant not in VARIANTS:
        raise ValueError('Unknown method or reader variant')
    return method + '__' + variant


def messages_for(row):
    # Only these three fields affect the prompt, including for support diagnostics.
    messages = ORIGINAL_MESSAGES_FOR(dict(question=row['question'], evidence_text=row['evidence_text']))
    variant = row['reader_variant']
    if variant not in VARIANTS:
        raise ValueError('Unknown reader variant')
    if variant == 'original':
        return messages
    text = messages[1]['content']
    replacements = list(FORMAT_REPLACEMENTS)
    if variant == 'relation_aware':
        replacements.append(RELATION_REPLACEMENT)
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError('Original prompt anchor changed or is ambiguous')
        text = text.replace(old, new, 1)
    return [messages[0], dict(role='user', content=text)]


def expand(plans):
    result = []
    for row in plans:
        method = row['arm']
        for variant in VARIANTS:
            result.append(dict(**{k:v for k,v in row.items() if k != 'arm'},
                arm=arm(method, variant), method=method, reader_variant=variant))
    return result


def fixed_plans():
    public = load(PLAN / 'public/method_plans.json')
    diagnostic = load(PLAN / 'private/support_plans.json')
    rows = public + diagnostic
    sources = load(parent.PLAN / 'public/source_inputs.json')
    positions = {r['question_id']:i for i,r in enumerate(sources)}
    expected = {(q, a) for q in positions for a in METHODS}
    if len(rows) != 160 or {(r['question_id'],r['arm']) for r in rows} != expected:
        raise ValueError('All original32 questions and five evidence methods are required')
    if any(r['arm']=='support_only' or r['diagnostic_only'] or r['private_inputs_read'] for r in public):
        raise ValueError('Support annotations leaked into deployable plans')
    if any(r['arm']!='support_only' or not r['diagnostic_only'] for r in diagnostic):
        raise ValueError('Diagnostic support plan lost its status')
    return sorted(rows, key=lambda r:(positions[r['question_id']],METHODS.index(r['arm'])))


def freeze():
    inherited = load(parent.PLAN / 'manifest.json')
    names = list(inherited['files']) + [
        'research/coverage_reader_controls_v1/manifest.json',
        'reader_memory/reader_interface_controls_v1.py',
        'tools/prepare_reader_interface_controls_v1.py',
        'scripts/reader_interface_controls_v1_queue.py',
        'tests/test_reader_interface_controls_v1.py']
    names += ['research/reader_interface_controls_v1/' + n for n in (
        'PROTOCOL.md', 'public/method_plans.json', 'private/support_plans.json',
        'private/references.json', 'private/replay_cache.json', 'private/original_outcomes.json',
        'preparation.json')]
    value = dict(config=CONFIG, parent_signature=inherited['signature'],
        files={n:base.sha(ROOT/n) for n in sorted(set(names))})
    value['signature'] = fingerprint(value)
    immutable(PLAN / 'manifest.json', value)
    print('FROZEN_READER_INTERFACE_CONTROLS', value['signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or value['signature'] != fingerprint({k:v for k,v in value.items() if k!='signature'}):
        raise ValueError('Reader interface scientific identity changed')
    if parent.verify() != value['parent_signature']:
        raise ValueError('Original evidence experiment changed')
    for n, digest in value['files'].items():
        if base.sha(ROOT/n) != digest:
            raise ValueError('Frozen input changed: '+n)
    return value['signature']


def replay_cache():
    marker = load(parent.PLAN/'verification_markers.json')['Qwen3.5-9B']
    cache = {}
    for item in load(PLAN/'private/replay_cache.json'):
        parent.check_cache_entry(item, marker)
        if item['key'] in cache:
            raise ValueError('Duplicate frozen historical request')
        cache[item['key']] = item['entry']
    return cache


def preflight():
    sid = verify()
    from transformers import AutoTokenizer
    from tools.prepare_replacement_learning_v1 import ExactCounter
    counter = ExactCounter()
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3.5-9B'),
        local_files_only=True, trust_remote_code=False)
    preparation = load(PLAN/'preparation.json')
    for n, digest in preparation['server_source_files'].items():
        if base.sha(ROOT/n) != digest:
            raise ValueError('Completed source artifact changed: '+n)
    marker = load(parent.PLAN/'verification_markers.json')['Qwen3.5-9B']
    if base.sha(ROOT/'models/Qwen3.5-9B/.verified.json') != marker:
        raise ValueError('Reader marker changed')
    cache = replay_cache()
    rows = fixed_plans()
    source = {(r['question_id'],r['arm']):r for r in load(parent.RUN/'qa_plans.json')}
    if any(r != source[r['question_id'],r['arm']] for r in rows):
        raise ValueError('Original question, evidence, or metadata changed')
    plans = expand(rows)
    prompts = []
    unique, old_keys = set(), set()
    for p in plans:
        if counter.count(p['evidence_text']) != p['evidence_tokens'] or p['evidence_tokens']>2048 or fingerprint(p['evidence_text'])!=p['evidence_sha256']:
            raise ValueError('Evidence changed or exceeds unchanged budget')
        messages = messages_for(p)
        key = parent.request_key(messages,marker)
        unique.add(key)
        if p['reader_variant']=='original':
            old_keys.add(key)
            if key not in cache:
                raise ValueError('Original requests must reuse all historical outcomes, including failures')
        elif key in cache:
            raise ValueError('New prompt unexpectedly matches original cache')
        prompt = tokenizer.apply_chat_template(messages,tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        n = counter.count(prompt)
        if n+2048>32768:
            raise ValueError('Full prompt too long; no truncation')
        prompts.append(dict(question_id=p['question_id'],arm=p['arm'],request_key=key,
            full_prompt_tokens=n,messages_sha256=fingerprint(messages)))
    if len(plans)!=480 or len(unique-set(cache))>320 or set(cache)!=old_keys:
        raise ValueError('Unexpected condition or call budget')
    immutable(RUN/'qa_plans.json',plans)
    immutable(RUN/'selection_complete.json',dict(signature=sid,plans_sha256=base.sha(RUN/'qa_plans.json')))
    immutable(RUN/'preflight.json',dict(signature=sid,passed=True,conditions=480,
        unique_requests=len(unique),uncached_requests=len(unique-set(cache)),
        original_requests=len(old_keys),original_conditions=160,
        full_prompt_max_tokens=max(p['full_prompt_tokens'] for p in prompts),
        prompt_identities=prompts,new_training=False,new_selection=False,
        protected_final_data_accessed=False))
    print('READER_INTERFACE_PREFLIGHT_OK',len(plans),'conditions',len(unique-set(cache)),'new calls',flush=True)


def qa(endpoint):
    sid = verify()
    # Reuse the unchanged transaction ledger and official parser in an isolated process.
    previous = ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for
    ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = RUN, lambda:sid, replay_cache, messages_for
    try:
        ledger.qa(endpoint)
    finally:
        ledger.RUN, ledger.verify, ledger.saved_cache, ledger.messages_for = previous


def paired(a,b):
    if set(a)!=set(b):
        raise ValueError('Unpaired comparison')
    deltas, normal, groups = [], [], defaultdict(list)
    for q in sorted(a):
        x,y = a[q],b[q]
        delta = x['local_native_f1']-y['local_native_f1']
        deltas.append(delta)
        groups[x['native_id']].append(delta)
        if x['status']==y['status']=='ok':
            normal.append(delta)
    rng = random.Random(29)
    keys = sorted(groups)
    boots = sorted(100*statistics.mean(v for _ in keys for v in groups[rng.choice(keys)]) for _ in range(2000))
    return dict(n=len(deltas),sources=len(groups),delta_points=100*statistics.mean(deltas),
        wins=sum(d>1e-9 for d in deltas),ties=sum(abs(d)<=1e-9 for d in deltas),
        losses=sum(d < -1e-9 for d in deltas),normal_pairs=len(normal),
        normal_pair_delta_points=100*statistics.mean(normal) if normal else None,
        source_bootstrap_95_points=[boots[49],boots[1949]])


def score():
    sid = verify()
    plans = load(RUN/'qa_plans.json')
    refs = load(PLAN/'private/references.json')
    scored = parent.score_rows(plans,list(read_jsonl(RUN/'qa/predictions.jsonl')),refs,sid)
    originals = {(r['question_id'],r['arm']):r for r in load(PLAN/'private/original_outcomes.json')}
    for r in scored:
        if r['reader_variant']=='original':
            old = originals[r['question_id'],r['method']]
            if any(old[k]!=r[k] for k in ('hypothesis','status','local_native_f1')) or not r['reused']:
                raise ValueError('Original outcome or F1 was altered')
    by = {(m,v):{r['question_id']:r for r in scored if r['method']==m and r['reader_variant']==v}
        for m in METHODS for v in VARIANTS}
    summaries, contrasts = [], []
    for (method,variant), mapping in by.items():
        for scope in ('overall','multi','single'):
            values = [r for r in mapping.values() if scope=='overall' or r['mode']==scope]
            summaries.append(dict(method=method,variant=variant,scope=scope,n=len(values),
                diagnostic_only=method=='support_only',f1_points=100*statistics.mean(r['local_native_f1'] for r in values),
                full_f1_count=sum(r['local_native_f1']==1 for r in values),
                statuses=dict(Counter(r['status'] for r in values)),
                mean_answer_words=statistics.mean(len(r['hypothesis'].split()) for r in values)))
    pairs = []
    for m in METHODS:
        for a,b in (('format_only','original'),('relation_aware','format_only'),('relation_aware','original')):
            pairs.append(((m,a),(m,b),'reader_prompt'))
    for v in VARIANTS:
        for a,b in (('raw20_batch4','raw14_batch4'),('raw20_batch4','semantic_mixed_batch4'),
                    ('semantic_mixed_batch4','static_fusion'),('support_only','raw20_batch4')):
            pairs.append(((a,v),(b,v),'evidence_method'))
    for a,b,kind in pairs:
        for scope in ('overall','multi','single'):
            choose = lambda key:{q:r for q,r in by[key].items() if scope=='overall' or r['mode']==scope}
            contrasts.append(dict(kind=kind,arm=list(a),comparator=list(b),scope=scope,
                diagnostic_only='support_only' in (a[0],b[0]),**paired(choose(a),choose(b))))
    calls = list(read_jsonl(RUN/'qa/answer_calls.jsonl'))
    if len(calls)>320:
        raise ValueError('Exceeded frozen call budget')
    costs = []
    for variant in VARIANTS:
        group = [c for c in calls if c['source_condition'][1].endswith('__'+variant)]
        costs.append(dict(variant=variant,new_calls=len(group),statuses=dict(Counter(c['status'] for c in group)),
            seconds=sum(c['seconds'] for c in group),
            **{k:sum(c['result'].get('usage',{}).get(k,0) for c in group) for k in ('prompt_tokens','completion_tokens')}))
    report = dict(signature=sid,complete=True,scores=summaries,comparisons=contrasts,
        conditions=len(scored),new_reader_calls=len(calls),reused_conditions=sum(r['reused'] for r in scored),
        reader_cost=costs,original_outcomes_unchanged=True,references_unchanged=True,
        scope=CONFIG['scope'],independent_confirmation=False,training=False,new_selection=False,
        protected_final_data_accessed=False,paper_ready=False,automatic_followup=False,
        next_action='Review common reader effects before further selection research; no automatic launch')
    write_json(RUN/'scored.json',scored)
    write_json(RUN/'report.json',report)
    print('READER_INTERFACE_SCORED',len(scored),'conditions',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['freeze','verify','preflight','qa','score'])
    parser.add_argument('--endpoint',default='http://127.0.0.1:8020')
    args = parser.parse_args()
    if args.mode=='qa':
        qa(args.endpoint)
    else:
        result = globals()[args.mode]()
        if result:
            print(result,flush=True)
