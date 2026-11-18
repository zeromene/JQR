"""Bounded full-public-source diagnostic on the exposed 24-source panel."""
from collections import Counter
import json
import statistics
import time
from . import anchored_quote_9b_v1 as parent
from .discriminative_pilot_v1 import messages, validate_public, paired
from .relation_preserving_pilot_v1 import render_ranges

ROOT = parent.ROOT
BASE = ROOT / 'research/full_source_probe_v1'
RUN = ROOT / 'runs/full_source_probe_v1'
load, sha, fp, immutable, write_json = parent.load, parent.sha, parent.fp, parent.immutable, parent.write_json
READERS = ('Qwen3.5-4B', 'Qwen3.5-9B')
CONFIG = dict(version='full-source-probe-v1', sources=24, max_calls=48, max_input=24576, max_output=128,
              readers=list(READERS), diagnostic_only=True, public_source_order=True,
              resource_caps_mib={'qa4': 13000, 'qa9': 20500}, resource_headroom_mib={'qa4': 1000, 'qa9': 500})


def verify():
    m = load(BASE / 'manifest.json')
    assert m['config'] == CONFIG and fp({k: v for k, v in m.items() if k != 'signature'}) == m['signature']
    for n, d in m['files'].items():
        if sha(ROOT / n) != d:
            raise ValueError('Frozen file changed: ' + n)
    return m['signature']


def sources():
    return load(BASE / 'public/sources.json')


def public_messages(s):
    validate_public(s)
    ranges = {p['source_id']: [[0, len(p['text'])]] for p in s['paragraphs']}
    return messages(s['question'], render_ranges(s, ranges))


def preflight():
    from transformers import AutoTokenizer
    signature = verify()
    for n, d in load(BASE / 'manifest.json')['model_files'].items():
        assert sha(ROOT / n) == d, n
    assert len(sources()) == 24
    lengths = {}
    for reader in READERS:
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'models' / reader), local_files_only=True, trust_remote_code=False)
        lengths[reader] = [len(tokenizer.encode(tokenizer.apply_chat_template(public_messages(s), tokenize=False,
                          add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)) for s in sources()]
        assert max(lengths[reader]) <= CONFIG['max_input']
    immutable(RUN / 'preflight.json', dict(signature=signature, lengths=lengths, passed=True))
    print('PREFLIGHT_OK', flush=True)


def qa(reader):
    from .backend import TransformersBackend
    from .mainline_service_v1 import generate
    from .full_comparison_v1 import torch_setup
    signature = verify()
    torch_setup()
    backend = TransformersBackend(ROOT / 'models' / reader, max_input_tokens=CONFIG['max_input'])
    ordered = sorted(sources(), key=lambda s: fp([30, s['question_id']]))
    for i, source in enumerate(ordered, 1):
        request = public_messages(source)
        key = fp([signature, reader, request, CONFIG['max_output']])
        path = RUN / 'responses' / reader / (key + '.json')
        begun = path.with_suffix('.started.json')
        if path.exists() or begun.exists():
            raise ValueError('Existing request; no unknown replay')
        base = dict(signature=signature, reader=reader, question_id=source['question_id'], source_hash=fp(source),
                    request_hash=fp(request), max_output=CONFIG['max_output'])
        immutable(begun, dict(base, time=time.time()))
        try:
            result = generate(backend, request, CONFIG['max_output'])
        except Exception as exc:
            immutable(path, dict(base, status='runtime_failure', error=repr(exc)))
            raise
        immutable(path, dict(base, status='ok' if result['text'].strip() else 'empty', **result))
        print('QA', reader, i, '/ 24', flush=True)


def score():
    from .full_comparison_v1 import answer_scores
    signature = verify()
    rows = [load(p) for p in (RUN / 'responses').rglob('*.json') if not p.name.endswith('.started.json')]
    assert len(rows) == 48 and all(r['status'] in ('ok', 'empty') for r in rows)
    assert len({(r['reader'], r['question_id']) for r in rows}) == 48
    refs = load(BASE / 'private/references.json')
    baseline = load(BASE / 'private/ap3_records.json')
    index = {(r['reader'], r['question_id'], r['budget']): r for r in baseline}
    records = []
    ss = {s['question_id']: s for s in sources()}
    for row in rows:
        assert row['signature'] == signature and row['source_hash'] == fp(ss[row['question_id']])
        assert row['request_hash'] == fp(public_messages(ss[row['question_id']]))
        em, f1 = answer_scores(row['text'], refs[row['question_id']])
        ds = {str(b): f1 - index[row['reader'], row['question_id'], b]['f1'] for b in (256, 512)}
        records.append(dict(row, em=em, f1=f1, ap3_delta_by_budget=ds, pooled_delta=statistics.mean(ds.values())))
    grouped = {}
    for reader in READERS:
        rs = [r for r in records if r['reader'] == reader]
        grouped[reader] = dict(f1_points=100 * statistics.mean(r['f1'] for r in rs),
            mean_input_tokens=statistics.mean(r['usage']['prompt_tokens'] for r in rs),
            versus_ap3=paired([next(r['pooled_delta'] for r in rs if r['question_id'] == q) for q in refs]))
    signal = all(g['versus_ap3']['mean_delta_f1'] > 0 and g['versus_ap3']['wins'] >= 3 for g in grouped.values())
    summary = dict(signature=signature, complete=True, records=records, groups=grouped,
        development_routing_signal=signal, diagnostic_only=True, not_budget_matched=True,
        finish_reasons=dict(Counter(r['finish_reason'] for r in rows)),
        costs={k: sum(r['usage'].get(k, 0) for r in rows) for k in ('prompt_tokens', 'completion_tokens', 'elapsed_seconds')})
    immutable(RUN / 'summary.json', summary)
    print('SCORE_COMPLETE', json.dumps(dict(groups=grouped, development_routing_signal=signal)), flush=True)


def phase(name):
    if name == 'preflight':
        preflight()
    elif name in ('qa4', 'qa9'):
        qa('Qwen3.5-' + ('4B' if name == 'qa4' else '9B'))
    elif name == 'score':
        score()
    else:
        raise ValueError(name)
