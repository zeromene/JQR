"""Shared immutable-run and exact-request QA reuse for the replacement follow-up."""
from pathlib import Path

from . import replacement_precision_v2 as frozen
from . import replacement_learning_v1 as parent
from . import baseline_bootstrap_v2 as base
from .schema import fingerprint, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]


def load(path): return frozen.load(path)


def immutable_json(path, value):
    path = Path(path)
    if path.exists() and load(path) != value: raise ValueError('Immutable artifact changed: ' + str(path))
    if not path.exists(): write_json(path, value)


def identity(plan, config, paths):
    value = dict(config=config, parent_signature=parent.verify(), precision_signature=frozen.verify(),
                 files={p:base.sha(ROOT / p) for p in sorted(set(paths))})
    value['signature'] = fingerprint(value)
    immutable_json(plan / 'manifest.json', value)
    return value['signature']


def verify(plan, config):
    saved = load(plan / 'manifest.json')
    if saved['config'] != config or saved['parent_signature'] != parent.verify() or saved['precision_signature'] != frozen.verify():
        raise ValueError('Follow-up configuration or parent changed')
    if saved['signature'] != fingerprint({k:v for k,v in saved.items() if k!='signature'}):
        raise ValueError('Invalid manifest signature')
    for path, digest in saved['files'].items():
        if base.sha(ROOT / path) != digest: raise ValueError('Frozen dependency changed: ' + path)
    return saved['signature']


def cached_answers(extra_call_files=()):
    cache = frozen.saved_cache()
    predictions = list(read_jsonl(frozen.PARENT_RUN / 'rewards/predictions.jsonl'))
    calls = {(r['question_id'],r['variant']):r for r in read_jsonl(frozen.PARENT_RUN / 'rewards/answer_calls.jsonl')}
    for row in predictions:
        call = calls[row['question_id'],row['variant']]
        messages = frozen.messages_for(row)
        if messages != call['messages'] or call['max_tokens'] != 2048: raise ValueError('Old reward cache request mismatch')
        key = frozen.request_key(messages)
        entry = dict(hypothesis=row['hypothesis'],status=row['status'],error=row.get('error'),result=call['result'],
                     source='runs/replacement_learning_v1_repair1/rewards/predictions.jsonl',
                     source_condition=[row['question_id'],row['variant']])
        if entry['status']=='ok' and frozen.parse_answer(base.require_complete(entry['result'])) != entry['hypothesis']:
            raise ValueError('Old reward cache parser mismatch')
        _insert(cache,key,entry)
    for path in [ROOT / 'runs/replacement_precision_v2/qa/answer_calls.jsonl', *map(Path,extra_call_files)]:
        if not path.exists(): continue
        for call in read_jsonl(path):
            key = frozen.request_key(call['messages'],call['max_tokens'])
            if key != call['key']: raise ValueError('Cached request key mismatch')
            entry = {k:call[k] for k in ['hypothesis','status','error','result']}
            entry.update(source=str(path.relative_to(ROOT)),source_condition=call['source_condition'])
            if entry['status']=='ok' and frozen.parse_answer(base.require_complete(entry['result'])) != entry['hypothesis']:
                raise ValueError('Cached output/parser mismatch')
            _insert(cache,key,entry)
    return cache


def _insert(cache,key,entry):
    if key in cache and (cache[key]['hypothesis'],cache[key]['status']) != (entry['hypothesis'],entry['status']):
        raise ValueError('Identical requests have inconsistent saved outcomes')
    cache.setdefault(key,entry)


def run_qa(run, plans, signature, endpoint, extra_call_files=()):
    """Separate process uses the frozen, tested QA ledger with new paths/identity only."""
    immutable_json(run / 'qa_plans.json', plans)
    immutable_json(run / 'selection_complete.json', dict(signature=signature,plans_sha256=base.sha(run / 'qa_plans.json')))
    cache = cached_answers(extra_call_files)
    old_run, old_verify, old_cache = frozen.RUN, frozen.verify, frozen.saved_cache
    frozen.RUN = run; frozen.verify = lambda:signature; frozen.saved_cache = lambda:cache
    try: frozen.qa(endpoint)
    finally: frozen.RUN, frozen.verify, frozen.saved_cache = old_run, old_verify, old_cache


def semantic_ids(tokenizer, question, text):
    from . import mainline_reranker_v1 as semantic
    body = f"<Instruct>: {semantic.CONFIG['instruction']}\n<Query>: {question}\n<Document>: {text}"
    ids = tokenizer.encode(semantic.PREFIX,add_special_tokens=False)+tokenizer.encode(body,add_special_tokens=False)+tokenizer.encode(semantic.SUFFIX,add_special_tokens=False)
    if len(ids)>8192: raise ValueError('Semantic input exceeds8192; no truncation')
    return ids


def teacher_score(torch,tokenizer,model,question,text):
    from . import replacement_ranker_v1 as ranker
    return float(ranker.logit(torch,tokenizer,model,ranker.input_ids(tokenizer,question,text)))


def official_rows(plans, predictions):
    base.verify_inputs(private=True)
    refs = {r['question_id']:r for r in load(base.PLAN / 'private/references.json')}
    expected = {(r['question_id'],r['arm']) for r in plans}
    if len(predictions)!=len(expected) or {(r['question_id'],r['arm']) for r in predictions}!=expected:
        raise ValueError('Complete QA coverage required')
    out = []
    for row in predictions:
        ref = refs[row['question_id']]
        value = base.native_f1(row['hypothesis'],ref['answer'],ref['category']) if row['status']=='ok' else 0.
        out.append(dict(**row,official_native_f1=value))
    return out
