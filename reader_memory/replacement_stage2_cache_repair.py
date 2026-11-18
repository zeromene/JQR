"""Materialize already-known empty-result cache failures without making or retrying RPCs."""
from .schema import fingerprint
from . import replacement_precision_v2 as frozen


def known_failure_rows(plans, committed, cache, signature, key_for=None):
    key_for=key_for or (lambda p:frozen.request_key(frozen.messages_for(p)))
    expected={(p['question_id'],p['arm']):p for p in plans}
    if len(expected)!=len(plans):raise ValueError('Duplicate planned condition')
    done=set()
    for row in committed:
        condition=row['question_id'],row['arm']
        if condition in done or condition not in expected or row['signature']!=signature:
            raise ValueError('Invalid committed condition')
        if row['plan_sha256']!=fingerprint(expected[condition]) or row['request_key']!=key_for(expected[condition]):
            raise ValueError('Committed plan or request changed')
        done.add(condition)
    additions=[]
    for plan in plans:
        condition=plan['question_id'],plan['arm']
        if condition in done:continue
        key=key_for(plan);entry=cache.get(key)
        if entry is None or entry.get('result'):continue
        if entry['status']!='failed' or entry['hypothesis'] or not entry.get('error'):
            raise ValueError('Empty cache result is not a documented failure')
        additions.append(dict(**plan,signature=signature,plan_sha256=fingerprint(plan),request_key=key,
            hypothesis=entry['hypothesis'],status=entry['status'],error=entry['error'],reused=True,
            cache_source=entry['source'],cache_condition=entry['source_condition']))
    return additions
