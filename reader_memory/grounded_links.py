"""Source-grounded link contract, separate from the frozen lexical baseline.

Exact citations are necessary, not sufficient, for semantic identity. A reviewed
relation is an auditable AI decision, never a new human gold label. No question,
answer, options, score, or task metadata is accepted by this interface.
"""
from __future__ import annotations

import re

from .schema import fingerprint

RELATIONS = {'same_object_update', 'same_object_continuation', 'unrelated', 'uncertain'}
TRANSITION = re.compile(r'\b(?:no longer|stopped|quit|gave up|give up|step back|stepped back|'
                        r'withdrew|dropped out|cancelled|canceled|resumed|restarted|instead|'
                        r'switched|changed|returned|another shot)\b', re.I)


def order(group):
    orders = []
    for span in group.metadata['spans']:
        match = re.fullmatch(r'p(\d+):m(\d+):t\d+', span['source_id'])
        if match is None or span['role'] != 'user':
            raise ValueError('Expected native user provenance with explicit order')
        orders.append((int(match[1]), int(match[2])))
    if len({p for p, _ in orders}) != 1:
        raise ValueError('Mixed user identities')
    return min(orders)


def citation(group, text, *, unique=True):
    """Locate an exact, unique source quotation; offsets address original text."""
    if not isinstance(text, str) or not text:
        raise ValueError('Empty quotation')
    matches = []
    for span in group.metadata['spans']:
        cursor = 0
        while (index := span['text'].find(text, cursor)) >= 0:
            matches.append({'source_id': span['source_id'], 'start': span['start'] + index,
                            'end': span['start'] + index + len(text), 'text': text})
            cursor = index + 1
    if not matches or (unique and len(matches) != 1):
        raise ValueError('Quotation must have one exact source occurrence')
    return matches[0]


def check_citation(group, item):
    if item != citation(group, item['text']):
        raise ValueError('Quotation/source offsets do not match')


def review(earlier, later, relation, rationale, *, object_text=None,
           earlier_state=None, later_state=None, reason=None):
    if relation not in RELATIONS or not isinstance(rationale, str) or not rationale.strip():
        raise ValueError('Unknown relation or missing decision rationale')
    a, b = order(earlier), order(later)
    if a[0] != b[0] or a[1] >= b[1]:
        raise ValueError('A link requires the same user and two ordered, distinct turns')
    item = {'earlier_group_id': earlier.group_id, 'later_group_id': later.group_id,
        'earlier_sha256': fingerprint(earlier.to_dict()), 'later_sha256': fingerprint(later.to_dict()),
        'relation': relation, 'rationale': rationale, 'object_text': object_text,
        'identity_citations': [], 'state_citations': [], 'reason_citation': None,
        'review_origin': 'AI-source-text-review', 'human_gold': False,
        'semantic_correctness_guaranteed': False, 'accepted_update': False}
    if relation in {'same_object_update', 'same_object_continuation'}:
        # A single generic noun cannot establish identity. Exact multiword
        # identity is a conservative requirement; aliases remain abstentions.
        terms = re.findall(r'[A-Za-z]+', object_text or '')
        if len(terms) < 2:
            raise ValueError('Require an explicit multiword object/activity mention')
        item['identity_citations'] = [citation(g, object_text, unique=False) for g in (earlier, later)]
    if relation == 'same_object_update':
        if not earlier_state or not later_state or object_text not in earlier_state or object_text not in later_state:
            raise ValueError('State quotations must explicitly name the same object')
        item['state_citations'] = [citation(earlier, earlier_state), citation(later, later_state)]
        if not TRANSITION.search(later_state):
            raise ValueError('Missing explicit later transition; abstain')
        if reason is not None:
            item['reason_citation'] = citation(later, reason)
        item['accepted_update'] = True
    return item


def validate(earlier, later, item):
    """Reject stale reviews, forged quotes, object/state drift and reordered turns."""
    if (item['earlier_sha256'], item['later_sha256']) != (fingerprint(earlier.to_dict()), fingerprint(later.to_dict())):
        raise ValueError('Reviewed source changed')
    states = item['state_citations']
    expected = review(earlier, later, item['relation'], item['rationale'], object_text=item['object_text'],
        earlier_state=states[0]['text'] if states else None,
        later_state=states[1]['text'] if states else None,
        reason=item['reason_citation']['text'] if item['reason_citation'] else None)
    if expected != item:
        raise ValueError('Review contract or provenance changed')
    return item['accepted_update']
