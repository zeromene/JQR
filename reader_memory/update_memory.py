"""Transparent lexical update links and query-blind extractive controls."""
from __future__ import annotations

from collections import Counter
import math
import re

from .independent_pack import normalize
from .retrieval import words
from .selection import Choice, render_evidence

CHANGE = re.compile(r"\b(?:no longer|not|never|don't|doesn't|didn't|cannot|can't|instead|stopp?ed|quit|abandon(?:ed)?|withdraw|withdrew|cancel(?:led|ed)?|"
                    r"discontinu(?:ed|e)|drop(?:ped)?|skip(?:ped)?|switch(?:ed)?|shift(?:ed)?|"
                    r"decided|realized|rejoined|resumed|restarted|returned|reopen(?:ed)?|"
                    r"step(?:ped)? back|gave up|give up|but|however|unfortunately)\b", re.I)
STOP = set("a an the and or of to for in on at by as with from into about up out over under "
           "i me my we us our you your they their it its this that these those is am are was were be been being "
           "have has had do does did not no so but if then than very more most much many also just really "
           "recently lately after before now still actually decided realized found felt feel user assistant "
           "something things thing experience experiences time way new started began".split())


def lead(text):
    return re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]


def group_order(group):
    return min(int(re.search(r":m(\d+)", span['source_id'])[1]) for span in group.metadata['spans'])


def content_terms(text):
    values = set()
    for term in words(re.sub(r'^\s*User:\s*', '', text, flags=re.I)):
        if term in STOP or len(term) < 3 or term.isdigit():
            continue
        if term.endswith('ies') and len(term) > 5:
            term = term[:-3]+'y'
        elif term.endswith('s') and not term.endswith('ss') and len(term) > 4:
            term = term[:-1]
        values.add(term)
    return values


def first_text(group):
    return ' '.join(lead(s['text']) for s in group.metadata['spans'])


def render_spans(group, mode):
    parts, protected = [], []
    for span in group.metadata['spans']:
        first = lead(span['text'])
        keep = mode == 'raw' or (mode == 'guard' and bool(CHANGE.search(first)))
        value = span['text'] if keep else first
        if keep and mode == 'guard':
            protected.append(span['source_id'])
        parts.append(f"[{span['source_id']}@{span['start']}:{span['start']+len(value)}] [{span['date']}] {span['role']}: {value}")
    return '\n'.join(parts), protected


def variants(groups, counter):
    """All policies inspect source text only; they receive no current query."""
    for group in groups:
        if render_spans(group, 'raw')[0] != group.variants['raw']:
            raise ValueError('Raw source serialization changed')
        group.variants['lead'], _ = render_spans(group, 'lead')
        guard, protected = render_spans(group, 'guard')
        proposed = Choice(group.group_id,'guard',guard,tuple(group.source_ids),group.date,0,0)
        raw = make_choice(group,'raw')
        fallback = counter.count(render_evidence([proposed])) >= counter.count(render_evidence([raw]))
        group.variants['guard'] = group.variants['raw'] if fallback else guard
        group.metadata['guard_control'] = {'protected_source_ids':protected, 'raw_fallback':fallback,
                                           'semantically_complete_verified':False}
    return groups


def make_choice(group, variant):
    return Choice(group.group_id, variant, group.variants[variant], tuple(group.source_ids), group.date,0,0)


def link_target(anchor, groups):
    """Prefer the latest lexically related update after an anchor, else before.

    This is a heuristic candidate relationship, not semantic contradiction proof.
    """
    tokens = {g.group_id:content_terms(first_text(g)) for g in groups}
    df = Counter(t for ts in tokens.values() for t in ts)
    weight = {t:math.log(1+(len(groups)+1)/(f+1)) for t,f in df.items()}
    a = tokens[anchor.group_id]
    anchor_sources = {s['source_id'] for s in anchor.metadata['spans']}
    candidates = []
    for group in groups:
        if group.group_id == anchor.group_id or any(s['source_id'] in anchor_sources for s in group.metadata['spans']):
            continue
        if not CHANGE.search(first_text(group)):
            continue
        shared = a & tokens[group.group_id]
        overlap = sum(weight[t] for t in shared)/max(1e-12,sum(weight[t] for t in a))
        if shared and overlap >= .18:
            candidates.append((group, {'shared_terms':sorted(shared), 'weighted_anchor_overlap':overlap,
                                       'after_anchor':group_order(group)>group_order(anchor)}))
    after = [c for c in candidates if c[1]['after_anchor']]
    eligible = after or candidates
    if not eligible:
        return [anchor], {'linked':False, 'anchor_id':anchor.group_id, 'candidate_links':0,
                          'policy':'lexical update relation; no qualifying partner'}
    partner, trace = max(eligible,key=lambda item:(group_order(item[0]),item[1]['weighted_anchor_overlap'],item[0].group_id))
    return sorted([anchor,partner],key=group_order), {'linked':True, 'anchor_id':anchor.group_id,
            'partner_id':partner.group_id, 'candidate_links':len(candidates), **trace,
            'policy':'latest qualifying later update, else latest qualifying earlier update',
            'semantic_update_relation_verified':False}


def intervention_plan(ranked, target, anchor, counter, budget):
    """One identical background for every target representation, exact final costs."""
    target_sources = {s['source_id'] for g in target for s in g.metadata['spans']}
    texts = {normalize(s['text']) for g in target for s in g.metadata['spans']}
    additions = {'without':[], 'anchor':[make_choice(anchor,'raw')],
                 'bundle_raw':[make_choice(g,'raw') for g in target],
                 'bundle_lead':[make_choice(g,'lead') for g in target],
                 'bundle_guard':[make_choice(g,'guard') for g in target]}
    if any(counter.count(render_evidence(v))>budget for v in additions.values()):
        raise ValueError('Target cannot fit every intervention branch')
    background = []
    for group,_ in ranked:
        spans = group.metadata['spans']
        if any(s['source_id'] in target_sources or normalize(s['text']) in texts for s in spans):
            continue
        proposed = [*background,make_choice(group,'raw')]
        if all(counter.count(render_evidence([*proposed,*extra]))<=budget for extra in additions.values()):
            background = proposed
    result = {}
    for name,extra in additions.items():
        value = render_evidence([*background,*extra])
        result[name] = {'text':value, 'tokens':counter.count(value),
                        'target_chunk_ids':[c.group_id for c in extra]}
        if result[name]['tokens']>budget:
            raise ValueError('Final intervention exceeds evidence budget')
    return {'background':render_evidence(background), 'background_chunk_ids':[c.group_id for c in background],
            'excluded_source_ids':sorted(target_sources), 'branches':result}
