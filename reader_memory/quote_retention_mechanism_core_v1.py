"""Answer-blind quotation controls with the production source-offset renderer."""
from copy import deepcopy
import math

ARMS = ('A_jqr', 'B_protect_fill', 'C_reselect_tight', 'D_reselect_full')


def span_for(source, uid):
    pid, _, index = uid.split(':')
    paragraph = next(p for p in source['paragraphs'] if p['source_id'] == pid)
    span = paragraph['spans'][int(index)]
    return dict(source=pid, start=span['start'], end=span['end'])


def render(source, scores, ranges, count):
    from .path_budget_controls_v1 import sentence_order
    from .relation_preserving_pilot_v1 import merge_ranges, render_ranges
    by = {p['source_id']: p for p in source['paragraphs']}
    ranks = {uid: i for i, uid in enumerate(sentence_order(source, scores))}
    merged = {pid: merge_ranges(by[pid]['text'], rr) for pid, rr in ranges.items() if rr}
    def key(pid):
        return min(ranks[pid + ':span:' + str(i)] for i, sp in enumerate(by[pid]['spans'])
                   if any(a < sp['end'] and b > sp['start'] for a, b in merged[pid]))
    merged = {pid: merged[pid] for pid in sorted(merged, key=key)}
    text = render_ranges(source, merged)
    return dict(evidence_text=text, evidence_tokens=count(text), selected_ranges=merged,
                selected_spans=[])


def extend(ranges, item):
    result = deepcopy(ranges)
    result.setdefault(item['source'], []).append([item['start'], item['end']])
    return result


def covered(item, ranges):
    return any(a <= item['start'] and b >= item['end']
               for a, b in ranges.get(item['source'], []))


def retention(proposal, ranges):
    """Containment, not candidate identity; overlapping quotations count as proposed."""
    candidates = proposal['candidates']
    kept = [covered(c, ranges) for c in candidates]
    return dict(quote_count=len(kept), retained_quotes=sum(kept),
                quote_retention=sum(kept) / len(kept) if kept else None,
                all_quotes_retained=all(kept) if kept else None)


def candidate_pool(source, scores, proposal, fallback):
    """Verified quotes + exactly the sentence universe used by production filling.

    A quote inherits the maximum score of its intersecting original sentences.
    Equal scores use native paragraph, start, end; no type-specific preference.
    Identical spans are deduplicated and overlaps are merged only at rendering.
    """
    by = {p['source_id']: p for p in source['paragraphs']}
    native = {p['source_id']: i for i, p in enumerate(source['paragraphs'])}
    items = {}
    for c in proposal['candidates']:
        assert by[c['source']]['text'][c['start']:c['end']] == c['text'], 'Quote provenance'
        key = (c['source'], c['start'], c['end'])
        items.setdefault(key, dict(source=key[0], start=key[1], end=key[2], kinds=[]))['kinds'].append('quote')
    for uid in fallback['selected_spans']:
        c = span_for(source, uid)
        key = (c['source'], c['start'], c['end'])
        items.setdefault(key, dict(c, kinds=[]))['kinds'].append('fallback_sentence')
    for item in items.values():
        pid = item['source']
        overlaps = [scores[pid + ':span:' + str(i)] for i, s in enumerate(by[pid]['spans'])
                    if item['start'] < s['end'] and item['end'] > s['start']]
        assert overlaps and all(math.isfinite(x) for x in overlaps)
        item['score'] = max(overlaps)
        item['kinds'] = sorted(set(item['kinds']))
    return sorted(items.values(), key=lambda c: (-c['score'], native[c['source']], c['start'], c['end']))


def pack(source, scores, pool, seed, cap, count):
    plan = render(source, scores, seed, count)
    assert plan['evidence_tokens'] <= cap, 'Anchor alone exceeds target; stop before inference'
    accepted = []
    for i, candidate in enumerate(pool):
        if covered(candidate, plan['selected_ranges']):
            continue
        trial = render(source, scores, extend(plan['selected_ranges'], candidate), count)
        if trial['evidence_tokens'] <= cap:
            plan = trial
            accepted.append(i)
    plan['accepted_pool_indices'] = accepted
    return plan


def make_controls(source, scores, proposal, fallback, original, filled, trace, cap, count):
    """All original non-admissions use exactly the original fallback in all arms."""
    if trace.get('fallback'):
        assert original['evidence_text'] == fallback['evidence_text'] == filled['evidence_text']
        return {name: deepcopy(original) for name in ARMS}, dict(
            admitted=False, fallback=trace['fallback'], pool=[], changed_tight=False, changed_full=False)
    pool = candidate_pool(source, scores, proposal, fallback)
    anchor = span_for(source, trace['anchor'])
    anchor_ranges = extend({}, anchor)
    controls = dict(zip(ARMS[:2], (deepcopy(original), deepcopy(filled))))
    # Independent reconstruction must reproduce the existing production fill byte-for-byte.
    check = pack(source, scores, pool, original['selected_ranges'], cap, count)
    assert check['evidence_text'] == filled['evidence_text'], 'Common-pool fill differs from production'
    controls[ARMS[2]] = pack(source, scores, pool, anchor_ranges, original['evidence_tokens'], count)
    controls[ARMS[3]] = pack(source, scores, pool, anchor_ranges, cap, count)
    for arm, plan in controls.items():
        assert covered(anchor, plan['selected_ranges']), 'Anchor lost'
        assert count(plan['evidence_text']) == plan['evidence_tokens'] <= cap
        if arm in ARMS[:2]:
            assert retention(proposal, plan['selected_ranges'])['all_quotes_retained']
    return controls, dict(admitted=True, fallback=None, anchor=anchor, pool=pool,
        changed_tight=controls[ARMS[0]]['evidence_text'] != controls[ARMS[2]]['evidence_text'],
        changed_full=controls[ARMS[1]]['evidence_text'] != controls[ARMS[3]]['evidence_text'])
