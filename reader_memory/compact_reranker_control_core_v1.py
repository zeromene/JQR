"""Full-pool production sentence packing and answer-blind C-construction audit."""
from copy import deepcopy
from . import quote_retention_mechanism_core_v1 as quotes
from . import relation_preserving_pilot_v1 as production


def full_pool_plan(source, scores, counter, cap, anchor=None):
    """Use the unchanged production packer; never inspect proposals or answers."""
    plan, trace = production.pack(source, scores, 'recomp_sentence', counter, cap)
    assert counter.count(plan['evidence_text']) == plan['evidence_tokens'] <= cap
    if anchor is not None:
        assert quotes.covered(anchor, plan['selected_ranges']), 'Original fitting anchor lost'
    return plan, trace


def geometry(source, ranges):
    full = partial = intervals = 0
    for p in source['paragraphs']:
        rr = ranges.get(p['source_id'], [])
        intervals += len(rr)
        for sp in p['spans']:
            if any(a <= sp['start'] and b >= sp['end'] for a, b in rr):
                full += 1
            elif any(a < sp['end'] and b > sp['start'] for a, b in rr):
                partial += 1
    return dict(whole_sentences=full,partly_included_sentences=partial,rendered_intervals=intervals)


def traced_pack(source, scores, pool, anchor, cap, count):
    seed = quotes.extend({}, anchor)
    plan = quotes.render(source, scores, seed, count)
    assert plan['evidence_tokens'] <= cap
    steps, accepted = [], []
    for i, item in enumerate(pool):
        if quotes.covered(item, plan['selected_ranges']):
            steps.append(dict(index=i,decision='already_covered',before_tokens=plan['evidence_tokens']))
            continue
        trial = quotes.render(source, scores, quotes.extend(plan['selected_ranges'],item),count)
        accept = trial['evidence_tokens'] <= cap
        steps.append(dict(index=i,decision='accepted' if accept else 'over_cap',
            before_tokens=plan['evidence_tokens'],trial_tokens=trial['evidence_tokens'],
            incremental_tokens=trial['evidence_tokens']-plan['evidence_tokens']))
        if accept:
            accepted.append(i)
            plan = trial
    plan['accepted_pool_indices'] = accepted
    original = quotes.pack(source,scores,pool,seed,cap,count)
    assert plan == original, 'Tracing changed the frozen C algorithm'
    return plan,steps


def construction_audit(source, scores, proposal, fallback, a, c, anchor, cap, count):
    pool = quotes.candidate_pool(source,scores,proposal,fallback)
    rebuilt,steps = traced_pack(source,scores,pool,anchor,cap,count)
    assert rebuilt['evidence_text'] == c['evidence_text']
    proposals = proposal['candidates']
    lost = [i for i,q in enumerate(proposals) if not quotes.covered(q,c['selected_ranges'])]
    retained = [i for i in range(len(proposals)) if i not in lost]
    by_span = {(x['source'],x['start'],x['end']):i for i,x in enumerate(pool)}
    losses = []
    for qi in lost:
        q = proposals[qi]
        pi = by_span[q['source'],q['start'],q['end']]
        step = steps[pi]
        assert step['decision'] == 'over_cap'
        alone = quotes.render(source,scores,quotes.extend(quotes.extend({},anchor),q),count)
        losses.append(dict(quote_index=qi,pool_index=pi,score=pool[pi]['score'],
            with_anchor_tokens=alone['evidence_tokens'],fits_with_anchor=alone['evidence_tokens']<=cap,
            **{k:v for k,v in step.items() if k!='index'}))
    # A construction witness, not a deployable alternative or a necessary-fact label:
    # replace one accepted containing sentence by already proposed subquotes and
    # check whether one lost quote can fit while preserving all previously kept quotes.
    witnesses = []
    for pi in rebuilt['accepted_pool_indices'] if lost else []:
        item = pool[pi]
        if 'fallback_sentence' not in item['kinds']:
            continue
        contained = [q for q in proposals if q['source']==item['source']
            and item['start']<=q['start'] and item['end']>=q['end']
            and (item['start'],item['end'])!=(q['start'],q['end'])]
        if not contained:
            continue
        rr = quotes.extend({},anchor)
        for other in rebuilt['accepted_pool_indices']:
            if other != pi:
                rr = quotes.extend(rr,pool[other])
        for q in contained:
            rr = quotes.extend(rr,q)
        reduced = quotes.render(source,scores,rr,count)
        if reduced['evidence_tokens'] >= c['evidence_tokens']:
            continue
        if not all(quotes.covered(proposals[i],reduced['selected_ranges']) for i in retained):
            continue
        for qi in lost:
            trial = quotes.render(source,scores,quotes.extend(rr,proposals[qi]),count)
            if trial['evidence_tokens'] <= cap:
                same_score_later = [by_span[q['source'],q['start'],q['end']] for q in contained
                    if by_span[q['source'],q['start'],q['end']]>pi
                    and pool[by_span[q['source'],q['start'],q['end']]]['score']==item['score']]
                witnesses.append(dict(sentence_pool_index=pi,recovered_quote_index=qi,
                    before_tokens=c['evidence_tokens'],after_tokens=trial['evidence_tokens'],
                    released_tokens=c['evidence_tokens']-reduced['evidence_tokens'],
                    later_contained_same_score_quote_indices=same_score_later))
                break
    return dict(pool=pool,steps=steps,lost_quotes=losses,
        single_sentence_replacement_witnesses=witnesses,
        A_geometry=geometry(source,a['selected_ranges']),C_geometry=geometry(source,c['selected_ranges']),
        caveat='Proposal containment and geometric construction only; not gold support, semantic redundancy, or causal necessity')
