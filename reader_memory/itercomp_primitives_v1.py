"""Paper-derived IterCOMP primitives; no dataset inference entry point yet."""
import json
import math


def sparse_weights(token_ids, weights, ignored):
    assert len(token_ids) == len(weights)
    out = {}
    for token, weight in zip(token_ids, weights):
        assert math.isfinite(weight) and weight >= 0
        if token not in ignored and weight > 0: out[str(token)] = max(out.get(str(token), 0.), float(weight))
    return out


def dual_scores(query_dense, query_sparse, document_dense, document_sparse):
    assert len(document_dense) == len(document_sparse) > 0
    scores = []
    for dense, sparse in zip(document_dense, document_sparse):
        assert len(dense) == len(query_dense)
        semantic = sum(float(a) * float(b) for a, b in zip(query_dense, dense))
        lexical = sum(float(v) * float(sparse.get(k, 0)) for k, v in query_sparse.items())
        score = .6 * semantic + .4 * lexical
        assert math.isfinite(score)
        scores.append(score)
    return scores


def percentile_candidates(scores, percentile=90):
    assert scores and all(math.isfinite(v) for v in scores) and 0 <= percentile <= 100
    ordered = sorted(scores); x = (len(ordered) - 1) * percentile / 100; left = math.floor(x); right = math.ceil(x)
    threshold = ordered[left] + (ordered[right] - ordered[left]) * (x - left)
    # Equation5 uses >=; retain ALL threshold ties, do not arbitrarily truncate to top10%.
    return [i for i, score in enumerate(scores) if score >= threshold], threshold


def parse_judgment(text):
    raw = text.strip(); normalization = None
    # Paper Table7 prints mapping syntax inside square brackets; support that literal typo only.
    if raw.startswith('["answer":') or raw.startswith('["answer" :'):
        assert raw.endswith(']')
        raw = '{' + raw[1:-1] + '}'; normalization = 'paper_outer_square_bracket_mapping'
    try: obj = json.loads(raw)
    except (ValueError, TypeError) as exc: raise ValueError('Invalid judge JSON') from exc
    if not isinstance(obj, dict) or set(obj) != {'answer', 'follow_up_question'}: raise ValueError('Invalid judge schema')
    if obj['answer'] not in ('answerable', 'unanswerable') or not isinstance(obj['follow_up_question'], str): raise ValueError('Invalid judge fields')
    if obj['answer'] == 'unanswerable' and not obj['follow_up_question'].strip(): raise ValueError('Missing follow-up query')
    return dict(obj, normalization=normalization)


def accumulate(previous, new, total):
    assert all(type(i) is int and 0 <= i < total for i in [*previous, *new])
    return sorted(set(previous) | set(new))
