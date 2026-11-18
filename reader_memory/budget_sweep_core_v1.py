"""Fixed exogenous budgets, using unchanged production constructors."""
from . import short_budget_controls_v1 as short
from . import anchored_quote_v1 as anchored
from types import FunctionType

BUDGETS = (128, 256, 384, 512)
METHODS = ('quote_anchor', 'strong_rerank')


def expanded_arm(method, budget):
    allowed = set(short.original.METHODS) | set(short.original.anchor.METHODS)
    if method not in allowed or budget not in BUDGETS:
        raise ValueError('Unregistered sweep condition')
    return method+'__b'+str(budget)


# Preserve the production function bytecode; extend only historical arm-name
# validation to the newly registered 128/384 budgets. Never mutate old modules.
def expanded(fn):
    return FunctionType(fn.__code__, dict(fn.__globals__, arm=expanded_arm), fn.__name__, fn.__defaults__, fn.__closure__)


guarded_plan = expanded(short.original.anchor.guarded_plan)
fail_plan = expanded(short.original.fail_plan)


def baseline_and_fallback(source, scores, seeds, counter, budget):
    original = short.original
    try:
        baseline, _ = original.ranker.recomp.pack(source, scores, 'recomp_sentence', counter, budget)
    except ValueError as exc:
        if not original.empty_packing_error(exc): raise
        return [fail_plan(source, m, budget, str(exc)) for m in (original.ranker.NEW_METHOD, original.anchor.MAIN)]
    try:
        candidate, _ = original.anchor.packing.seed_fill(source, scores, seeds, counter, original.anchor.PREVIOUS, baseline['evidence_tokens'])
        fallback, _ = guarded_plan(source, scores, baseline, candidate, counter, budget, original.anchor.MAIN)
    except ValueError as exc:
        if not original.empty_packing_error(exc): raise
        fallback = fail_plan(source, original.anchor.MAIN, budget, str(exc))
    return [baseline, fallback]


def construct(source, scores, seeds, proposal, counter, budget):
    if budget not in BUDGETS:
        raise ValueError('Unregistered budget')
    global_plan, fallback = baseline_and_fallback(source, scores, seeds, counter, budget)
    if global_plan.get('selection_error') or fallback.get('selection_error'):
        quote, trace = fallback, dict(fallback='baseline_selection_failure')
    else:
        quote, trace = anchored.quote_plan(source, scores, global_plan, fallback, proposal, counter, 'quote_anchor')
    result = {'quote_anchor': quote, 'strong_rerank': global_plan}
    for plan in result.values():
        assert counter.count(plan['evidence_text']) == plan['evidence_tokens'] <= budget
    assert quote['evidence_tokens'] <= global_plan['evidence_tokens']
    return result, dict(trace, nominal_budget=budget, effective_budget=global_plan['evidence_tokens'],
                        admitted=not bool(trace.get('fallback')))


def bootstrap_matrix(values, cluster_ids, draws=10000, seed=30):
    """Question-weighted component bootstrap; columns share the same resamples."""
    import numpy as np
    x = np.asarray(values, dtype=float)
    assert x.ndim == 2 and len(x) == len(cluster_ids) and len(x) > 0
    assert np.isfinite(x).all()
    labels = sorted(set(cluster_ids))
    rows = {label: [] for label in labels}
    for i, label in enumerate(cluster_ids): rows[label].append(i)
    sums = np.array([x[rows[label]].sum(axis=0) for label in labels])
    sizes = np.array([len(rows[label]) for label in labels])
    rng = np.random.default_rng(seed)
    output = np.empty((draws, x.shape[1]))
    for start in range(0, draws, 128):
        count = min(128, draws-start)
        sample = rng.integers(len(labels), size=(count, len(labels)))
        output[start:start+count] = sums[sample].sum(axis=1)/sizes[sample].sum(axis=1)[:, None]
    return x.mean(axis=0), output


def interval_family(estimates, bootstrap):
    """Marginal and joint max-standardized-deviation intervals, in points."""
    import numpy as np
    est = np.asarray(estimates)
    boot = np.asarray(bootstrap)
    se = boot.std(axis=0, ddof=1)
    active = se > 1e-12
    maxima = np.max(np.abs((boot[:, active]-est[active])/se[active]), axis=1) if active.any() else np.zeros(len(boot))
    critical = float(np.quantile(maxima, .95))
    lower, upper = np.quantile(boot, [.025, .975], axis=0)
    radius = critical*np.where(active, se, 0.)
    return [dict(delta_points=float(100*e), ci95_points=[float(100*lo), float(100*hi)],
                 simultaneous_ci95_points=[float(100*(e-r)), float(100*(e+r))])
            for e, lo, hi, r in zip(est, lower, upper, radius)], critical
