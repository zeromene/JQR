"""Two pre-fixed ordinary packers, exact rendered cost, auditable decisions."""
import math
import time
from .question_sensitive_baseline_v1 import render


def sigmoid(x):
    return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))


def pack(us, scores, budget, tokenizer, method):
    assert method in ['A', 'B']
    assert len(us) == len(scores) and len({u['id'] for u in us}) == len(us)
    assert all(math.isfinite(s) for s in scores)
    begin = time.perf_counter()
    selected = []
    trace = []
    costs = {}

    def cost(indices):
        key = tuple(sorted(indices))
        if key not in costs:
            costs[key] = len(tokenizer.encode(render(us, key), add_special_tokens=False).ids)
        return costs[key]

    if method == 'A':
        for i in sorted(range(len(us)), key=lambda i: (-scores[i], i)):
            before = cost(selected)
            after = cost(selected+[i])
            accept = after <= budget
            trace.append(dict(id=us[i]['id'], before_tokens=before, trial_tokens=after, accepted=accept))
            if accept:
                selected.append(i)
    else:
        remaining = set(range(len(us)))
        while remaining:
            before = cost(selected)
            feasible = []
            for i in sorted(remaining):
                after = cost(selected+[i])
                if after <= budget:
                    feasible.append((sigmoid(scores[i])/max(1, after-before), i, after))
            if not feasible:
                break
            ratio, i, after = min(feasible, key=lambda x: (-x[0], x[1]))
            trace.append(dict(id=us[i]['id'], before_tokens=before, trial_tokens=after,
                              accepted=True, utility_per_added_token=ratio))
            selected.append(i)
            remaining.remove(i)
    text = render(us, selected)
    n = cost(selected)
    assert n <= budget
    return dict(ids=[us[i]['id'] for i in sorted(selected)], tokens=n, text=text, trace=trace,
                cost_evaluations=len(costs), seconds=time.perf_counter()-begin)
