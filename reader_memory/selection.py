"""Grouped knapsack plus final serialized-token validation, without gold labels."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re


PATTERNS = {
    "negation": r"\b(?:not|never|no longer|cannot|don't|doesn't|without)\b|不|从未|没有|停止",
    "condition": r"\b(?:if|unless|except|only when|occasionally|but)\b|如果|除非|偶尔|但是|仅当",
    "temporal": r"\b(?:previously|currently|moved|now|before|after|since|last|updated)\b|以前|现在|搬|更新|后来|目前",
    "numeric": r"\d",
}


def features(text):
    # Writer-generated source IDs and session dates must not make every fact "numeric".
    text = re.sub(r"(?m)^\[[^\]\n]+\]\s+\[[^\]\n]*\]\s+[^:\n]+:\s*", "", text)
    return {name: float(bool(re.search(pattern, text, re.I))) for name, pattern in PATTERNS.items()}


@dataclass(frozen=True)
class Choice:
    group_id: str
    representation: str
    text: str
    source_ids: tuple[str, ...]
    date: str
    utility: float
    cost: int


def render_piece(group_id, date, text):
    return f"[{group_id}] [source date: {date or 'unknown'}]\n{text}"


def render_evidence(choices):
    return "\n\n".join(render_piece(choice.group_id, choice.date, choice.text) for choice in choices)


class RuleUtility:
    """Provisional rule model. Its weights are hypotheses, not measured abilities."""
    def __init__(self, profile=None):
        self.profile = profile

    def score(self, query, group, representation, relevance):
        quality = {"raw": 1.0, "fact": 0.94, "summary": 0.78, "llmlingua2": 0.84}[representation]
        info = features(group.variants["raw"])
        risk = 0.5 * sum(info.values()) / len(info)
        if self.profile is not None:
            abilities = self.profile.get("abilities", {})
            risk = sum(value * (1 - float(abilities.get(name, 0.5))) for name, value in info.items()) / len(info)
        # Shared rule selector versus reader-conditioned selector: only the profile differs.
        if representation == "raw":
            quality += 1.5 * risk
        elif representation == "fact":
            quality += 0.5 * risk
        else:
            quality = max(0.1, quality - 0.8 * risk)
        return (0.1 + relevance) * quality


def allocate(query, ranked_groups, counter, budget, strategy="mixed", profile=None, compressor=None):
    if budget < 0:
        raise ValueError("Budget must be nonnegative")
    if strategy == "reader-aware" and profile is None:
        raise ValueError("reader-aware requires an explicit reader profile")
    if strategy == "reader-aware":
        for value in profile.get("abilities", {}).values():
            if not 0 <= float(value) <= 1:
                raise ValueError("Profile abilities must lie in [0, 1]")
    utility = RuleUtility(profile if strategy == "reader-aware" else None)
    maximum = max([score for _, score in ranked_groups] or [1.0]) or 1.0
    options = []
    for group, score in ranked_groups:
        names = [strategy] if strategy in {"raw", "fact", "summary", "llmlingua2"} else ["raw", "fact", "summary"]
        variants = dict(group.variants)
        if strategy == "llmlingua2":
            if compressor is None:
                raise ValueError("LLMLingua-2 strategy needs the local compressor")
            variants["llmlingua2"] = compressor(group.variants["raw"])
        candidates = []
        for name in names:
            if not variants.get(name):
                raise ValueError(f"Missing representation {name} in {group.group_id}")
            cost = counter.count(render_piece(group.group_id, group.date, variants[name]) + "\n\n")
            if 0 < cost <= budget:
                candidates.append(Choice(group.group_id, name, variants[name], tuple(group.source_ids), group.date,
                                         utility.score(query, group, name, score / maximum), cost))
        options.append(candidates)
    # One option per memory group; skip is always available. Sparse states retain backpointers.
    states = {0: (0.0, ())}
    for candidates in options:
        previous = states
        states = dict(previous)
        for spent, (value, selected) in previous.items():
            for choice in candidates:
                new_cost = spent + choice.cost
                if new_cost <= budget and value + choice.utility > states.get(new_cost, (-math.inf, ()))[0]:
                    states[new_cost] = (value + choice.utility, selected + (choice,))
        # Pareto pruning keeps optimal additive-utility states only.
        best, pruned = -math.inf, {}
        for spent in sorted(states):
            if states[spent][0] > best + 1e-12:
                pruned[spent] = states[spent]
                best = states[spent][0]
        states = pruned
    _, selected = max(states.values(), key=lambda item: item[0])
    selected = list(selected)
    # BPE costs need not add. Enforce the budget against the actual final string.
    repairs = 0
    while selected and counter.count(render_evidence(selected)) > budget:
        selected.remove(min(selected, key=lambda choice: choice.utility / max(choice.cost, 1)))
        repairs += 1
    return selected, {"budget": budget, "evidence_tokens": counter.count(render_evidence(selected)), "counter_exact": counter.exact,
                      "selected_groups": len(selected), "serialization_repairs": repairs, "utility_model": "provisional-rules-v1"}
