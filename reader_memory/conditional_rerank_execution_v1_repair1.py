"""A metadata-key adapter; preserve all frozen scientific code and selected text."""
import argparse
from pathlib import Path

from . import conditional_rerank_execution_v1 as original
from .schema import fingerprint

ROOT = original.ROOT
RUN = ROOT / 'runs/conditional_rerank_execution_v1_repair1'
PLAN = ROOT / 'research/conditional_rerank_execution_v1_repair1'
EXPECTED_SCIENCE = '162301f0048b9585cd1706bdf845ec631fc2a6e02d2a220963e917c861bcf488'
EXPECTED_PLANS = 'bd5b5ccd514abc0d9a752d25b702e2602428db0064c111bf73389b96df8ada84'
load, sha, immutable = original.load, original.sha, original.immutable


def compatibility_meta(value, actual_plans_sha):
    """Return the legacy key in memory only; reject a conflicting alias."""
    if value['qa_plans_sha256'] != actual_plans_sha:
        raise ValueError('Selected QA plans changed')
    if 'plans_sha256' in value and value['plans_sha256'] != actual_plans_sha:
        raise ValueError('Conflicting QA plan hash keys')
    return dict(value, plans_sha256=actual_plans_sha)


def prepare():
    if original.verify() != EXPECTED_SCIENCE:
        raise ValueError('Unexpected scientific identity')
    if (RUN / 'queue_process.json').exists():
        raise ValueError('Already submitted; no repeated repair')
    old = original.RUN
    if load(old / 'queue_status.json')['status'] != 'failed':
        raise ValueError('Original queue not failed')
    if "KeyError: 'plans_sha256'" not in (old / 'qa.log').read_text(encoding='utf-8'):
        raise ValueError('This adapter addresses only the known metadata failure')
    for name in ['qa/predictions.jsonl', 'qa/answer_calls.jsonl', 'qa/inflight.json']:
        if (old / name).exists():
            raise ValueError('Unexpected prior QA state; inspect instead of retrying')
    if list((old / 'qa/cache').glob('*.json')):
        raise ValueError('Unexpected prior QA cache')
    meta = load(old / 'selection_complete.json')
    actual = sha(old / 'qa_plans.json')
    if actual != EXPECTED_PLANS or meta['uncached_requests'] != 113 or not meta['length_gate_passed']:
        raise ValueError('Frozen selection or call cap changed')
    compatibility_meta(meta, actual)
    names = ['reader_memory/conditional_rerank_execution_v1_repair1.py',
        'scripts/conditional_rerank_execution_v1_repair1_queue.py',
        'tools/conditional_rerank_execution_v1_repair1_remote.py',
        'tests/test_conditional_rerank_execution_v1_repair1.py',
        'research/conditional_rerank_execution_v1_repair1/REPAIR.md',
        'research/conditional_rerank_execution_v1/manifest.json']
    names += ['runs/conditional_rerank_execution_v1/' + n for n in [
        'selection_complete.json', 'qa_plans.json', 'scoring_complete.json', 'preflight.json',
        'length_audit.json', 'selection_traces.json', 'queue_status.json', 'qa.log']]
    value = dict(scientific_signature=EXPECTED_SCIENCE, qa_plans_sha256=actual,
        original_new_qa_calls=0, remaining_new_qa_calls=113,
        change='in-memory metadata alias qa_plans_sha256 -> plans_sha256',
        selection_rerun=False, method_changed=False, new_experiment=False,
        files={n: sha(ROOT / n) for n in names})
    value['repair_signature'] = fingerprint(value)
    immutable(PLAN / 'manifest.json', value)
    print('PREPARED_EXECUTION_REPAIR', value['repair_signature'], flush=True)


def verify():
    value = load(PLAN / 'manifest.json')
    if value['repair_signature'] != fingerprint({k: v for k, v in value.items() if k != 'repair_signature'}):
        raise ValueError('Repair identity changed')
    if original.verify() != value['scientific_signature']:
        raise ValueError('Scientific identity changed')
    for name, digest in value['files'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Repair dependency changed: ' + name)
    return value['repair_signature']


def qa(endpoint):
    verify()
    ledger = original.recomp.ledger
    prior_load = ledger.load
    target = original.RUN / 'selection_complete.json'
    def adapted_load(path):
        value = prior_load(path)
        if Path(path) == target:
            return compatibility_meta(value, sha(original.RUN / 'qa_plans.json'))
        return value
    ledger.load = adapted_load
    try:
        original.qa(endpoint)
    finally:
        ledger.load = prior_load


def score():
    verify()
    original.score()
    immutable(RUN / 'completion_receipt.json', dict(repair_signature=verify(),
        scientific_signature=EXPECTED_SCIENCE, report_sha256=sha(original.RUN / 'report.json'),
        results_directory=original.RUN.relative_to(ROOT).as_posix(),
        scientific_files_unchanged=True, automatic_followup=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'verify', 'qa', 'score'])
    p.add_argument('--endpoint', default='http://127.0.0.1:8035')
    a = p.parse_args()
    result = qa(a.endpoint) if a.mode == 'qa' else globals()[a.mode]()
    if result:
        print(result, flush=True)
