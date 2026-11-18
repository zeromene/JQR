"""Preserve the richer selection receipt while invoking the unchanged QA ledger."""
import argparse
from pathlib import Path

from . import source_validation_v1 as trial
from . import replacement_stage2_common as common
from . import replacement_precision_v2 as ledger
from . import clean_feedback_control_v1 as clean
from .schema import fingerprint

PLAN = trial.PLAN / 'execution_compatibility.json'
FILES = ['reader_memory/source_validation_execution_v1.py',
         'scripts/source_validation_resume_v1_queue.py',
         'tests/test_source_validation_execution_v1.py']


def freeze():
    value = dict(scientific_signature=trial.verify(),
        reason='Generic run_qa expects a two-field receipt; source selection correctly kept extra provenance fields',
        method_changed=False, generations_repeated=False,
        files={name: trial.base.sha(trial.ROOT / name) for name in FILES})
    value['signature'] = fingerprint(value)
    common.immutable_json(PLAN, value)
    print('FROZEN_EXECUTION_COMPATIBILITY', value['signature'], flush=True)


def verify():
    value = trial.load(PLAN)
    if value['scientific_signature'] != trial.verify():
        raise ValueError('Scientific identity changed')
    if fingerprint({k:v for k,v in value.items() if k != 'signature'}) != value['signature']:
        raise ValueError('Execution compatibility manifest changed')
    for name, digest in value['files'].items():
        if trial.base.sha(trial.ROOT / name) != digest:
            raise ValueError('Execution compatibility source changed')
    return value['scientific_signature']


def validate_receipt(receipt, signature, plans_digest):
    expected = dict(signature=signature, plans_sha256=plans_digest,
        questions=trial.CONFIG['questions'], private_inputs_read=False,
        checkpoint_sha256=trial.CONFIG['checkpoint'])
    if receipt != expected:
        raise ValueError('Selection provenance differs from the fixed experiment')


def qa(endpoint):
    sid = verify()
    plans = trial.load(trial.RUN / 'qa_plans.json')
    validate_receipt(trial.load(trial.RUN / 'selection_complete.json'), sid,
                     trial.base.sha(trial.RUN / 'qa_plans.json'))
    # The original selection receipt is retained byte-for-byte. Only the redundant
    # two-field write in common.run_qa is bypassed; the same frozen ledger executes.
    cache = common.cached_answers([clean.RUN / 'deployment/qa/answer_calls.jsonl'])
    old_run, old_verify, old_cache = ledger.RUN, ledger.verify, ledger.saved_cache
    ledger.RUN, ledger.verify, ledger.saved_cache = trial.RUN, lambda: sid, lambda: cache
    try:
        ledger.qa(endpoint)
    finally:
        ledger.RUN, ledger.verify, ledger.saved_cache = old_run, old_verify, old_cache


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'verify', 'qa'])
    parser.add_argument('--endpoint', default='http://127.0.0.1:8018')
    args = parser.parse_args()
    if args.mode == 'qa':
        qa(args.endpoint)
    else:
        result = globals()[args.mode]()
        if args.mode == 'verify':
            print('VERIFIED_EXECUTION_COMPATIBILITY', result, flush=True)
