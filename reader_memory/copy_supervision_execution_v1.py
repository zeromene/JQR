"""Byte-format-only runtime compatibility for the unchanged, frozen QA pilot."""
import argparse

from . import copy_supervision_pilot_v1 as pilot
from .schema import fingerprint

ROOT, PLAN, RUN, CONFIG, load = pilot.ROOT, pilot.PLAN, pilot.RUN, pilot.CONFIG, pilot.load
META = 'models/Qwen3-Reranker-4B/artifact_manifest.json'
MANIFEST = PLAN / 'execution_compat/manifest.json'


def freeze():
    parent = load(PLAN / 'manifest.json')
    names = ['reader_memory/copy_supervision_execution_v1.py',
             'scripts/copy_supervision_execution_v1_queue.py',
             'research/copy_supervision_pilot_v1/execution_compat/NOTES.md']
    value = dict(version='copy-pilot-runtime-metadata-v1', parent_signature=parent['signature'],
        parent_manifest_sha256=pilot.base.sha(PLAN / 'manifest.json'), config=CONFIG,
        canonical_model_metadata=load(ROOT / META),
        allowed_metadata_byte_hashes=[parent['files'][META],
            '782179d8ec99b57c268584b123190df6eb05fddb40148c5596f45d6d1d276545'],
        files={n: pilot.base.sha(ROOT / n) for n in names},
        scientific_data_or_thresholds_changed=False, feedback_calls_before_compatibility=0)
    value['signature'] = fingerprint(value)
    pilot.common.immutable_json(MANIFEST, value)
    print('FROZEN_COPY_PILOT_EXECUTION', value['signature'])


def verify():
    value, parent = load(MANIFEST), load(PLAN / 'manifest.json')
    if value['config'] != CONFIG or value['signature'] != fingerprint(
            {k: v for k, v in value.items() if k != 'signature'}):
        raise ValueError('Execution identity changed')
    if (pilot.base.sha(PLAN / 'manifest.json') != value['parent_manifest_sha256']
            or parent['signature'] != value['parent_signature']
            or parent['config'] != CONFIG
            or parent['signature'] != fingerprint({k: v for k, v in parent.items() if k != 'signature'})):
        raise ValueError('Frozen scientific parent changed')
    for name, digest in {**parent['files'], **value['files']}.items():
        actual = pilot.base.sha(ROOT / name)
        if name == META:
            if (actual not in value['allowed_metadata_byte_hashes']
                    or load(ROOT / name) != value['canonical_model_metadata']):
                raise ValueError('Model metadata content changed (format exception only)')
        elif actual != digest:
            raise ValueError('Frozen pilot/runtime file changed: ' + name)
    return value['signature']


def execute(mode, endpoint=None):
    original = pilot.verify
    pilot.verify = verify
    try:
        return pilot.feedback(endpoint) if mode == 'feedback' else getattr(pilot, mode)()
    finally:
        pilot.verify = original


def preflight():
    return execute('preflight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'verify', 'preflight', 'feedback', 'score'])
    parser.add_argument('--endpoint', default='http://127.0.0.1:8018')
    args = parser.parse_args()
    if args.mode in ('freeze', 'verify'):
        result = globals()[args.mode]()
    else:
        result = execute(args.mode, args.endpoint)
    if result:
        print(result)
