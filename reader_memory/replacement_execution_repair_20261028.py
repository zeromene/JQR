"""Execution-only wrapper; frozen algorithm, inputs, and scientific identity are unchanged."""
import argparse
from pathlib import Path
from . import replacement_learning_v1 as frozen
from . import baseline_bootstrap_v2 as base
from .schema import fingerprint

RUN = Path(__file__).resolve().parents[1] / 'runs/replacement_learning_v1_repair1'


def verify_execution():
    meta = frozen.load(RUN / 'execution_manifest.json')
    assert fingerprint({k: v for k, v in meta.items() if k != 'signature'}) == meta['signature']
    assert meta['scientific_signature'] == frozen.verify()
    for path, digest in meta['files'].items():
        if base.sha(frozen.ROOT / path) != digest:
            raise ValueError('Changed execution repair: ' + path)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preflight', 'rewards', 'score_rewards', 'train', 'select', 'qa', 'score'])
    parser.add_argument('--endpoint', default='http://127.0.0.1:8018')
    args = parser.parse_args()
    verify_execution()
    frozen.RUN = RUN
    # Exercise the actual parsing path before every stage, including before GPU service launch.
    assert frozen.parse_answer('{"answer":"Berlin"}') == 'Berlin'
    assert frozen.parse_answer('```json\n{"answer":"Berlin"}\n```') == 'Berlin'
    assert base.native_f1('Berlin', 'Berlin', 2) == 1.
    if args.mode in ('rewards', 'qa'):
        getattr(frozen, args.mode)(args.endpoint)
    else:
        getattr(frozen, args.mode)()


if __name__ == '__main__':
    main()
