# JQR: Joint Retention of Source-Traceable Evidence

Research code and frozen evaluation records for **Joint Retention of Source-Traceable Evidence for Budgeted Multi-Hop Question Answering**.

JQR validates source quotations, admits the complete proposed quotation collection when it fits, retains an anchor sentence, and stops filling. Otherwise it uses the frozen sentence-based fallback. The published construction never reads reference answers or supporting-fact labels.

## Start here

Python 3.12 or later is recommended. The record checks use the standard library. Install the CPU dependencies for interval reproduction, construction replay and figures:

```bash
python -m pip install -r requirements-reproduce.txt
python reproduce.py verify
python reproduce.py statistics --bootstrap
python -m unittest discover -s tests -v
python reproduce.py figures
```

`statistics --bootstrap` recomputes all 32 budget score groups, all 16 paired gain intervals, and all 12 budget-interaction intervals with the original component bootstrap, 10,000 draws and seeds. It also checks all 7,740 original 512-token score reproductions. Generated files go to `outputs/`, separate from the frozen records.

## Reconstruct the evidence without a GPU

The exact [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) tokenizer is included as `tokenizers/qwen3.5-9b.json`, with its Apache 2.0 license. Its SHA-256 is in `environment/tokenizer.json`; the command refuses a different tokenizer. No model-weight download is needed.

```bash
python reproduce.py construct
```

This reconstructs JQR and full-candidate reranker packing for **1,935 questions 脳 4 budgets**, checking all 15,480 evidence strings and their token counts against the frozen experiment. `--limit 10` is a quick smoke test. The replay contains the original candidate passages, sentence scores, predicted fallback paths, parsed quotation proposals, and proposal attempts. It uses the same production Python functions, not a simplified implementation.

For new construction inputs, call `reader_memory.budget_sweep_core_v1.construct(source, scores, seeds, proposal, counter, budget)`. Its input schema is demonstrated by the released replay records and the synthetic unit tests. The budget is one of 128/256/384/512 and the counter must count the final rendered evidence.

## Source map

| Paper component | Production source |
|---|---|
| Whole quotation collection | `reader_memory/conjunctive_quote_plan.py` |
| Anchor and early stopping | `reader_memory/anchored_quote_v1.py` |
| Fixed-budget construction and fallback | `reader_memory/budget_sweep_core_v1.py` |
| Proposal parsing and validation | `reader_memory/role_quote_v1.py`, `hotpot_confirmation_v1.py` |
| Inference, repair and complete requests | `reader_memory/paper_confirmation_runtime_v1.py` |
| Original evaluation and intervals | `reader_memory/paper_confirmation_analysis_v1.py`, `budget_sweep_v1.py` |
| Compact selection controls | `reader_memory/compact_reranker_control_core_v1.py` |
| Exact-length controls | `reader_memory/proposal_retention_v1.py` |

The historical `reader_memory` namespace is preserved to keep the evaluated implementation intact. Older internal identifiers such as `quote_anchor`, `strong_rerank` and `ap3` mean JQR, reranker packing and the fallback construction in these records. They are implementation names, not additional public paper baselines. Other historical modules are retained as dependencies and provenance, not additional JQR contributions.

## What can be reproduced

1. **Record and statistical verification:** no GPU or model weights. All completed conditions, including failed answers, are retained. The system and fixed-budget panels are separate from the 480-question public-comparison panel.
2. **Exact evidence construction:** CPU plus the pinned tokenizer. The release includes frozen upstream predictions, allowing the reported JQR evidence to be reconstructed without rerunning proposal generation or the retriever.
3. **Fresh model inference:** original phase executors, prompts and runtime settings are included. These historical executors also expect their original checkpoint/data/manifest directory structure; they are not a one-command fresh-machine launcher. Public model weights must be obtained separately. The fallback Beam checkpoint was adapted before the reported evaluation and then frozen (SHA-256 `eef0b33fabbf69d998d3d339685cecfc2e7aafe61da4e72572c22d8396184bdd`); its predicted paths are included, but its 701 MiB checkpoint is not bundled. Re-running the upstream retriever requires that checkpoint or reconstructing its training workflow. This boundary does not affect the CPU construction and statistical reproductions above.

See [DATA.md](DATA.md) for data provenance and reuse terms, [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the experiment-to-artifact mapping, and `environment/` for pinned model/data metadata. No new experimental scores are generated by the release checks. Frozen historical decisions in archived records retain their original wording and do not describe the current manuscript's submission status.

## License

Original project source is released under the MIT license. Third-party code, benchmark text and model artifacts retain their original licenses; see [THIRD_PARTY.md](THIRD_PARTY.md). This repository contains no model weights, server credentials, or private deployment configuration.
