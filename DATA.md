# Data availability and provenance

The experiments use existing public benchmarks, not newly collected human-subject data.

| Dataset | Upstream source | Reuse terms |
|---|---|---|
| HotpotQA | https://hotpotqa.github.io/ ; https://github.com/hotpotqa/hotpot | Dataset: CC BY-SA 4.0. Official evaluation code: Apache 2.0. |
| MuSiQue-Ans v1.0 | https://github.com/StonyBrookNLP/musique | CC BY 4.0; the upstream license is included in `licenses/`. |

The original Wikipedia-derived passages retain their source rights and attribution. This repository's MIT license does not relicense benchmark text or third-party artifacts. HotpotQA-derived data remain under CC BY-SA 4.0. MuSiQue-derived data remain under CC BY 4.0. Original benchmark titles and native identifiers are retained in the construction records.

## Released data

- `artifacts/system_allocation.json`: all 1,935 system-panel native IDs, dataset labels, source-component assignments and exact-length diagnostic flags.
- `artifacts/construction_inputs.jsonl.gz`: all 1,935 questions and candidate passages, frozen sentence scores, Beam paths, proposals and parser outputs, plus expected evidence hashes for four budgets. Passage title, source position and native question ID identify the benchmark source. This is a selected benchmark subset with project-generated predictions, not a newly annotated dataset. Reference answers are absent from construction inputs.
- `artifacts/budget_analysis.json.gz`: all 30,960 scored logical conditions, answer strings, failures, token counts, groups, comparisons and interactions.
- `artifacts/system_analysis.json.gz`: all 19,576 complete-system and exact-length diagnostic conditions, including non-significant results and failures.
- `figures/figure_data.json`: question-level scores and plotting data for the separate MuSiQue-480 public-method panel, plus published case excerpts.
- `artifacts/budget_raw_completion_audit.json.gz`: completion checks, source-component metadata and original file hashes.

The system panels were allocated from benchmark training pools with the project's exclusion rules; they are not the official hidden test sets. The separate 480-question public comparison uses MuSiQue development questions. Sample allocations and upstream checksums are provided, so these populations can be distinguished.

## Obtain the full original benchmarks

Use the official links above. Exact evaluated download revisions, mirror locations and whole-file hashes are in `environment/datasets/`. The MuSiQue mirror is documented as a pinned mirror rather than falsely presented as an independently verified official archive. Model weights are not part of the data release and retain their upstream terms.

The release contains frozen model-generated text and evaluation records. No API tokens, login passwords, SSH keys, private host inventories, GPU process logs or deployment credentials are part of the released data.
