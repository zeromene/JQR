# Reproducibility map

| Manuscript evidence | Released artifact | Verification route |
|---|---|---|
| MuSiQue-480 public comparisons, hops and evidence lengths | `figures/figure_data.json`, `table_data.json` | Individual method score records and published plot inputs |
| Complete systems on HotpotQA-1,400 / MuSiQue-535 | `artifacts/system_analysis.json.gz` | 19,576 records, complete groups and paired intervals |
| Four fixed budgets, two methods, two readers | `artifacts/budget_analysis.json.gz` | `python reproduce.py statistics --bootstrap` |
| Joint admission and fallback at each budget | `artifacts/construction_inputs.jsonl.gz` | `python reproduce.py construct --tokenizer ...` |
| Compact full-pool selection | `artifacts/compact_control_data.json.gz` | Full recorded aggregate and planned contrast tables |
| A/C/D gap decomposition | `artifacts/compact_difference_decomposition.json.gz` | Stored reconstruction and score decomposition |
| Branch-specific answer changes | `artifacts/branch_scores.json.gz` | Full branch denominators and paired scores |
| Method and evidence figures | `figures/render_readable.py` and JSON inputs | `python reproduce.py figures` renders empirical figures only |

The release checks preserve exact source hashes and request identities. They do not rerun LLMs or invent missing independent replications. Construction replay checks text identity and token counts, not merely aggregate means. The interval command resamples source components with the original question weighting and shared draws across budgets/readers, then verifies pointwise and simultaneous intervals.

`reader_memory/` contains the evaluated source code unchanged; `environment/source_hashes.json` records its hashes. `reproduce.py` is a release wrapper around those production functions, with portable file locations. `tests/` contains the self-contained construction and statistical contract tests. Historical tests requiring unpublished local manifests or multiple local model copies are not part of the portable test suite; the full real-input construction replay supplies the integration check.

The archived Beam retriever was trained before the JQR evaluations and held fixed. JQR's proposal/retention policy receives no additional training. The frozen Beam predictions are included for exact construction replay. The original retriever training and inference code is available in `reader_memory/path_retrieval_pilot_v1.py` and `path_retrieval_execution_v1.py`; a fresh upstream rerun needs the corresponding checkpoint or training prerequisites, as stated in the README.

Floating-point generation can vary across hardware and library versions. The published score records remain the reference for the reported numbers; new generation should be written to a separate output directory. Budget lengths are Qwen tokenizer lengths; Phi receives identical evidence strings, and its actual token counts are separately recorded.
