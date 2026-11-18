# Release validation (2026-11-17)

Validated on the release directory, using the published wrapper and copied production source:

- 19 self-contained unit tests passed.
- All 30,960 fixed-budget score records accounted for.
- All 7,740 original 512-token F1/EM reproductions matched.
- All 32 absolute score groups recomputed.
- All 16 method-gain intervals and all 12 budget-interaction intervals reproduced, including simultaneous intervals.
- All 7,740 source × budget pairs reconstructed: **15,480 evidence strings, their token counts, admission decisions and effective budgets matched the frozen records exactly**.
- All seven empirical vector PDF figures rendered successfully from the released inputs.
- No model inference or new experimental data collection was performed for these release checks.

The token-construction check used the Qwen tokenizer with the SHA-256 recorded in `environment/tokenizer.json`. Its engine was the `tokenizers` library; exact agreement against the complete frozen construction matrix validates this CPU counting route for these inputs.
