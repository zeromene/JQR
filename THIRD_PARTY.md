# Third-party notices

The MIT license at the repository root covers original project code only.

- **SimpleMem**: the unmodified `answer_generator.py` is retained because a historical dependency extracts two formatting functions from it when the production modules are imported. Its original MIT license and provenance are included under `third_party/simplemem_official/`. This import dependency is not a SimpleMem baseline claim in the JQR paper.
- **LLMLingua, LongMemEval and SimpleMem prompt sources**: upstream license/provenance files are preserved under `third_party/`; project adapters and the small prompt helpers identify their origins.
- **LoCoMo scoring helper**: `reader_memory/_vendor/locomo_LICENSE.txt` accompanies the original helper.
- **HotpotQA and MuSiQue text**: see `DATA.md`; their dataset licenses take precedence over the project source license. `licenses/hotpot-code-LICENSE.txt` is the evaluation-code license, not the HotpotQA dataset license.
- **Public model checkpoints and other baseline implementations** are not redistributed here. Their identifiers and exact available manifests are in `environment/models/`. Downloading or running them is subject to their upstream licenses.

The source tree preserves historical adapter code for research traceability. Such code may require separately installed upstream projects when explicitly invoked; the supported CPU reproduction commands and their dependencies are documented in the README.

- **Qwen3.5-9B tokenizer**: the exact evaluated tokenizer JSON is redistributed under Apache 2.0; see `licenses/qwen-tokenizer-LICENSE.txt`. Original source: https://huggingface.co/Qwen/Qwen3.5-9B . No model weights are included.
