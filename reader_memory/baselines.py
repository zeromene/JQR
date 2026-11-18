from pathlib import Path


class LLMLingua2Compressor:
    def __init__(self, model_path, device="cpu", rate=0.5):
        from llmlingua import PromptCompressor
        self.model = PromptCompressor(model_name=str(Path(model_path).resolve()), device_map=device,
                                      use_llmlingua2=True,
                                      model_config={"local_files_only": True, "trust_remote_code": False})
        if self.model.model.config.model_type != "bert":
            raise ValueError("This offline baseline requires the staged multilingual BERT checkpoint")
        # LLMLingua 0.2.2 selects word-boundary rules by a substring in model_name.
        # Loading uses the local path above; compression needs the original identity.
        self.model.model_name = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
        self.rate = rate

    def __call__(self, text):
        return self.model.compress_prompt(text, rate=self.rate, force_tokens=["\n", "?", "not", "no"])["compressed_prompt"]
