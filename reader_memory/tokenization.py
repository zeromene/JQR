import re


class TokenCounter:
    exact = True

    def __init__(self, model_path):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))


class ApproximateCounter:
    """Only for CPU fixtures; never accepted for production budget claims."""
    exact = False

    def count(self, text):
        return len(re.findall(r"\w+|[^\w\s]", text))
