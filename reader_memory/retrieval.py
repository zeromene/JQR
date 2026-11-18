"""Small deterministic BM25 index; deliberately shared by every representation."""
from collections import Counter
import math
import re


def words(text):
    return re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", text.lower())


class BM25:
    def __init__(self, groups, k1=1.2, b=0.75):
        self.groups = list(groups)
        self.k1, self.b = k1, b
        self.documents = [Counter(words(group.variants["raw"])) for group in self.groups]
        self.lengths = [sum(document.values()) for document in self.documents]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.df = Counter(term for document in self.documents for term in document)

    def search(self, query, top_k=24):
        terms = set(words(query))
        count = len(self.groups)
        scored = []
        for index, (group, document) in enumerate(zip(self.groups, self.documents)):
            score = 0.0
            for term in terms:
                frequency = document.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (count - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * self.lengths[index] / max(1, self.average_length))
                score += idf * frequency * (self.k1 + 1) / denominator
            scored.append((group, score, index))
        scored.sort(key=lambda item: (-item[1], item[2]))
        return [(group, score) for group, score, _ in scored[:top_k]]
