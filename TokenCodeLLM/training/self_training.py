"""Self-training helpers: sample synthesis and quality scoring."""

import random
from typing import List


class SyntheticGenerator:
    def __init__(self):
        self.names = ["Alex", "AI", "TokenCode", "The System", "User", "The Architect"]
        self.actions = ["optimizes", "generates", "calculates", "dreams of", "analyzes", "refines"]
        self.concepts = ["neural networks", "knowledge graphs", "data streams", "logic chains", "reasoning loops"]

    def build(self, n: int) -> List[str]:
        return [
            f"{random.choice(self.names)} {random.choice(self.actions)} {random.choice(self.concepts)}. "
            + random.choice(["It works perfectly.", "Learning in progress.", "The answer is contextual."])
            for _ in range(n)
        ]


def score_text_quality(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    uniq_ratio = len(set(w.lower() for w in words)) / len(words)
    length_score = min(len(words) / 60.0, 1.0)
    punctuation_bonus = 0.1 if any(p in text for p in ".!?") else 0.0
    return max(0.0, min(1.0, 0.55 * uniq_ratio + 0.35 * length_score + punctuation_bonus))
