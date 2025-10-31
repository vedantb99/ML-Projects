# triplet_dataset_snli.py
import random
from datasets import load_dataset
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    """
    Builds (anchor, positive, negative) triplets from SNLI.
    - anchor: premise sentence
    - positive: hypothesis that entails the premise (label == 0)
    - negative: hypothesis from a contradiction pair (label == 2)
    """
    def __init__(self, split="train"):
        # Load the SNLI split
        dataset = load_dataset("snli", split=split)

        # Filter valid examples
        entailment = dataset.filter(lambda ex: ex["label"] == 0 and ex["premise"] and ex["hypothesis"])
        contradiction = dataset.filter(lambda ex: ex["label"] == 2 and ex["premise"] and ex["hypothesis"])

        # Store pairs
        self.positive_pairs = [
            (ex["premise"], ex["hypothesis"]) for ex in entailment
        ]
        self.negative_pairs = [
            (ex["premise"], ex["hypothesis"]) for ex in contradiction
        ]

        print(f"Loaded {len(self.positive_pairs)} positive pairs and {len(self.negative_pairs)} negative pairs.")

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, idx):
        # Anchor-positive pair
        anchor, positive = self.positive_pairs[idx]

        # Random negative pair (sampled from a contradiction)
        neg_premise, negative = random.choice(self.negative_pairs)

        # Return text triplet
        return {
            "anchor": anchor,
            "positive": positive,
            "negative": negative
        }


# ========== Quick sanity check ==========
if __name__ == "__main__":
    dataset = TripletDataset(split="train")
    print(dataset[0])
