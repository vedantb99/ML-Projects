# In data_loader.py

import torch
import random
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer # Make sure this is imported

# ... SNLITripletDataset class remains exactly the same ...
class SNLITripletDataset(Dataset):
    """
    A custom Dataset for loading SNLI triplets.
    ... (no changes here) ...
    """
    def __init__(self, split='train'):
        print(f"Loading SNLI dataset for {split} split...")
        dataset = load_dataset("snli", split=split)
        
        self.positive_pairs = []
        self.negative_pairs = []

        # Filter out pairs with -1 label (no consensus)
        dataset = dataset.filter(lambda example: example['label'] != -1)
        
        # Pre-process into positive and negative pools
        for example in dataset:
            anchor = example['premise']
            hypothesis = example['hypothesis']
            label = example['label']

            if label == 0: # Entailment
                self.positive_pairs.append((anchor, hypothesis))
            elif label == 2: # Contradiction
                self.negative_pairs.append((anchor, hypothesis))
        
        if not self.positive_pairs or not self.negative_pairs:
             # Handle splits like 'test' which have no contradictions
             print(f"Warning: Not enough entailment/contradiction pairs in {split} split. Using fallbacks.")
             if not self.negative_pairs:
                 # Use contradictions from train split as a fallback for dev/test
                 train_data_neg = load_dataset("snli", split='train').filter(lambda ex: ex['label'] == 2)
                 for ex in train_data_neg:
                     self.negative_pairs.append((ex['premise'], ex['hypothesis']))

        print(f"Loaded {len(self.positive_pairs)} positive pairs.")
        print(f"Loaded {len(self.negative_pairs)} negative pairs.")

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, idx):
        anchor, positive = self.positive_pairs[idx % len(self.positive_pairs)] # Use modulo for safety
        _, negative = random.choice(self.negative_pairs)
        
        return anchor, positive, negative

# === NEW CODE ===
# Replaces create_collate_fn
# In data_loader.py

class TripletBatchCollator:
    """
    A collate function class that tokenizes a batch of
    (anchor, positive, negative) text triplets.
    """
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        anchors, positives, negatives = zip(*batch)
        
        # --- CHANGE HERE: Use padding="max_length" ---
        tokenized_anchors = self.tokenizer(
            list(anchors), 
            padding="max_length", # Explicitly pad to max_length
            truncation=True, 
            return_tensors="pt",
            max_length=self.max_length
        )
        tokenized_positives = self.tokenizer(
            list(positives), 
            padding="max_length", # Explicitly pad to max_length
            truncation=True, 
            return_tensors="pt",
            max_length=self.max_length
        )
        tokenized_negatives = self.tokenizer(
            list(negatives), 
            padding="max_length", # Explicitly pad to max_length
            truncation=True, 
            return_tensors="pt",
            max_length=self.max_length
        )
        # --- END CHANGE ---
        
        return tokenized_anchors, tokenized_positives, tokenized_negatives

# ... rest of the file ...
# ... your __main__ test block ...
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    train_dataset = SNLITripletDataset(split='train')
    print("\n--- Sample Triplet ---")
    print(train_dataset[0])
    
    # === TEST NEW CLASS ===
    collate_fn = TripletBatchCollator(tokenizer)
    
    dummy_batch = [train_dataset[i] for i in range(4)]
    
    print("\n--- Collated Batch ---")
    anchors, positives, negatives = collate_fn(dummy_batch) # Test the __call__
    
    print("Anchors batch shape:", anchors['input_ids'].shape)
    print("Positives batch shape:", positives['input_ids'].shape)
    print("Negatives batch shape:", negatives['input_ids'].shape)