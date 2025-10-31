# train_sbert.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from scipy.stats import spearmanr
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# ======= Import custom components =======
from sbert_model import SBERTEncoder
from data_loader import SNLITripletDataset, create_collate_fn


# ==============================================
#              SETUP & INITIALIZATION
# ==============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model = SBERTEncoder().to(device)
criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = AdamW(model.parameters(), lr=2e-5)

# ==============================================
#                DATA PIPELINES
# ==============================================
train_dataset = SNLITripletDataset(split='train')
collate_fn = create_collate_fn(tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn
)

# ==============================================
#                TRAINING LOOP
# ==============================================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for anchors, positives, negatives in tqdm(dataloader, desc="Training"):
        # Move all tokenized batches to GPU
        anchors = {k: v.to(device) for k, v in anchors.items()}
        positives = {k: v.to(device) for k, v in positives.items()}
        negatives = {k: v.to(device) for k, v in negatives.items()}

        # Forward pass
        anchor_emb = model(**anchors)
        pos_emb = model(**positives)
        neg_emb = model(**negatives)

        # Compute triplet loss
        loss = criterion(anchor_emb, pos_emb, neg_emb)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * anchor_emb.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch Training Loss: {avg_loss:.4f}")
    return avg_loss


# ==============================================
#                VALIDATION LOOP
# ==============================================
def validate(model, tokenizer, device):
    """
    Evaluate on STS-B: compute sentence embeddings for each pair,
    then measure Spearman correlation between cosine similarity and gold score.
    """
    model.eval()
    dataset = load_dataset("stsb_multi_mt", name="en", split="dev")

    sentences1 = dataset["sentence1"]
    sentences2 = dataset["sentence2"]
    scores = dataset["similarity_score"]

    all_embeddings1 = []
    all_embeddings2 = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sentences1), 16), desc="Validating"):
            batch_s1 = sentences1[i:i+16]
            batch_s2 = sentences2[i:i+16]

            tok_s1 = tokenizer(batch_s1, padding=True, truncation=True, return_tensors="pt").to(device)
            tok_s2 = tokenizer(batch_s2, padding=True, truncation=True, return_tensors="pt").to(device)

            emb1 = model(**tok_s1)
            emb2 = model(**tok_s2)

            all_embeddings1.append(emb1.cpu())
            all_embeddings2.append(emb2.cpu())

    emb1 = torch.cat(all_embeddings1)
    emb2 = torch.cat(all_embeddings2)

    # Cosine similarity
    cosine_scores = torch.nn.functional.cosine_similarity(emb1, emb2).numpy()

    # Spearman correlation
    corr, _ = spearmanr(cosine_scores, np.array(scores))
    print(f"Validation Spearman Correlation: {corr:.4f}")
    return corr


# ==============================================
#                MAIN EXECUTION
# ==============================================
if __name__ == "__main__":
    num_epochs = 2
    best_corr = 0.0

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        corr = validate(model, tokenizer, device)

        if corr > best_corr:
            best_corr = corr
            torch.save(model.state_dict(), "best_sbert.pt")
            print(f"New best model saved! Spearman = {best_corr:.4f}")

    print("\nTraining complete!")
    print(f"Best Spearman correlation: {best_corr:.4f}")
