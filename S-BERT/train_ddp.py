import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from scipy.stats import spearmanr
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# ======= DDP Imports =======
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
# ===========================

# ======= Import custom components =======
from sbert_model import SBERTEncoder
from data_loader import SNLITripletDataset, TripletBatchCollator

# ==============================================
#           DDP HELPER FUNCTIONS
# ==============================================
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# ==============================================
#             TRAINING LOOP (Adapted)
# ==============================================
# --- DDP CHANGE: Added rank and epoch ---
# In train_sbert_ddp.py

# In train_sbert_ddp.py

def train_one_epoch(model, dataloader, optimizer, criterion, device, rank, epoch):
    model.train()
    running_loss = 0.0
    total_samples = 0
    num_batches = 0

    dataloader.sampler.set_epoch(epoch)
    dataloader_iterable = dataloader # Removed tqdm

    for anchors, positives, negatives in dataloader_iterable:
        # Move inputs to the correct device first
        anc_in = {k: v.to(device) for k, v in anchors.items()}
        pos_in = {k: v.to(device) for k, v in positives.items()}
        neg_in = {k: v.to(device) for k, v in negatives.items()}

        # --- NEW: Concatenate inputs ---
        # Store original batch size (per GPU)
        batch_size_gpu = anc_in['input_ids'].size(0)

        # Combine input_ids and attention_masks
        # Shape becomes (batch_size_gpu * 3, seq_len)
        all_input_ids = torch.cat([anc_in['input_ids'], pos_in['input_ids'], neg_in['input_ids']], dim=0)
        all_attention_mask = torch.cat([anc_in['attention_mask'], pos_in['attention_mask'], neg_in['attention_mask']], dim=0)
        # --- END NEW ---

        # --- NEW: Single Forward Pass ---
        # Pass the combined batch through the model *once*
        all_embeddings = model(input_ids=all_input_ids, attention_mask=all_attention_mask)
        # --- END NEW ---

        # --- NEW: Split Embeddings ---
        # Split the resulting embeddings back into anchor, positive, negative
        # Each part will have shape (batch_size_gpu, embedding_dim)
        anchor_emb, pos_emb, neg_emb = torch.split(all_embeddings, batch_size_gpu, dim=0)
        # --- END NEW ---

        # Compute triplet loss (criterion works the same)
        loss = criterion(anchor_emb, pos_emb, neg_emb)

        # Backpropagation (works the same)
        optimizer.zero_grad()
        loss.backward() # DDP syncs gradients from the single forward pass
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * batch_size_gpu
        total_samples += batch_size_gpu
        num_batches += 1

    avg_loss = running_loss / total_samples if total_samples > 0 else 0.0

    if rank == 0:
        print(f"Epoch Training Loss: {avg_loss:.4f} (processed {num_batches} batches)")

    return avg_loss
# ==============================================
#           VALIDATION LOOP (Unchanged)
# ==============================================
# This function is expensive and doesn't need to be run by all processes.
# We will only call it from rank 0.
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

            # --- THE FIX ---
            # Call the main 'model' wrapper, not 'model.module'.
            # torch.no_grad() will prevent DDP from doing work.
            emb1 = model(**tok_s1)
            emb2 = model(**tok_s2)
            # --- END FIX ---

            all_embeddings1.append(emb1.cpu())
            all_embeddings2.append(emb2.cpu())
    
    # ... rest of the function is unchanged ...
    emb1 = torch.cat(all_embeddings1)
    emb2 = torch.cat(all_embeddings2)

    cosine_scores = torch.nn.functional.cosine_similarity(emb1, emb2).numpy()
    corr, _ = spearmanr(cosine_scores, np.array(scores))
    
    print(f"Validation Spearman Correlation: {corr:.4f}")
    return corr

# ==============================================
#              MAIN DDP WORKER
# ==============================================
def main_worker(rank, world_size, num_epochs, batch_size, lr):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    
    device = rank  # Each process gets one GPU
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # --- DDP CHANGE: Model to device BEFORE DDP wrapper ---
    model = SBERTEncoder().to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)    
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = AdamW(model.parameters(), lr=lr)

    # --- DDP CHANGE: Custom Dataset + Sampler ---
    train_dataset = SNLITripletDataset(split='train')
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True,drop_last=True)
    
    collate_fn = TripletBatchCollator(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Sampler handles shuffling
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        sampler=train_sampler
    )
    # --- END DDP CHANGE ---

    best_corr = 0.0
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
            
        train_one_epoch(model, train_loader, optimizer, criterion, device, rank, epoch)
        
        # --- DDP CHANGE: Only validate and save on rank 0 ---
        if rank == 0:
            corr = validate(model, tokenizer, device)
            if corr > best_corr:
                best_corr = corr
                # Save the .module's state_dict
                torch.save(model.module.state_dict(), "best_sbert_ddp.pt")
                print(f"New best model saved! Spearman = {best_corr:.4f}")
        # --- END DDP CHANGE ---

    cleanup()

# ==============================================
#              MAIN LAUNCHER
# ==============================================
if __name__ == "__main__":
    world_size = 2
    num_epochs = 2
    batch_size = 16  # This is per-GPU, so effective batch size is 32
    lr = 2e-5
    
    args = (num_epochs, batch_size, lr)
    mp.spawn(
        main_worker,
        args=(world_size, *args),
        nprocs=world_size,
        join=True
    )
    print("\nDDP Training complete!")