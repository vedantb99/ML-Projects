"""
=====================================================
Fine-tuning DistilBERT on SST-2 Sentiment Classification
=====================================================

This script:
  - Loads the GLUE/SST-2 dataset
  - Tokenizes text with truncation and padding
  - Converts data to PyTorch tensors
  - Builds DataLoaders
  - Implements a custom training & validation loop
"""

# =====================================================
# Imports
# =====================================================
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from torch.optim import AdamW
from tqdm import tqdm

# =====================================================
# 1. Load Dataset
# =====================================================
# Dataset: SST-2 from GLUE benchmark
dataset = load_dataset("glue", "sst2")
print(dataset)

# =====================================================
# 2. Tokenizer & Preprocessing
# =====================================================
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def preprocess_function(batch):
    """
    Tokenize a batch of examples.
    - truncation=True cuts long sentences
    - padding="max_length" ensures fixed-size tensors
    """
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# Apply preprocessing
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set PyTorch tensor format
tokenized_datasets.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

# Split into train and validation
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]

# =====================================================
# 3. DataLoaders
# =====================================================
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    train_dataset, shuffle=True, batch_size=16, collate_fn=data_collator
)
val_loader = DataLoader(
    val_dataset, batch_size=16, collate_fn=data_collator
)

# =====================================================
# 4. Model & Optimizer
# =====================================================
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2  # SST-2 has positive/negative labels
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

# =====================================================
# 5. Training and Validation Functions
# =====================================================
def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        # Move entire batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(preds)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# =====================================================
# 6. Training Loop
# =====================================================
num_epochs = 3
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, device)

    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# =====================================================
# 7. Save the Model
# =====================================================
model.save_pretrained("distilbert-sst2-finetuned")
tokenizer.save_pretrained("distilbert-sst2-finetuned")

print("\n✅ Training complete. Model saved to 'distilbert-sst2-finetuned/'")
