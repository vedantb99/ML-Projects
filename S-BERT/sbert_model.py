# sbert_model.py
import torch
import torch.nn as nn
from transformers import AutoModel

class SBERTEncoder(nn.Module):
    """
    A single SBERT tower that converts a sentence into a 768-d embedding
    using DistilBERT and mean pooling.
    """
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, last_hidden_state, attention_mask):
        """
        Perform mean pooling over token embeddings, weighted by attention mask.
        """
        # last_hidden_state: (batch_size, seq_len, hidden_dim)
        # attention_mask: (batch_size, seq_len)

        # Expand mask dimensions to match embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        # Sum embeddings while ignoring [PAD] tokens
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)

        # Count of valid tokens (avoid division by zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        # Return mean pooled embedding
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through DistilBERT, then mean-pool to get sentence embedding.
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Pool token-level embeddings to get one vector per sentence
        sentence_embedding = self.mean_pooling(last_hidden_state, attention_mask)

        return sentence_embedding


# ========== Quick test ==========
if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = SBERTEncoder()

    sentences = ["A man is running.", "A woman is eating a sandwich."]
    batch = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    embeddings = model(**batch)
    print("Sentence embedding shape:", embeddings.shape)  # (2, 768)
