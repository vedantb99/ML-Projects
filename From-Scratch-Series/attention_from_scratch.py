import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import os
import math
import copy

class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal Positional Encoding layer.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model (int): The embedding dimension (required).
            dropout (float): The dropout probability (default=0.1).
            max_len (int): The maximum sequence length (default=5000).
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a positional encoding matrix of shape [max_len, d_model]
        position = torch.arange(max_len).unsqueeze(1) # [max_len, 1]
        
        # The core formula
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # pe[0, 0::2] = sin(pos * div_term)
        # pe[0, 1::2] = cos(pos * div_term)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension so it can be broadcasted
        # Final shape [1, max_len, d_model]
        pe = pe.unsqueeze(0) 
        
        # Register 'pe' as a buffer. 
        # This means it's part of the model's state, but not a trainable parameter.
        # It will be saved with the model's state_dict and moved to .to(device).
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input embeddings, shape [batch_size, seq_len, d_model]
        
        Returns:
            torch.Tensor: The input embeddings + positional encoding
        """
        # x.size(1) is the sequence length (seq_len)
        # We add the positional encoding up to the length of the input sequence
        # pe is [1, max_len, d_model]. pe[:, :x.size(1)] slices it to [1, seq_len, d_model]
        # This is then broadcast-added to x [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        """
        Args:
            d_model (int): Total dimension of the model.
            h (int): Number of attention heads.
            dropout (float): Dropout probability.
        """
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_model = d_model
        self.h = h
        self.d_head = d_model // h
        
        # --- YOUR CODE HERE ---
        # 1. Define W_q, W_k, W_v linear layers (nn.Linear)
        #    Input dim: d_model, Output dim: d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        
        # 2. Define the output linear layer W_o (nn.Linear)
        #    Input dim: d_model, Output dim: d_model
        self.W_o = nn.Linear(d_model,d_model)
        # 3. Define dropout
        self.dropout = nn.Dropout(p=dropout)

        # --- END YOUR CODE ---
    
    def scaled_dot_product_attention(self, 
                                     query: torch.Tensor, 
                                     key: torch.Tensor, 
                                     value: torch.Tensor, 
                                     mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Calculates the scaled dot-product attention.
        
        Args:
            query (torch.Tensor): Shape [batch_size, h, seq_len_q, d_head]
            key (torch.Tensor): Shape [batch_size, h, seq_len_k, d_head]
            value (torch.Tensor): Shape [batch_size, h, seq_len_v, d_head] (note: seq_len_k == seq_len_v)
            mask (torch.Tensor): Shape [batch_size, 1, 1, seq_len_k] or [batch_size, 1, seq_len_q, seq_len_k]
        
        Returns:
            (torch.Tensor, torch.Tensor): (Output, Attention Weights)
        """
        d_k = query.size(-1) # d_head
        
        # --- YOUR CODE HERE ---
        # 1. Matrix multiply Q and K.T 
        #    (Hint: use torch.matmul). K.T shape should be [batch_size, h, d_head, seq_len_k]
        #    Final scores shape: [batch_size, h, seq_len_q, seq_len_k]
        scores = torch.matmul(query,key.transpose(-2,-1))

        # 2. Scale the scores by 1 / sqrt(d_k)
        scores = scores / math.sqrt(d_k)
        # 3. Apply the mask (if mask is not None)
        #    (Hint: use scores.masked_fill(mask == 0, -1e9))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 4. Apply softmax on the last dimension (seq_len_k) to get weights
        weights = torch.softmax(scores, dim=-1)
        # 5. Apply dropout to the weights
        weights = self.dropout(weights)
        # 6. Matrix multiply weights and V
        #    Final output shape: [batch_size, h, seq_len_q, d_head]
        output = torch.matmul(weights, value)
        # 7. Return (output, weights)
        return output, weights
        # --- END YOUR CODE ---
        
        # Placeholder
        # return torch.zeros_like(query), torch.zeros(query.size(0), self.h, query.size(2), key.size(2))

    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query (torch.Tensor): Shape [batch_size, seq_len_q, d_model]
            key (torch.Tensor): Shape [batch_size, seq_len_k, d_model]
            value (torch.Tensor): Shape [batch_size, seq_len_v, d_model]
            mask (torch.Tensor): Mask for attention.
        
        Returns:
            torch.Tensor: Output tensor, shape [batch_size, seq_len_q, d_model]
        """
        batch_size = query.size(0)
        
        # --- YOUR CODE HERE ---
        
        # 1. Pass query, key, value through their respective linear layers
        #    Resulting Q, K, V shape: [batch_size, seq_len, d_model]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. Reshape Q, K, V to split into heads
        #    (Hint: .view(batch_size, -1, self.h, self.d_head).transpose(1, 2))
        #    Final shape: [batch_size, h, seq_len, d_head]
        Q = Q.view(batch_size, -1, self.h, self.d_head).transpose(1, 2)
        K = K.view(batch_size, -1, self.h, self.d_head).transpose(1, 2)
        V = V.view(batch_size, -1, self.h, self.d_head).transpose(1, 2)
        # 3. Apply scaled_dot_product_attention
        #    x, attention_weights = self.scaled_dot_product_attention(...)
        x, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        # 4. Concatenate heads and "un-reshape"
        #    (Hint: x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model))
        #    Final shape: [batch_size, seq_len_q, d_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5. Pass the final tensor through the output linear layer W_o
        x = self.W_o(x)

        # 6. Return the final tensor
        return x
        # --- END YOUR CODE ---

        # Placeholder
        # return torch.zeros_like(query)

class PositionWiseFeedForward(nn.Module):
    """
    Implements the Position-wise Feed-Forward Network (FFN).
    This is applied to each position (token) independently.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, d_model]
        x = self.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        # output shape: [batch_size, seq_len, d_model]
        return x

class ResidualConnection(nn.Module):
    """
    A helper class for the residual connection + layer norm.
    """
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        """
        Apply the residual connection to any sublayer.
        
        Args:
            x (torch.Tensor): The input to the sublayer.
            sublayer (nn.Module): The sublayer to apply (e.g., MHA or FFN).
        
        Returns:
            torch.Tensor: The output of LayerNorm(x + Dropout(Sublayer(x)))
        """
        # The paper applies norm *after* the addition, but it's now common
        # to apply it *before* (Pre-Norm), which is more stable.
        # Let's stick to the original paper's "Post-Norm" for now.
        return x + self.dropout(sublayer(self.norm(x))) # This is Pre-Norm
        # return self.norm(x + self.dropout(sublayer(x))) # This is Post-Norm
    

class EncoderLayer(nn.Module):
    """
    One layer of the Encoder.
    Contains two sub-layers: Self-Attention and a Feed-Forward Network.
    Uses Pre-Norm architecture for stability.
    """
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # --- YOUR CODE HERE ---
        # 1. Instantiate self.self_attn (your MultiHeadAttention module)
        self.self_attn = MultiHeadAttention(d_model,h,dropout)
        # 2. Instantiate self.feed_forward (your PositionWiseFeedForward module)
        self.feed_forward = PositionWiseFeedForward(d_model,d_ff,dropout=dropout)
        # 3. Instantiate self.norm1 (nn.LayerNorm)
        self.norm1 = nn.LayerNorm(d_model)
        # 4. Instantiate self.norm2 (nn.LayerNorm)
        self.norm2 = nn.LayerNorm(d_model)
        # 5. Instantiate self.dropout1 (nn.Dropout)
        self.dropout1 = nn.Dropout(dropout)
        # 6. Instantiate self.dropout2 (nn.Dropout)
        self.dropout2 = nn.Dropout(dropout)

        # --- END YOUR CODE ---

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input, shape [batch_size, seq_len, d_model]
            mask (torch.Tensor): Source mask, shape [batch_size, 1, 1, seq_len]
        
        Returns:
            torch.Tensor: Output, shape [batch_size, seq_len, d_model]
        """
        
        # --- YOUR CODE HERE ---
        # Implement the Pre-Norm architecture
        
        # 1. First sub-layer: Self-Attention
        #    a. Normalize the input x
        residual = x
        x = self.norm1(x)
        #    b. Pass normalized x as Q, K, and V to self.self_attn
        attn_output = self.self_attn(x, x, x, mask)
        #    c. Apply dropout to the attention output
        attn_output = self.dropout1(attn_output)
        #    d. Add the residual (the original x)
        x = residual + attn_output
        # 2. Second sub-layer: Feed-Forward
        #    a. Normalize the output of the first sub-layer
        x_norm = self.norm2(x)
        #    b. Pass the normalized tensor to self.feed_forward
        ff_output = self.feed_forward(x_norm)
        #    c. Apply dropout to the FFN output
        ff_output = self.dropout2(ff_output)
        #    d. Add the residual (the output from the first sub-layer)
        x = residual + ff_output
        # 3. Return the final output
        return x

        # --- END YOUR CODE ---

        # Placeholder
        # return torch.zeros_like(x)

class DecoderLayer(nn.Module):
    """
    One layer of the Decoder.
    Contains three sub-layers: Masked Self-Attention, Cross-Attention, 
    and a Feed-Forward Network.
    Uses Pre-Norm architecture.
    """
    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # --- YOUR CODE HERE ---
        # 1. Instantiate self.self_attn (MHA for decoder's own tokens)
        self.self_attn = MultiHeadAttention(d_model,h,dropout)
        # 2. Instantiate self.cross_attn (MHA for decoder-encoder communication)
        self.cross_attn = MultiHeadAttention(d_model,h,dropout)
        # 3. Instantiate self.feed_forward (FFN)
        self.feed_forward = PositionWiseFeedForward(d_model,d_ff,dropout=dropout)
        # 4. Instantiate self.norm1, self.norm2, self.norm3 (LayerNorm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # 5. Instantiate self.dropout1, self.dropout2, self.dropout3 (Dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # --- END YOUR CODE ---

    def forward(self, 
                x: torch.Tensor, 
                encoder_output: torch.Tensor, 
                tgt_mask: torch.Tensor, 
                src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input from previous decoder layer, shape [batch_size, tgt_seq_len, d_model]
            encoder_output (torch.Tensor): Output from encoder stack, shape [batch_size, src_seq_len, d_model]
            tgt_mask (torch.Tensor): Target mask (look-ahead mask), shape [batch_size, 1, tgt_seq_len, tgt_seq_len]
            src_mask (torch.Tensor): Source mask (padding mask), shape [batch_size, 1, 1, src_seq_len]
        
        Returns:
            torch.Tensor: Output, shape [batch_size, tgt_seq_len, d_model]
        """
        
        # --- YOUR CODE HERE ---
        
        # 1. First sub-layer: Masked Self-Attention
        #    (Use the Pre-Norm pattern)
        #    Q, K, V are all from the *normalized* x.
        #    Use the tgt_mask.
        residual = x
        masked_attn_output= self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask)
        masked_attn_output = self.dropout1(masked_attn_output)
        x = residual + masked_attn_output
        
        # 2. Second sub-layer: Cross-Attention
        #    (Use the Pre-Norm pattern)
        #    Q is from the *normalized* output of the previous layer.
        #    K and V are from the *encoder_output* (this is key!)
        #    Use the src_mask.
        residual = x
        cross_attn_output = self.cross_attn(self.norm2(x), encoder_output, encoder_output, src_mask)
        cross_attn_output = self.dropout2(cross_attn_output)
        x = residual + cross_attn_output

        # 3. Third sub-layer: Feed-Forward
        #    (Use the Pre-Norm pattern)
        residual = x
        ff_output = self.feed_forward(self.norm3(x))
        ff_output = self.dropout3(ff_output)
        x = residual + ff_output
        # 4. Return the final output
        return x
        # --- END YOUR CODE ---

        # Placeholder
        # return torch.zeros_like(x)

# (Paste all your previous correct classes here: PositionalEncoding, 
#  MultiHeadAttention, PositionWiseFeedForward, EncoderLayer, DecoderLayer)

class Encoder(nn.Module):
    """
    A stack of N EncoderLayers.
    """
    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        # Use nn.ModuleList to hold N identical copies of the layer
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        # A final LayerNorm for the output
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Pass the input through each layer in sequence
        for layer in self.layers:
            x = layer(x, mask)
        # Normalize the final output
        return self.norm(x)

class Decoder(nn.Module):
    """
    A stack of N DecoderLayers.
    """
    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        # Use nn.ModuleList to hold N identical copies of the layer
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        # A final LayerNorm for the output
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, tgt_mask: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Pass the input through each layer in sequence
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        # Normalize the final output
        return self.norm(x)

class Transformer(nn.Module):
    """
    The full Transformer model.
    """
    def __init__(self, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int, 
                 d_model: int, 
                 N: int, 
                 h: int, 
                 d_ff: int, 
                 dropout: float = 0.1):
        super().__init__()
        
        # --- YOUR CODE HERE ---
        
        # 1. Instantiate a "base" EncoderLayer and DecoderLayer to be copied
        c = copy.deepcopy # A shorthand for readability
        attn = MultiHeadAttention(d_model, h, dropout)
        ff = PositionWiseFeedForward(d_model, d_ff, dropout)
        encoder_layer = EncoderLayer(d_model, h, d_ff, dropout)
        decoder_layer = DecoderLayer(d_model, h, d_ff, dropout)
        # 2. Instantiate self.encoder (the Encoder stack)
        self.encoder = Encoder(encoder_layer, N)
        
        # 3. Instantiate self.decoder (the Decoder stack)
        self.decoder = Decoder(decoder_layer, N)
        # 4. Instantiate self.src_embed (nn.Embedding + PositionalEncoding)
        #    (Hint: Use nn.Sequential)
        self.src_embed = nn.Sequential(nn.Embedding(src_vocab_size,d_model), PositionalEncoding(d_model, dropout))
        # 5. Instantiate self.tgt_embed (nn.Embedding + PositionalEncoding)
        #    (Hint: Use nn.Sequential)
        self.tgt_embed = nn.Sequential(nn.Embedding(tgt_vocab_size,d_model), PositionalEncoding(d_model, dropout))
        # 6. Instantiate self.output_projection (nn.Linear)
        #    (This is the final layer that maps to vocab size)
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # --- END YOUR CODE ---

    def forward(self, 
                src: torch.Tensor, 
                tgt: torch.Tensor, 
                src_mask: torch.Tensor, 
                tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src (torch.Tensor): Source sequence, shape [batch_size, src_seq_len]
            tgt (torch.Tensor): Target sequence, shape [batch_size, tgt_seq_len]
            src_mask (torch.Tensor): Source padding mask
            tgt_mask (torch.Tensor): Target look-ahead + padding mask
        
        Returns:
            torch.Tensor: Output logits, shape [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        
        # --- YOUR CODE HERE ---
        
        # 1. Encode the source sequence
        #    a. Pass src through self.src_embed
        embedded_src = self.src_embed(src)
        #    b. Pass the embedded src through self.encoder
        encoder_output = self.encoder(embedded_src, src_mask)
        # 2. Decode the target sequence
        #    a. Pass tgt through self.tgt_embed
        embedded_tgt = self.tgt_embed(tgt)
        #    b. Pass the embedded tgt *and* the encoder_output through self.decoder
        # decoder_output = self.decoder(embedded_tgt, encoder_output, tgt_mask)
        decoder_output = self.decoder(embedded_tgt, encoder_output, tgt_mask, src_mask)
        # 3. Project to the final vocabulary
        #    a. Pass the decoder_output through self.output_projection
        logits = self.output_projection(decoder_output)
    
        # 4. Return the logits
        return logits       
        # --- END YOUR CODE ---
        
def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int = 0):
    """
    Helper function to create all necessary masks.
    """
    # Get the device from the input tensor
    device = tgt.device
    
    # Source padding mask
    # Shape: [batch_size, 1, 1, src_seq_len]
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # Target padding mask
    # Shape: [batch_size, 1, 1, tgt_seq_len]
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)

    # Target look-ahead mask (subsequent mask)
    # Shape: [1, 1, tgt_seq_len, tgt_seq_len]
    tgt_len = tgt.size(1)
    
    # --- FIX IS HERE ---
    # Create the look_ahead_mask directly on the correct device
    look_ahead_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
    look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0)

    # Combine target padding mask and look-ahead mask
    # Shape: [batch_size, 1, tgt_seq_len, tgt_seq_len]
    tgt_mask = tgt_pad_mask & look_ahead_mask
    
    # No need for .to(device) at the end, as all tensors are now on the correct device
    return src_mask, tgt_mask




class DummyTranslationDataset(Dataset):
    def __init__(self, num_samples, src_vocab_size, tgt_vocab_size, max_seq_len):
        self.num_samples = num_samples
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_seq_len = max_seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq_len = torch.randint(10, self.max_seq_len + 1, (1,)).item()
        src = torch.randint(1, self.src_vocab_size, (seq_len,))
        tgt = torch.randint(1, self.tgt_vocab_size, (seq_len,))
        
        # Create a "dummy" target input and output (shifted by one)
        # We'll use 0 as PAD, 1 as BOS, 2 as EOS
        
        # Example: src = [5, 8, 3]
        #          tgt = [7, 4, 9]
        # We need:
        #   src_input = [5, 8, 3]
        #   tgt_input = [1, 7, 4, 9] (starts with BOS)
        #   tgt_output = [7, 4, 9, 2] (ends with EOS)
        
        # This is too complex for a dummy. Let's simplify.
        # We'll just return two sequences of the same length.
        src = torch.randint(1, self.src_vocab_size, (self.max_seq_len,))
        tgt = torch.randint(1, self.tgt_vocab_size, (self.max_seq_len,))
        
        # The model expects tgt_input (e.g., [BOS, w1, w2]) 
        # and tgt_output (e.g., [w1, w2, EOS])
        tgt_input = tgt[:-1]
        tgt_output = tgt[1:]
        
        return src, tgt_input, tgt_output

def collate_fn(batch):
    # This is a basic collate, in reality you'd pad to max len in batch
    src_batch, tgt_input_batch, tgt_output_batch = [], [], []
    for src, tgt_input, tgt_output in batch:
        src_batch.append(src)
        tgt_input_batch.append(tgt_input)
        tgt_output_batch.append(tgt_output)
    return torch.stack(src_batch), torch.stack(tgt_input_batch), torch.stack(tgt_output_batch)

# --- [DDP SETUP FUNCTIONS] ---

def setup_ddp():
    """
    Initializes the distributed process group.
    `torchrun` handles setting the environment variables.
    """
    # DDP CRITICAL: 'nccl' is the fastest backend for NVIDIA GPUs.
    dist.init_process_group(backend="nccl")
    
    # DDP CRITICAL: Get the rank (process ID) and world size (total processes)
    # from the environment variables set by torchrun.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"]) # GPU ID for this process

    # DDP CRITICAL: Pin this process to a specific GPU.
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def cleanup_ddp():
    """Cleans up the process group."""
    dist.destroy_process_group()

# --- [MAIN TRAINING FUNCTION] ---
# This function will be run by *each* of your 2 processes.

def train(epochs: int, batch_size: int):
    
    # 1. Initialize DDP
    rank, world_size, device = setup_ddp()
    
    # DDP CRITICAL: Only the main process (rank 0) should print logs.
    is_main_process = (rank == 0)
    if is_main_process:
        print(f"Starting DDP training on {world_size} GPUs.")

    # 2. Model Hyperparameters (Example)
    SRC_VOCAB_SIZE = 5000
    TGT_VOCAB_SIZE = 5000
    D_MODEL = 512
    N_LAYERS = 6
    H_HEADS = 8
    D_FF = 2048
    DROPOUT = 0.1
    PAD_IDX = 0 # Assuming 0 is our padding index

    # 3. Create Model
    model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, D_MODEL, N_LAYERS, H_HEADS, D_FF, DROPOUT)
    # DDP CRITICAL: Move model to the correct GPU *before* wrapping with DDP.
    model.to(device)
    # DDP CRITICAL: Wrap the model. This handles gradient sync.
    model = DDP(model, device_ids=[device], output_device=device)

    # 4. Create Dataset and Sampler
    dataset = DummyTranslationDataset(num_samples=1000, src_vocab_size=SRC_VOCAB_SIZE, 
                                      tgt_vocab_size=TGT_VOCAB_SIZE, max_seq_len=30)
    
    # DDP CRITICAL: DistributedSampler ensures each GPU gets a unique
    # non-overlapping *shard* of the data.
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # num_workers > 0 speeds up data loading
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # 5. Optimizer and Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    # We ignore the loss on <pad> tokens
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 6. Training Loop
    model.train()
    for epoch in range(epochs):
        
        # DDP CRITICAL: We must set the epoch on the sampler.
        # This ensures the data is shuffled differently each epoch.
        sampler.set_epoch(epoch)
        
        total_loss = 0.0
        for i, (src, tgt_input, tgt_output) in enumerate(dataloader):
            
            # Move data to the correct GPU
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device) # Shape: [B, seq_len]

            # Create masks
            src_mask, tgt_mask = create_masks(src, tgt_input, PAD_IDX)
            
            # Forward pass
            logits = model(src, tgt_input, src_mask, tgt_mask) # Shape: [B, seq_len, vocab_size]
            
            # Calculate loss
            # We need to flatten the logits and targets for CrossEntropyLoss
            # Logits: [B * seq_len, vocab_size]
            # Target: [B * seq_len]
            loss = criterion(logits.view(-1, TGT_VOCAB_SIZE), tgt_output.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward() # DDP CRITICAL: This is where gradients are synced (all-reduced)
            optimizer.step()
            
            total_loss += loss.item()

        # DDP CRITICAL: Only log from the main process
        if is_main_process:
            print(f"Epoch: {epoch+1}, Loss: {total_loss / len(dataloader)}")

    # DDP CRITICAL: Only save the model from the main process
    if is_main_process:
        # We save the model's underlying state_dict
        torch.save(model.module.state_dict(), "transformer_ddp.pth")
        print("Training complete. Model saved.")

    # 7. Clean up DDP
    cleanup_ddp()

# --- [LAUNCHER] ---

if __name__ == "__main__":
    
    # These parameters can be passed via command line args
    EPOCHS = 10
    BATCH_SIZE_PER_GPU = 32
    
    # We don't need mp.spawn. `torchrun` will handle launching.
    # The `train` function will be called by each process.
    train(epochs=EPOCHS, batch_size=BATCH_SIZE_PER_GPU)