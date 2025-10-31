# Transformer from Scratch with Multi-GPU DDP Training

This repository contains a complete, from-scratch implementation of the "Attention Is All You Need" Transformer model in PyTorch.

The implementation focuses on clarity and faithfulness to the original paper's architecture, with one modern improvement: the use of a **Pre-Norm** (Layer Normalization before the sub-layer) architecture for more stable training.

The project also includes a production-grade, multi-GPU training script using **`DistributedDataParallel` (DDP)** and the `torchrun` launcher.

## Key Features

* **PositionalEncoding:** Sinusoidal positional encodings.
* **MultiHeadAttention:** A full, from-scratch MHA module.
* **Encoder/Decoder Layers:** Assembles MHA and FFNs into stable `EncoderLayer` and `DecoderLayer` blocks.
* **Masking:** Correctly implements both source-padding masks and target look-ahead masks.
* **Multi-GPU Training:** A complete DDP training loop demonstrating industry-standard distributed training.

## How to Run

This project is designed to be run on a multi-GPU machine using `torchrun`.

### Prerequisites

* PyTorch (`torch`)
* An NVIDIA GPU-enabled machine with the `nccl` backend (for DDP). This was tested on 2x RTX 3090s.

### Running on Multi-GPU

To launch the distributed training script, use the `torchrun` command. The following command will launch the training on 2 GPUs:

```bash
torchrun --standalone --nproc_per_node=2 attention_from_scratch.py