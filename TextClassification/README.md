# Project 3: High-Performance Transformer Fine-Tuning with DDP

This project demonstrates the complete workflow for fine-tuning a modern Transformer model for a downstream NLP task. The project starts with a simple, single-GPU baseline and is then refactored into a high-performance, multi-GPU pipeline using **PyTorch's `DistributedDataParallel (DDP)`**.

The goal is to showcase **Transfer Learning**—the core paradigm of modern NLP—and the engineering skills required to scale this process efficiently across multiple GPUs.

  * **Model:** `distilbert-base-uncased` (a smaller, faster version of BERT).
  * **Dataset:** **SST-2 (Stanford Sentiment Treebank)** for 2-class sentiment analysis.
  * **Libraries:** Hugging Face `transformers`, `datasets`, and `PyTorch`.

## Core Concepts & Techniques

This project demonstrates mastery of:

  * **Transfer Learning:** Fine-tuning a large, pre-trained language model for a specific task (Sequence Classification) in only 3 epochs.
  * **Hugging Face Ecosystem:** Using `AutoTokenizer` to prepare text data and `AutoModelForSequenceClassification` to load a pre-trained model.
  * **Data Processing:** Writing a `dataset.map()` function to apply tokenization (`truncation=True`, `padding="max_length"`) in a fast, batched way.
  * **Custom Training Loop:** Handling the dictionary-based inputs and outputs of Transformer models in a standard PyTorch training loop (`outputs = model(**batch)`).
  * **NLP Optimization:** Using the `AdamW` optimizer, which correctly implements weight decay for Transformers.
  * **DDP Refactoring:** Systematically converting the single-GPU script into a multi-process, multi-GPU DDP script, proving the DDP pattern is universal across both CV and NLP domains.

## Final Results

The baseline model achieves a peak validation accuracy of **90.94%** on the SST-2 validation set after just 3 epochs.

| Model | Optimizer | LR | Epochs | Validation Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| `distilbert-base-uncased` | AdamW | `2e-5` | 3 | **90.94%** |

## How to Run

1.  **Clone the repository and install dependencies:**

    ```bash
    git clone https://github.com/vedantb99/ML-Projects.git
    cd TextClassification
    pip install torch transformers datasets accelerate
    ```

2.  **Run the single-GPU baseline:**

    ```bash
    python train_nlp.py
    ```

3.  **Run the optimized multi-GPU (DDP) version:**
    *(This script is hard-coded for `world_size=2` GPUs)*

    ```bash
    python train_nlp_ddp.py
    ```

## Project Structure

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── train_nlp.py            # Single-GPU fine-tuning script
├── train_nlp_ddp.py        # Multi-GPU (DDP) fine-tuning script
```
