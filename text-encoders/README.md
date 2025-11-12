
# The Evolution of Text Embeddings: A Benchmark

This repository contains a simple experiment to benchmark the performance of different text embedding models on a standard clustering task. The goal is to demonstrate the performance gap between classic context-free models, adapted Large Language Models (LLMs), and current State-of-the-Art (SOTA) encoders.

The clustering task is performed on the **20 Newsgroups** dataset. Performance is measured using the **V-Measure** score (higher is better).

## 📊 Final Results

Our experiments show a clear and dramatic improvement with each generation of embedding technology.

| Model | Model Type | V-Measure Score |
| :--- | :--- | :--- |
| **Avg. Word2Vec** | Classic (Context-Free) | **18.21** |
| **LLM2Vec-Mistral-7B** | 2024 Adapted LLM | **30.26** |
| **Qwen3-Embedding-8B** | 2025 SOTA (Trained LLM) | **61.55** |

  - **LLM2Vec** shows a **66% improvement** over the classic Word2Vec baseline.
  - **Qwen3** shows a **103% improvement** over the LLM2Vec model.

## 🚀 Running the Experiments

The experiments are split into two files.

### 1\. Baseline (Avg. Word2Vec)

This script (`a.py`) runs the Word2Vec baseline and compares it to the score reported in the LLM2Vec paper.

**Setup:**

```bash
pip install scikit-learn gensim numpy
```

**Run:**

```bash
python a.py
```

### 2\. SOTA (Qwen3-Embedding-8B)

This script (`b.py`) runs the SOTA Qwen3 model on the same task.

**Setup:**

```bash
# This requires a modern Python (3.10+)
pip install scikit-learn gensim numpy
pip install --upgrade transformers sentence-transformers
pip install torch bitsandbytes
```

**Run:**

```bash
python b.py
```

**Note:** This script requires a CUDA-enabled GPU. On a 24GB RTX 3090, it uses 4-bit quantization and takes approximately 20-25 minutes to encode the dataset.

