# From-Scratch Speculative Decoding for LLM Inference

This project is a from-scratch implementation of the **speculative decoding** algorithm (also known as "assisted generation") in PyTorch. The goal is to accelerate the inference (generation) speed of a large, high-quality language model by using a smaller, faster "draft" model to propose candidate tokens.

This implementation demonstrates a core, cutting-edge technique used in modern LLM inference engines to reduce latency. It sits at the intersection of complex algorithms and high-performance systems engineering.

- **Target Model (Large):** `gpt2-medium`
- **Draft Model (Small):** `distilgpt2`

## Core Concept: How It Works

Standard autoregressive decoding is slow because it's sequential: you must run the entire large model to get just one token.

Speculative decoding parallelizes this process:

1.  **Draft:** The small, fast `draft_model` generates `k` candidate tokens autoregressively. This is very quick.
2.  **Verify:** The large, slow `target_model` is run **only once** on the original prompt *plus* all `k` draft tokens. This single pass produces `k+1` logit predictions in parallel.
3.  **Accept/Reject:** We compare the draft's "guesses" to the target's "answers" token by token.
    - If `draft_token[i] == target_token[i]`, we **accept** the draft token.
    - If `draft_token[i] != target_token[i]`, we **reject** it (and all subsequent drafts). We take the target model's "correct" token instead.
    - If all `k` tokens are accepted, we get a "bonus" token from the target model's `k+1`th prediction.

The speedup comes from accepting multiple tokens (e.g., 3-5) for the cost of only *one* `target_model` pass.

## 📈 Performance & Benchmarking

The choice of `k` (the number of draft tokens) is a critical trade-off.
-   If `k` is too small, the speedup is minimal.
-   If `k` is too large, the draft model makes more mistakes, and the time spent generating and rejecting tokens (wasted work) makes it *slower* than the baseline.

An experiment was run to find the optimal `k` for this model pair (`distilgpt2` -> `gpt2-medium`) for generating 100 tokens.

| Generation Method | `k` | Time (seconds) | Speedup vs. Baseline |
| :--- | :---: | :---: | :---: |
| Baseline (Autoregressive) | N/A | 1.56s | 0% |
| **Speculative Decoding** | **3** | **1.24s** | **~20.1%** |
| Speculative Decoding | 5 | 1.40s | ~10.2% |
| Speculative Decoding | 7 | 1.72s | -10.2% (Slowdown) |
| Speculative Decoding | 10 | 2.10s | -34.6% (Slowdown) |

### Conclusion

The data shows a clear performance curve with an optimal value at **`k=3`**, achieving a **~20% speedup**. Beyond this point, the draft model's diminishing accuracy leads to more rejections, and the overhead of the algorithm outweighs its benefits.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install torch transformers
    ```

2.  **Run the script:**
    ```bash
    python speculative.py
    ```