# High-Performance Multi-GPU Image Classification

This project implements a production-grade, multi-GPU training pipeline for an image classifier. Starting from a single-GPU baseline, the project is systematically scaled and optimized to run on multi-GPU hardware using **PyTorch's `DistributedDataParallel (DDP)`**.

The core of this project is not just to build a classifier, but to explore the engineering challenges of scaling deep learning training, including advanced optimization for large-batch training.

The model is a **ResNet-18** trained on the **CIFAR-10** dataset, achieving a final validation accuracy of **87.38%**.

## Core Concepts & Techniques

This project demonstrates mastery of the following critical ML engineering concepts:

  * **Model Architecture:** Implementing a **ResNet-18** CNN for image classification.
  * **Baseline Training:** Building a robust single-GPU training and validation script with `torchvision` data augmentation.
  * **Advanced Optimization:**
      * **Regularization:** Using **Weight Decay** (L2) to combat overfitting.
      * **LR Scheduling:** Implementing a **Cosine Annealing** learning rate schedule to find a sharp loss minimum.
  * **Distributed Training:**
      * Refactoring a single-GPU script to a high-performance multi-process pipeline using `DistributedDataParallel (DDP)`.
      * Using `torch.multiprocessing.spawn` to launch processes and `torch.distributed` to manage the process group.
      * Sharding data across GPUs using the `DistributedSampler`.
  * **Large-Batch Optimization:**
      * Applying the **Linear Scaling Rule** to adjust the learning rate based on the effective batch size.
      * Implementing a **Learning Rate Warmup** phase (`LinearLR`) chained with a `CosineAnnealingLR` (`SequentialLR`) to ensure stable convergence at high learning rates.

## Final Results

  * **Architecture:** ResNet-18
  * **Dataset:** CIFAR-10
  * **Final Validation Accuracy:** 86.45%
  * **Optimizer:** Adam
  * **Regularization:** Weight Decay (`1e-4`)
  * **Schedule:** 10-epoch linear warmup followed by 90-epoch cosine annealing.
  * **Environment:** 2 x NVIDIA RTX 3090 (DDP)

## How to Run

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-project-folder>
    ```

2.  **Create a virtual environment and install dependencies:**
    *(First, make sure you've created a `requirements.txt` file with `pip freeze > requirements.txt`)*

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Run the single-GPU baseline (100 epochs):**

    ```bash
    python train.py
    ```

4.  **Run the optimized multi-GPU version (100 epochs):**
    *(This script is hard-coded for `world_size=2` GPUs)*

    ```bash
    python train_ddp.py
    ```

## Project Structure

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── train.py                # Single-GPU training script
├── train_ddp.py            # Multi-GPU (DDP) training script
│
└── src/
    ├── __init__.py
    ├── dataloader.py       # Helper functions to get CIFAR-10 datasets
    └── model.py            # Helper function to get ResNet-18 model
```

