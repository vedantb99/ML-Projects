## Summary of Project Learnings

This project was a deep dive into the practical engineering required to build, optimize, and scale a modern computer vision model.

### 1. Baseline Implementation
You started by building a clean, production-style, single-GPU training pipeline.
* **Key Skills:** Using `torchvision` for data augmentation (`RandomCrop`, `Normalize`), loading a pretrained architecture (`ResNet-18`), modifying its classification head, and writing a clean training/validation loop with `Adam` and `CrossEntropyLoss`.

### 2. Optimization and Regularization
When the baseline model plateaued at 83% accuracy and showed signs of overfitting, you learned how to fix it.
* **Key Skills:**
    * **Weight Decay:** You applied L2 regularization directly in the optimizer (`weight_decay=1e-4`) to penalize large weights and force the model to learn more robust, generalizable features.
    * **LR Scheduling:** You moved beyond a fixed learning rate by implementing `CosineAnnealingLR`. This allowed the model to make large progress initially (high LR) and then fine-tune its weights as it got closer to the solution (low LR), pushing past the plateau.

### 3. Scaling with `DistributedDataParallel` (DDP)
To leverage all your hardware, you refactored the entire script from a single-GPU process to a high-performance, multi-GPU pipeline.
* **Key Skills:**
    * **DDP Architecture:** You learned the "multi-process" model: spawning one process per GPU, initializing a `ProcessGroup` (`dist.init_process_group`), and wrapping your model in `DDP`.
    * **Data Sharding:** You replaced the standard `DataLoader`'s shuffle with a `DistributedSampler` to ensure each GPU received a unique, non-overlapping slice of the data.
    * **Gradient Synchronization:** You learned that DDP automatically averages gradients from all processes (All-Reduce) after the backward pass, ensuring all models stay perfectly in sync.

### 4. Optimizing Large-Batch Training
Finally, you made your new DDP script robust and scalable by addressing the challenges of large-batch training.
* **Key Skills:**
    * **Linear Scaling Rule:** You learned that as effective batch size increases, the learning rate should also be scaled linearly (e.g., 2x batch size -> 2x LR).
    * **LR Warmup:** To prevent a high learning rate from causing the model to "explode" at the start, you implemented a warmup phase using `LinearLR`.
    * **Chained Schedulers:** You mastered `SequentialLR` to chain the warmup schedule and the main cosine annealing schedule together, creating a production-grade, scalable training process.

