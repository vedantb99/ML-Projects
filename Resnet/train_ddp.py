import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from src.dataloader import get_cifar10_datasets # NOTE: Get datasets, not loaders
from src.model import get_resnet18_model
import os

# ----------------------------
# 1. DDP Setup Functions
# ----------------------------
def setup(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # Any free port
    # 'nccl' is the standard backend for NVIDIA GPUs
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Destroys the distributed process group."""
    dist.destroy_process_group()

# ----------------------------
# 2. Refactored Train/Val (mostly the same)
# ----------------------------
def train_one_epoch(model, train_loader, criterion, optimizer, device, rank, epoch):
    model.train()
    
    # CRITICAL: Set the epoch on the sampler.
    # This ensures proper shuffling across epochs.
    train_loader.sampler.set_epoch(epoch)
    
    running_loss = 0.0
    for images, labels in train_loader:
        # Move data to the correct device
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    
    # We only want to print from one process
    if rank == 0:
        print(f"  Train Loss: {epoch_loss:.4f}")
    
    return epoch_loss

def validate(model, test_loader, criterion, device, rank):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(test_loader.dataset)
    val_accuracy = 100.0 * correct / total
    
    # We only want to print from one process
    if rank == 0:
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_accuracy:.2f}%")
        
    return val_loss, val_accuracy

# ----------------------------
# 3. The new "main worker" function
# ----------------------------
def main_worker(rank, world_size, num_epochs, batch_size, scaled_lr, weight_decay, warmup_epochs):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    
    # Device is now the rank
    device = rank
    
    # Load datasets
    train_dataset, test_dataset = get_cifar10_datasets() # Assumes you have this helper
    
    # --- DDP CHANGES ---
    # 1. Create the DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    # Test sampler is optional, but good practice
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # 2. Create DataLoaders with the samplers
    # Note: shuffle must be False, the sampler handles it.
    # Note: batch_size is now PER-GPU
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, sampler=test_sampler
    )
    # --- END DDP CHANGES ---

    # Model setup
    model = get_resnet18_model(num_classes=10, pretrained=False).to(device)
    # --- DDP CHANGE ---
    # 3. Wrap the model in DDP
    model = DDP(model, device_ids=[rank])
    # --- END DDP CHANGE ---

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=scaled_lr, weight_decay=weight_decay)
    
    # 1. The warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6, # Start at a tiny LR
        end_factor=1.0,    # End at the full scaled_lr
        total_iters=warmup_epochs # Number of epochs to warmup
    )
    # 2. The main cosine scheduler
    # It will run for the *remaining* epochs
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=(num_epochs - warmup_epochs)
    )
    # 3. Chain them together
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs] # The epoch to switch from scheduler 1 to 2
    )
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print("-" * 40)
            
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, rank, epoch)
        val_loss, val_accuracy = validate(model, test_loader, criterion, device, rank)
        
        scheduler.step()

    # --- DDP CHANGE ---
    # 4. Save model correctly
    # Only the main process should save the model
    if rank == 0:
        # We save the model's 'module' state_dict, not the DDP wrapper
        torch.save(model.module.state_dict(), "resnet18_cifar10_ddp.pth")
        print("Model saved as resnet18_cifar10_ddp.pth")
    # --- END DDP CHANGE ---

    cleanup()

# ----------------------------
# 4. The main "launcher"
# ----------------------------
if __name__ == "__main__":
    
    world_size = 2  # Set to your number of GPUs
    num_epochs = 100
    batch_size = 128 # This will be 128 *per GPU*, so 256 effective batch size
    weight_decay = 1e-4
    base_lr = 1e-3
    warmup_epochs = 10 # Let's warmup for 10 epochs
    scaled_lr = base_lr * world_size
    args = (num_epochs, batch_size, scaled_lr, weight_decay, warmup_epochs) # Add new args    
    print("Spawning DDP processes...")
    mp.spawn(
        main_worker,
        args=(world_size, *args),
        nprocs=world_size,
        join=True
    )
    print("✅ DDP Training complete!")