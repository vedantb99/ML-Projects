import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_cifar10_datasets(data_dir="./data"):
    """
    Returns the CIFAR-10 train and test datasets with standard transforms.
    This is the helper function for DDP.
    """
    
    # CIFAR-10 normalization constants (standard)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # --- Data augmentation for training ---
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # --- Test data transform ---
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # --- Create Datasets ---
    # Note: Use download=True if you don't have the data locally yet
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    
    return train_dataset, test_dataset


def get_cifar10_loaders(batch_size=128, num_workers=2, data_dir="./data"):
    """
    Returns standard train and test DataLoaders for CIFAR-10.
    This function is for single-GPU use.
    """
    
    train_dataset, test_dataset = get_cifar10_datasets(data_dir)

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader

if __name__ == "__main__":
    # You can add a small test here to ensure it works
    print("Testing dataloader creation...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=2)
    
    train_batch_img, train_batch_label = next(iter(train_loader))
    print(f"Train batch shape: {train_batch_img.shape}")
    
    test_batch_img, test_batch_label = next(iter(test_loader))
    print(f"Test batch shape: {test_batch_img.shape}")
    
    print("✅ Dataloaders seem to work.")