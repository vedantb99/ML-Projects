import torch
import torch.nn as nn
import torch.optim as optim

from src.dataloader import get_cifar10_loaders
from src.model import get_resnet18_model

# ----------------------------
# 1. Training function
# ----------------------------
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  # set model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


# ----------------------------
# 2. Validation function
# ----------------------------
def validate(model, test_loader, criterion, device):
    model.eval()  # set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # no gradient tracking for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Compute accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(test_loader.dataset)
    val_accuracy = 100.0 * correct / total
    return val_loss, val_accuracy


# ----------------------------
# 3. Main training loop
# ----------------------------
def main():
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    # Load model
    model = get_resnet18_model(num_classes=10, pretrained=False)
    model = model.to(device)
    num_epochs = 100
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, test_loader, criterion, device)
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_accuracy:.2f}%")
        print("-" * 40)

    print("✅ Training complete!")

    # Optionally save model
    torch.save(model.state_dict(), "resnet18_cifar10.pth")
    print("Model saved as resnet18_cifar10.pth")


if __name__ == "__main__":
    main()
