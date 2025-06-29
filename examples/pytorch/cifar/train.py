"""Pytorch training module for CIFAR10."""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: nn.Module,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    momentum: float,
) -> None:
    """Train model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        running_loss, running_acc, n_samples = 0, 0, 0

        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for inputs, labels in pbar:
                X = inputs.to(device, non_blocking=True)
                y = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                # Update metrics
                running_loss += loss.item() * y.size(0)
                running_acc += (logits.argmax(dim=1) == y).sum().item()
                n_samples += y.size(0)

                pbar.set_postfix(
                    {
                        "loss": f"{running_loss / n_samples:.4f}",
                        "acc": f"{running_acc / n_samples:.4f}",
                    }
                )
    print("Finished Training")


def test(
    model: nn.Module,
    dataloader: DataLoader,
    classes: tuple[str, ...],
) -> None:
    """Test model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct, total = 0, 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    with torch.no_grad():
        for inputs, labels in dataloader:
            X, y = inputs.to(device), labels.to(device)
            logits = model(X)
            _, predicted = torch.max(logits, 1)

            # Overall accuracy
            correct += (predicted == y).sum().item()
            total += y.size(0)

            # Per-class accuracy
            for i in range(y.size(0)):
                label = y[i].item()
                class_correct[label] += (predicted[i] == y[i]).item()
                class_total[label] += 1

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")
    print("\nAccuracy per class:")
    for i, class_name in enumerate(classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f"Accuracy of {class_name:5s} : {accuracy:.2f}%")
