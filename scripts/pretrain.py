from config import cfg
import torch
from tqdm import tqdm

device = cfg.device

def train_epoch(model, dataloader, criterion, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (videos, labels) in enumerate(pbar):
        videos, labels = videos.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits, _ = model(videos)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc