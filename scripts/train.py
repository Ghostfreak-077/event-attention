import torch
import torch.nn as nn
from config import cfg
from models.model import StreamingTransformer
from dataset.dataloader import loader
from scripts.pretrain import train_epoch

device = cfg.device

model = StreamingTransformer(
        num_classes=400,        # Kinetics-400
        frame_size=224,
        patch_size=16,
        embed_dim=512,
        depth=6,
        num_heads=8,
        k=5,                    # Top-5 attention
        max_frames=64
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)

if __name__ == "__main__":
    for epoch in range(5):
        train_epoch(model, loader, criterion, optimizer, epoch)
    
    torch.save(model.state_dict(), f"./trained_models/trained.pth")