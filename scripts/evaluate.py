import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from config import cfg
from dataset.dataloader import loader

device = cfg.device
model = torch.load("./trained_models/trained.pth", map_location=device)
model.eval()
num_cache_frames = 4  # cache first 3-4 frames

all_logits = []
all_labels = []

criterion = torch.nn.CrossEntropyLoss()

total_videos = 0
correct = 0
total_loss = 0.
total_time = 0.


pbar = tqdm(loader, desc=f'Val')
with torch.no_grad():
    for batch_idx, (frames, labels) in enumerate(pbar):
        # frames: (B, T, C, H, W), labels: (B,)
        B, T, C, H, W = frames.shape
        frames = frames.to(device)
        labels = labels.to(device)

        # print(B, T)

        # Initialize per-video states using index
        states = {i: None for i in range(B)}

        # Step 1: Cache first few frames
        for t in range(num_cache_frames):
            frame_batch = frames[:, t]  # (B, C, H, W)
            for i in range(B):
                _, states[i] = model.forward_streaming(frame_batch[i].unsqueeze(0), states[i])

        # Step 2: Evaluate using only the next single frame
        t_next = num_cache_frames
        if t_next < T:  # make sure video is long enough
            frame_batch = frames[:, t_next]
            start_time = time.time()
            batch_logits = []
            for i in range(B):
                logits, states[i] = model.forward_streaming(frame_batch[i].unsqueeze(0), states[i])
                batch_logits.append(logits)
            end_time = time.time()

            batch_logits = torch.cat(batch_logits, dim=0)  # (B, num_classes)
            loss = criterion(batch_logits, labels)
            total_loss += loss.item() * B

            preds = batch_logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total_videos += B

            total_time += (end_time - start_time)

            all_logits.append(batch_logits.cpu())
            all_labels.append(labels.cpu())
            pbar.set_postfix({
                'loss': total_loss / total_videos,
                'acc': correct / total_videos,
                'time': total_time / total_videos
            })

# Metrics
avg_loss = total_loss / total_videos
accuracy = correct / total_videos
avg_time_per_video = total_time / total_videos

print(f"\nStreaming Eval Results:")
print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%, Inference Time per Video: {avg_time_per_video:.4f}s")
