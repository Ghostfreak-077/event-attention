import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F_torch
import cv2


class CustomKineticsDataset(Dataset):
    def __init__(self, root, annotation_file, frames_per_clip=8, image_size=224,
                 normalize=True, extensions=(".mp4", ".avi")):
        self.root = root
        self.frames_per_clip = frames_per_clip
        self.image_size = image_size
        self.normalize = normalize
        self.extensions = extensions

        # txt file format: "video_id class_name"
        self.video_to_class = {}
        all_classes = set()
        with open(annotation_file, "r") as f:
            for line in f:
                vid, cls = line.split(" ")
                self.video_to_class[vid] = int(cls.strip())
                all_classes.add(cls)

        self.classes = sorted(all_classes)

        self.samples = []
        for root_dir, _, files in os.walk(root):
            for fname in files:
                if fname.lower().endswith(self.extensions[0]):
                    if fname in self.video_to_class:
                        cls_name = self.video_to_class[fname]
                        self.samples.append((os.path.join(root_dir, fname), cls_name))

        if len(self.samples) == 0:
            raise RuntimeError("No videos found. Check root path, annotation file, and extensions.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        # Read frames using cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1)  # (C, H, W)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"Failed to load {video_path}")

        video = torch.stack(frames)  # (T, C, H, W)

        # Sample frames uniformly
        T_total = video.shape[0]
        if T_total >= self.frames_per_clip:
            idxs = torch.linspace(0, T_total - 1, self.frames_per_clip).long()
            video = video[idxs]
        else:
            pad = self.frames_per_clip - T_total
            last = video[-1:].repeat(pad, 1, 1, 1)
            video = torch.cat([video, last], dim=0)

        # Resize + Normalize
        frames_resized = []
        for frame in video:
            frame = F_torch.resize(frame, [self.image_size, self.image_size])
            frame = frame.float() / 255.0
            if self.normalize:
                frame = F_torch.normalize(frame,
                                    mean=[0.45, 0.45, 0.45],
                                    std=[0.225, 0.225, 0.225])
            frames_resized.append(frame)

        video = torch.stack(frames_resized)
        return video, label
