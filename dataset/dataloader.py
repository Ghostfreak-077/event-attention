import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import TimesformerModel, TimesformerConfig, TimesformerPreTrainedModel
import torchvision
import torchvision.transforms as T
from kagglehub import dataset_download
from dataset.k400 import CustomKineticsDataset

ds = dataset_download("ipythonx/k4testset")

BATCH_SIZE = 4
NUM_SAMPLES = 64

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

dataset = CustomKineticsDataset(root=ds, annotation_file=ds+'/kinetics400_val_list_videos.txt', frames_per_clip=8)

# pick a small subset for quick test
small_dataset = Subset(dataset, range(NUM_SAMPLES))
loader = DataLoader(small_dataset, batch_size=BATCH_SIZE, shuffle=False)