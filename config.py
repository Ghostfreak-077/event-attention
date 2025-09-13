import torch
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.HF_API = os.getenv('HF_API')