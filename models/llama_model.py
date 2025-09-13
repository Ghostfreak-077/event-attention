from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config
from huggingface_hub import login

cfg = Config()
login(token=cfg.HF_API)

model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name).to(cfg.device)
tokenizer = AutoTokenizer.from_pretrained(model_name)