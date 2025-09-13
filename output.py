from models.llama_model import model, tokenizer
from config import Config

cfg = Config()

def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(cfg.device) for k, v in inputs.items()}
    outputs = model(**inputs)
    return tokenizer.decode(outputs.logits.argmax(dim=-1)[0], skip_special_tokens=True)

if __name__ == "__main__":
    input_prompt = input("Enter your prompt: ")
    generated = generate_text(input_prompt)
    print(generated)