from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """加载预训练模型和 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer, device

def generate_text(model, tokenizer, device, prompt: str, max_length: int = 512):
    """使用模型生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=max_length, do_sample=False, temperature=0.)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    model_name = "/home/pretraining/klyang/mount_dir/eu_mount/hf_converted_model_edu"  # 你可以替换成任意 Hugging Face 上的模型
    model, tokenizer, device = load_model(model_name)
    
    prompt = "An example run script is shown below."
    generated_text = generate_text(model, tokenizer, device, prompt)
    
    print("Generated Text:")
    print(generated_text)