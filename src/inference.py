import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 1. Load config
with open("configs/qlora_config.yaml") as f:
    cfg = yaml.safe_load(f)

# 2. Load base model + your trained adapter
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
base = AutoModelForCausalLM.from_pretrained(
    cfg["model_name"], quantization_config=bnb, device_map="auto"
)
model = PeftModel.from_pretrained(base, cfg["output_dir"])
model.eval()

# 3. Function to evaluate a pitch
def evaluate_pitch(pitch: str) -> str:
    prompt = f"### Startup Pitch:\n{pitch}\n\n### Evaluation:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
        )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split("### Evaluation:")[-1].strip()

# 4. Test it
if __name__ == "__main__":
    test_pitch = (
        "A B2B platform that uses AI to automatically generate legal "
        "contracts for freelancers. $49/mo, 200 customers, 2 founders."
    )
    print("PITCH:", test_pitch)
    print("\nEVALUATION:\n", evaluate_pitch(test_pitch))
