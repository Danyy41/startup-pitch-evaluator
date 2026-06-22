import yaml
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Load config
with open("configs/qlora_config.yaml") as f:
    cfg = yaml.safe_load(f)

# 2. Log in (paste your HF token when prompted, or set HF_TOKEN env var)
login()

# 3. Load your trained adapter
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
base = AutoModelForCausalLM.from_pretrained(cfg["model_name"], device_map="auto")
model = PeftModel.from_pretrained(base, cfg["output_dir"])

# 4. Push adapter + tokenizer to the Hub
repo = cfg["hf_repo_id"]
model.push_to_hub(repo)
tokenizer.push_to_hub(repo)
print(f"✅ Pushed to https://huggingface.co/{repo}")
