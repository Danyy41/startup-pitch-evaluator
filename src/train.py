import json
import yaml
import torch
import mlflow
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 1. Load config
with open("configs/qlora_config.yaml") as f:
    cfg = yaml.safe_load(f)

# 2. Format each example into instruction text
def format_example(ex):
    out = ex["output"]
    return {
        "text": (
            f"### Startup Pitch:\n{ex['input']}\n\n"
            f"### Evaluation:\n"
            f"Strengths: {out['strengths']}\n"
            f"Weaknesses: {out['weaknesses']}\n"
            f"Market Opportunity: {out['market_opportunity']}\n"
            f"Risk Score: {out['risk_score']}/10"
        )
    }

dataset = load_dataset(
    "json", data_files="data/processed/pitches.jsonl", split="train"
).map(format_example)

# 3. Load model in 4-bit (QLoRA)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    cfg["model_name"], quantization_config=bnb, device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# 4. Apply LoRA / PEFT
lora = LoraConfig(
    r=cfg["lora_r"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=cfg["lora_dropout"],
    target_modules=cfg["target_modules"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# 5. Train + track with MLflow
mlflow.set_experiment("startup-pitch-evaluator")
with mlflow.start_run():
    mlflow.log_params({
        "lora_r": cfg["lora_r"],
        "lr": cfg["learning_rate"],
        "epochs": cfg["num_epochs"],
    })

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=cfg["output_dir"],
            num_train_epochs=cfg["num_epochs"],
            per_device_train_batch_size=cfg["batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"],
            max_seq_length=cfg["max_seq_length"],
            logging_steps=1,
        ),
    )
    trainer.train()

    final_loss = trainer.state.log_history[-1].get("train_loss", 0)
    mlflow.log_metric("final_train_loss", final_loss)

# 6. Save the LoRA adapter
trainer.save_model(cfg["output_dir"])
print("✅ Training done. Adapter saved to", cfg["output_dir"])
