import os
from pathlib import Path

import modal

app = modal.App("ocr-cccd-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.1.0",
        "torchvision",
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "datasets",
        "transformers>=4.51.3,<5.0",
        "accelerate",
        "peft",
        "bitsandbytes",
        "trl",
    )
)

DATA_VOLUME = modal.Volume.from_name("ocr-data", create_if_missing=True)
MODEL_VOLUME = modal.Volume.from_name("ocr-model", create_if_missing=True)

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/model"))

MAX_SEQ_LENGTH = 1024
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 12,
    volumes={str(DATA_DIR): DATA_VOLUME, str(MODEL_DIR): MODEL_VOLUME},
)
def train_model(
    num_train_epochs: int = 2,
    batch_size: int = 2,
    grad_accum: int = 8,
    learning_rate: float = 1e-4,
    seed: int = 42,
    output_dir: str = "/model/qwen_ocr_model",
):
    from unsloth import FastLanguageModel
    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    train_path = DATA_DIR / "train.jsonl"
    val_path = DATA_DIR / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Không thấy {train_path}. Hãy chạy gendata trước.")
    if not val_path.exists():
        raise FileNotFoundError(f"Không thấy {val_path}. Hãy chạy gendata trước.")

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "val": str(val_path)}
    )

    def to_text(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=seed,
    )

    dataset = dataset.map(
        to_text,
        remove_columns=dataset["train"].column_names,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            warmup_steps=50,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_steps=500,
            output_dir=output_dir,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            seed=seed,
            report_to="none",
            eval_strategy="steps",
            eval_steps=500,
        ),
    )

    trainer.train()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    MODEL_VOLUME.commit()
    return {"saved_to": str(out_dir)}


@app.local_entrypoint()
def main(
    num_train_epochs: int = 2,
    batch_size: int = 2,
    grad_accum: int = 8,
    learning_rate: float = 1e-4,
    seed: int = 42,
    output_dir: str = "/model/qwen_ocr_model",
):
    result = train_model.remote(
        num_train_epochs=num_train_epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        seed=seed,
        output_dir=output_dir,
    )
    print(result)
