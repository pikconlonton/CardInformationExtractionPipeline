import os
import re
from pathlib import Path

import modal
from huggingface_hub import upload_folder


from common import UNKNOWN, normalize_spaces, strip_accents

app = modal.App("ocr-cccd-demo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "unsloth",
        "torch",
        "huggingface_hub",
    )
    .add_local_file("common.py", remote_path="/root/common.py")
)

MODEL_VOLUME = modal.Volume.from_name("ocr-model", create_if_missing=True)
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/model"))
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 20,
    secrets=[modal.Secret.from_name("hf-secret")],
    volumes={str(MODEL_DIR): MODEL_VOLUME},
)
def upload_merged_model_to_hf():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    repo_id = "hellloooworlddd123/entityExtract_by_Qwen2.5_3B"

    adapter_dir = MODEL_DIR / "qwen_ocr_model"
    save_dir = MODEL_DIR / "merged_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    if not adapter_dir.exists():
        print(f"Không tìm thấy adapter model tại: {adapter_dir}")
        return

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("Thiếu biến môi trường HF_TOKEN")

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("Merging LoRA into base model...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {save_dir}")
    merged_model.save_pretrained(str(save_dir), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
    tokenizer.save_pretrained(str(save_dir))

    print("Uploading merged model to Hugging Face...")
    upload_folder(
        folder_path=str(save_dir),
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
    )

    print("Upload merged model thành công!")


@app.local_entrypoint()
def main():
    upload_merged_model_to_hf.remote()