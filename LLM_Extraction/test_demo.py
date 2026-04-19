import os
import re
from pathlib import Path

import modal
from huggingface_hub import upload_file
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
MAX_SEQ_LENGTH = 1024


def parse_output_to_dict(text: str):
    result = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        result[normalize_spaces(k)] = normalize_spaces(v)
    return result


def extract_labeled_value(text: str, labels):
    lines = [normalize_spaces(x) for x in text.splitlines()]
    for line in lines:
        low = strip_accents(line).lower()
        for lab in labels:
            if low.startswith(lab):
                if ":" in line:
                    return normalize_spaces(line.split(":", 1)[1])
                return normalize_spaces(line[len(lab):])
    return None


def extract_bank_name(text: str):
    lines = [normalize_spaces(x) for x in text.splitlines() if normalize_spaces(x)]
    for line in lines:
        low = strip_accents(line).lower()
        if any(x in low for x in ["valid thru", "card no", "debit card", "visa", "mastercard", "jcb", "unionpay"]):
            continue
        if re.fullmatch(r"[0-9 ]{12,19}", line):
            continue
        if line.lower().startswith("input:"):
            continue
        return line
    return None


def postprocess_from_input(input_text: str, model_text: str) -> str:
    data = parse_output_to_dict(model_text)

    name = extract_labeled_value(input_text, ["ho va ten", "ho ten"])
    if name:
        data["Tên"] = name

    qq = extract_labeled_value(input_text, ["que quan"])
    if qq:
        data["Quê quán"] = qq

    ntt = extract_labeled_value(input_text, ["noi thuong tru"])
    if ntt:
        data["Nơi thường trú"] = ntt

    if data.get("Loại giấy tờ", "") == "Thẻ ngân hàng":
        bank_name = extract_bank_name(input_text)
        if bank_name:
            data["Tên"] = bank_name

    ordered = [
        ("Loại giấy tờ", data.get("Loại giấy tờ", UNKNOWN)),
        ("Tên", data.get("Tên", UNKNOWN)),
        ("Số định danh", data.get("Số định danh", UNKNOWN)),
        ("Số thẻ", data.get("Số thẻ", UNKNOWN)),
        ("Ngày hết hạn", data.get("Ngày hết hạn", UNKNOWN)),
        ("Ngân hàng", data.get("Ngân hàng", UNKNOWN)),
        ("Loại thẻ", data.get("Loại thẻ", UNKNOWN)),
        ("Ngày sinh", data.get("Ngày sinh", UNKNOWN)),
        ("Giới tính", data.get("Giới tính", UNKNOWN)),
        ("Quê quán", data.get("Quê quán", UNKNOWN)),
        ("Nơi thường trú", data.get("Nơi thường trú", UNKNOWN)),
        ("Hạng", data.get("Hạng", UNKNOWN)),
    ]
    return "\n".join([f"{k}: {v}" for k, v in ordered] + ["<END>"])

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 30,
    volumes={str(MODEL_DIR): MODEL_VOLUME},
)
def demo(sample_text: str) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = MODEL_DIR / "qwen_ocr_model"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Không thấy model ở {adapter_dir}. Hãy chạy train trước.")

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    system_prompt = """Bạn là hệ thống trích xuất thông tin từ text OCR của CCCD, thẻ ngân hàng và bằng lái xe.
    Hãy trả về đúng schema, không giải thích, không thêm trường mới,
    và dùng 'Không rõ' nếu không suy ra được.
BẮT BUỘC:
- Chỉ trả về đúng các trường sau, đúng thứ tự:
Loại giấy tờ
Tên
Số định danh
Số thẻ
Ngày hết hạn
Ngân hàng
Loại thẻ
Ngày sinh
Giới tính
Quê quán
Nơi thường trú
Hạng

- CHỈ được lấy thông tin có trong INPUT, không thêm, không bớt, không giải thích
- Nếu không có thông tin: ghi "Không rõ"
- KHÔNG được suy đoán hoặc tự sửa nội dung
- KHÔNG được tự sửa tên riêng
- KHÔNG được tự chuẩn hoá tên/địa chỉ thành một tên khác
- Kết thúc bằng <END>
"""

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=160,
        temperature=0.0,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.convert_tokens_to_ids("<END>"),
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    if "<END>" in result:
        result = result.split("<END>", 1)[0] + "<END>"
    else:
        result = result.strip() + "\n<END>"

    return postprocess_from_input(sample_text, result)


@app.local_entrypoint()
def main():
    samples = [
        """can cuoc cong dan
so: 079123456789
ho va ten: tran thi thu ha
ngay sinh: 15/03/1998
gioi tinh: nu
que quan: phuong dich vong hau, cau giay, ha noi
noi thuong tru: 123 duong abc, ha noi
ngay het han: 15/03/2033""",
        """NGUYEN VAN A
1234 5678 9012 3456
VALID THRU 08/27
Vietcombank
VISA Platinum""",
        """NGUYEN VAN A
1234 5678 9012 3456
VALID THRU 08/27
Vietcombank""",
        """giay phep lai xe
so: 123456789
ho ten: le minh hoang
ngay sinh: 20/10/1995
gioi tinh: nam
que quan: da nang
hang: b2
ngay het han: 20/10/2035""",
        """giay phep lai xe
so: 0123456789
ho ten: le minh hoang
ngay sinh: 20/10/1995
gioi tinh: nam
que quan: da nang
hang: b2
ngay het han: 20/10/2035""",
    ]

    for i, sample in enumerate(samples, 1):
        print(f"\n--- SAMPLE {i} ---")
        print(demo.remote(sample))
        print()
