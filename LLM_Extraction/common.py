import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

SYSTEM_PROMPT = (
    "Bạn là hệ thống trích xuất thông tin từ text OCR của CCCD, thẻ ngân hàng và bằng lái xe. "
    "Hãy trả về đúng schema, không giải thích, không thêm trường mới, "
    "và dùng 'Không rõ' nếu không suy ra được."
)

UNKNOWN = "Không rõ"

FIELDS = [
    "Loại giấy tờ",
    "Tên",
    "Số định danh",
    "Số thẻ",
    "Ngày hết hạn",
    "Ngân hàng",
    "Loại thẻ",
    "Ngày sinh",
    "Giới tính",
    "Quê quán",
    "Nơi thường trú",
    "Hạng",
]

DOC_TYPES = ["CCCD", "Thẻ ngân hàng", "GPLX"]

PROVINCES = [
    "Hà Nội", "Hải Phòng", "Quảng Ninh", "Bắc Ninh", "Hải Dương", "Hưng Yên",
    "Thái Bình", "Nam Định", "Ninh Bình", "Thanh Hóa", "Nghệ An", "Hà Tĩnh",
    "Quảng Bình", "Quảng Trị", "Thừa Thiên Huế", "Đà Nẵng", "Quảng Nam",
    "Quảng Ngãi", "Bình Định", "Phú Yên", "Khánh Hòa", "Ninh Thuận",
    "Bình Thuận", "TP Hồ Chí Minh", "Bình Dương", "Đồng Nai",
    "Bà Rịa - Vũng Tàu", "Long An", "Tiền Giang", "Bến Tre",
    "Vĩnh Long", "Trà Vinh", "An Giang", "Kiên Giang", "Cần Thơ",
    "Sóc Trăng", "Bạc Liêu", "Cà Mau", "Đắk Lắk", "Gia Lai", "Kon Tum", "Lâm Đồng"
]

BANKS = [
    "AGRIBANK", "VIETCOMBANK", "BIDV", "VIETINBANK", "TECHCOMBANK",
    "ACB", "MB BANK", "SACOMBANK", "TPBANK", "VPBANK", "SHB", "HDBANK"
]

CARD_TYPES = [
    "Visa", "Visa Classic", "Visa Platinum",
    "MasterCard", "MasterCard World", "MasterCard Gold",
    "JCB", "UnionPay"
]

GPLX_CLASSES = ["A1", "A2", "B1", "B2", "C", "D", "E", "FC", "FE"]

GENDERS = ["Nam", "Nữ"]

LAST_NAMES = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương"]
MIDDLE_NAMES = ["Văn", "Thị", "Đức", "Ngọc", "Quang", "Thành", "Minh", "Thanh", "Hữu", "Anh", "Khánh", "Gia", "Thùy", "Bảo", "Phương"]
GIVEN_NAMES = ["An", "Bình", "Chi", "Dũng", "Hà", "Hạnh", "Hải", "Hiếu", "Hoa", "Hùng", "Lan", "Linh", "Long", "Mai", "Minh", "My", "Nam", "Nga", "Ngọc", "Phúc", "Phong", "Quân", "Quỳnh", "Sơn", "Thảo", "Trang", "Tuấn", "Tú", "Vy"]

OCR_CONFUSION = {
    "0": ["O", "o"],
    "1": ["I", "l", "|"],
    "2": ["Z"],
    "3": ["B"],
    "4": ["A"],
    "5": ["S"],
    "6": ["G"],
    "7": ["T"],
    "8": ["B"],
    "9": ["g", "q"],
    "a": ["à", "á", "ạ", "â", "ă"],
    "e": ["è", "é", "ẹ", "ê"],
    "i": ["ì", "í", "ỉ"],
    "o": ["ò", "ó", "ọ", "ô", "ơ"],
    "u": ["ù", "ú", "ụ", "ư"],
    "d": ["đ"],
    "A": ["À", "Á", "Ạ", "Â", "Ă"],
    "E": ["È", "É", "Ẹ", "Ê"],
    "I": ["Ì", "Í", "Ĩ"],
    "O": ["Ò", "Ó", "Ọ", "Ô", "Ơ"],
    "U": ["Ù", "Ú", "Ụ", "Ư"],
    "D": ["Đ"],
}


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def apply_char_confusion(rng: random.Random, ch: str, p: float) -> str:
    if ch in OCR_CONFUSION and rng.random() < p:
        return rng.choice(OCR_CONFUSION[ch])
    if ch.lower() in OCR_CONFUSION and rng.random() < p * 0.75:
        rep = rng.choice(OCR_CONFUSION[ch.lower()])
        return rep.upper() if ch.isupper() else rep
    return ch


def random_date(rng: random.Random, year_start: int = 1970, year_end: int = 2006) -> str:
    day = rng.randint(1, 28)
    month = rng.randint(1, 12)
    year = rng.randint(year_start, year_end)
    return f"{day:02d}/{month:02d}/{year:04d}"


def random_expiry_mm_yy(rng: random.Random) -> str:
    month = rng.randint(1, 12)
    year = rng.randint(26, 35)
    return f"{month:02d}/{year:02d}"


def random_valid_doc_no(rng: random.Random, length: int) -> str:
    return "".join(rng.choices("0123456789", k=length))


def random_card_no(rng: random.Random) -> str:
    return " ".join("".join(rng.choices("0123456789", k=4)) for _ in range(4))


def random_name(rng: random.Random) -> str:
    last = rng.choice(LAST_NAMES)
    middle = rng.choice(MIDDLE_NAMES)
    given = rng.choice(GIVEN_NAMES)
    if rng.random() < 0.30:
        middle2 = rng.choice(MIDDLE_NAMES)
        return f"{last} {middle} {middle2} {given}"
    return f"{last} {middle} {given}"


def random_long_address(rng: random.Random) -> str:
    house_no = str(rng.randint(1, 999))
    street = rng.choice([
        "đường Lê Lợi", "đường Nguyễn Trãi", "đường Trần Hưng Đạo",
        "đường Hai Bà Trưng", "đường Phạm Văn Đồng", "đường Cầu Giấy",
        "đường Kim Mã", "đường Bạch Mai", "khu phố 1", "ấp 2"
    ])
    ward = rng.choice([
        "phường Dịch Vọng Hậu", "phường Cầu Giấy", "phường Trần Hưng Đạo",
        "xã An Bình", "xã Tân Hòa", "thị trấn Đông Anh"
    ])
    district = rng.choice([
        "quận Cầu Giấy", "quận Ba Đình", "quận Hai Bà Trưng",
        "huyện Gia Lâm", "huyện Đông Anh", "thành phố Nha Trang"
    ])
    province = rng.choice(PROVINCES)
    if rng.random() < 0.5:
        return f"{house_no} {street}, {ward}, {district}, {province}"
    return f"{ward}, {district}, {province}"


def make_surface_form(rng: random.Random, text: str) -> str:
    text = normalize_spaces(text)
    variants = [
        text,
        strip_accents(text),
        strip_accents(text).upper(),
        strip_accents(text).lower(),
        text.upper(),
        text.lower(),
    ]
    return rng.choice(variants)


def preserve_raw(text: str) -> str:
    return normalize_spaces(text)


def title_vn(text: str) -> str:
    return normalize_spaces(text)


def noise_clean_very_light(rng: random.Random, text: str) -> str:
    text = normalize_spaces(text)
    out = []
    for ch in text:
        if ch == "\n":
            out.append(ch)
            continue
        if ch == " ":
            out.append(" " if rng.random() > 0.03 else "  ")
            continue
        out.append(apply_char_confusion(rng, ch, p=0.03))
    noisy = "".join(out)
    if rng.random() < 0.12:
        noisy = strip_accents(noisy)
    if rng.random() < 0.10:
        noisy = noisy.lower()
    return noisy.strip()


def noise_medium(rng: random.Random, text: str) -> str:
    text = normalize_spaces(text)
    out = []
    for ch in text:
        if ch == "\n":
            out.append(ch)
            continue
        if ch == " ":
            out.append("  " if rng.random() < 0.10 else " ")
            continue
        out.append(apply_char_confusion(rng, ch, p=0.10))
        if rng.random() < 0.02:
            out.append(rng.choice(["|", "-", ".", ","]))
    noisy = "".join(out)
    if rng.random() < 0.50:
        noisy = strip_accents(noisy)
    if rng.random() < 0.25:
        noisy = noisy.upper()
    elif rng.random() < 0.25:
        noisy = noisy.lower()
    return noisy.strip()


def noise_hard(rng: random.Random, text: str) -> str:
    text = normalize_spaces(text)
    out = []
    for ch in text:
        if ch == "\n":
            out.append(ch)
            continue
        if ch == " ":
            out.append(rng.choice([" ", "  ", "\t"]) if rng.random() < 0.18 else " ")
            continue
        if rng.random() < 0.22:
            out.append(apply_char_confusion(rng, ch, p=1.0))
        else:
            out.append(ch)
        if rng.random() < 0.05:
            out.append(rng.choice(["|", "_", "-", ".", ","]))
        if rng.random() < 0.03:
            continue
    noisy = "".join(out)
    if rng.random() < 0.85:
        noisy = strip_accents(noisy)
    mode = rng.random()
    if mode < 0.20:
        noisy = noisy.upper()
    elif mode < 0.40:
        noisy = noisy.lower()
    elif mode < 0.55:
        noisy = noisy.title()
    noisy = re.sub(r"[ ]{3,}", "  ", noisy)
    return noisy.strip()


def apply_noise_by_stage(rng: random.Random, text: str, stage: str) -> str:
    if stage == "clean":
        return noise_clean_very_light(rng, text)
    if stage == "medium":
        return noise_medium(rng, text)
    return noise_hard(rng, text)


def gen_cccd(rng: random.Random) -> Tuple[List[str], Dict[str, str], str]:
    raw_name = random_name(rng)
    raw_name_surface = make_surface_form(rng, raw_name)

    dob = random_date(rng, 1970, 2006)
    gender = rng.choice(GENDERS)
    id_no = random_valid_doc_no(rng, 12)

    if rng.random() < 0.65:
        raw_qq = random_long_address(rng)
    else:
        raw_qq = rng.choice(PROVINCES)

    raw_qq_surface = make_surface_form(rng, raw_qq)

    raw_noi_tt = random_long_address(rng)
    raw_noi_tt_surface = make_surface_form(rng, raw_noi_tt)

    fields = {
        "Tên": raw_name_surface,
        "Số định danh": id_no,
        "Số thẻ": UNKNOWN,
        "Ngày hết hạn": UNKNOWN,
        "Ngân hàng": UNKNOWN,
        "Loại thẻ": UNKNOWN,
        "Ngày sinh": dob,
        "Giới tính": gender,
        "Quê quán": raw_qq_surface,
        "Nơi thường trú": raw_noi_tt_surface,
        "Hạng": UNKNOWN,
    }

    lines = [
        "can cuoc cong dan",
        f"so: {id_no}",
        f"ho va ten: {raw_name_surface}",
        f"ngay sinh: {dob}",
        f"gioi tinh: {gender.lower()}",
        f"que quan: {raw_qq_surface}",
        f"noi thuong tru: {raw_noi_tt_surface}",
    ]
    return lines, fields, "CCCD"


def gen_bank_card(rng: random.Random) -> Tuple[List[str], Dict[str, str], str]:
    raw_name = random_name(rng)
    raw_name_surface = make_surface_form(rng, raw_name)

    bank = rng.choice(BANKS)
    card_type = rng.choice(CARD_TYPES)
    expiry = random_expiry_mm_yy(rng)

    include_card_no = rng.random() < 0.55
    card_no = random_card_no(rng) if include_card_no else None

    fields = {
        "Tên": raw_name_surface,
        "Số định danh": UNKNOWN,
        "Số thẻ": card_no if card_no else UNKNOWN,
        "Ngày hết hạn": expiry,
        "Ngân hàng": bank,
        "Loại thẻ": card_type,
        "Ngày sinh": UNKNOWN,
        "Giới tính": UNKNOWN,
        "Quê quán": UNKNOWN,
        "Nơi thường trú": UNKNOWN,
        "Hạng": UNKNOWN,
    }

    lines = [
        raw_name_surface,
        f"valid thru {expiry}",
        bank,
        card_type,
    ]
    if include_card_no:
        lines.insert(1, f"card no: {card_no}")
    if rng.random() < 0.35:
        lines.append("debit card")
    return lines, fields, "Thẻ ngân hàng"


def gen_gplx(rng: random.Random) -> Tuple[List[str], Dict[str, str], str]:
    raw_name = random_name(rng)
    raw_name_surface = make_surface_form(rng, raw_name)

    dob = random_date(rng, 1960, 2006)
    expiry = random_date(rng, 2030, 2035)
    class_ = rng.choice(GPLX_CLASSES)
    license_no = random_valid_doc_no(rng, rng.choice([8, 9, 10, 11, 12]))
    gender = rng.choice(GENDERS)

    raw_qq = random_long_address(rng) if rng.random() < 0.55 else rng.choice(PROVINCES)
    raw_qq_surface = make_surface_form(rng, raw_qq)

    fields = {
        "Tên": raw_name_surface,
        "Số định danh": license_no,
        "Số thẻ": UNKNOWN,
        "Ngày hết hạn": expiry,
        "Ngân hàng": UNKNOWN,
        "Loại thẻ": UNKNOWN,
        "Ngày sinh": dob,
        "Giới tính": gender,
        "Quê quán": raw_qq_surface,
        "Nơi thường trú": UNKNOWN,
        "Hạng": class_,
    }

    lines = [
        "giay phep lai xe",
        f"so: {license_no}",
        f"ho ten: {raw_name_surface}",
        f"ngay sinh: {dob}",
        f"gioi tinh: {gender.lower()}",
        f"que quan: {raw_qq_surface}",
        f"hang: {class_}",
        f"ngay het han: {expiry}",
    ]
    if rng.random() < 0.25:
        lines.append("quoc tich: viet nam")
    return lines, fields, "GPLX"


def maybe_shuffle_lines(rng: random.Random, lines: List[str], stage: str) -> List[str]:
    if stage == "clean":
        if rng.random() < 0.10:
            lines = lines[:]
            rng.shuffle(lines)
        return lines
    if stage == "medium":
        if rng.random() < 0.25:
            lines = lines[:]
            rng.shuffle(lines)
        return lines
    if rng.random() < 0.4:
        lines = lines[:]
        rng.shuffle(lines)
    return lines


def maybe_add_extras(rng: random.Random, lines: List[str], stage: str) -> List[str]:
    extras_clean = ["id", "card", "valid thru"]
    extras_medium = ["id", "card", "member since", "signature"]
    extras_hard = ["id", "card", "member since", "signature", "front side", "viet nam", "ocr text"]

    if stage == "clean":
        if rng.random() < 0.10:
            lines.append(rng.choice(extras_clean))
        return lines

    if stage == "medium":
        if rng.random() < 0.25:
            lines.append(rng.choice(extras_medium))
        if rng.random() < 0.10:
            lines.insert(0, rng.choice(["====", "ocr text", "front side"]))
        return lines

    if rng.random() < 0.35:
        lines.append(rng.choice(extras_hard))
    if rng.random() < 0.20:
        lines.insert(0, rng.choice(["====", ">>>", "ocr text", "front side"]))
    return lines


def generate_raw_input(rng: random.Random, doc_type: str, stage: str) -> Tuple[str, Dict[str, str], str]:
    if doc_type == "CCCD":
        lines, fields, label_doc_type = gen_cccd(rng)
    elif doc_type == "Thẻ ngân hàng":
        lines, fields, label_doc_type = gen_bank_card(rng)
    else:
        lines, fields, label_doc_type = gen_gplx(rng)

    lines = maybe_shuffle_lines(rng, lines, stage)
    lines = maybe_add_extras(rng, lines, stage)
    noisy_lines = [apply_noise_by_stage(rng, x, stage) for x in lines]
    input_text = "\n".join(noisy_lines)
    return input_text, fields, label_doc_type


def build_output(doc_type: str, fields: Dict[str, str]) -> str:
    out = {
        "Loại giấy tờ": doc_type,
        "Tên": fields.get("Tên", UNKNOWN),
        "Số định danh": fields.get("Số định danh", UNKNOWN),
        "Số thẻ": fields.get("Số thẻ", UNKNOWN),
        "Ngày hết hạn": fields.get("Ngày hết hạn", UNKNOWN),
        "Ngân hàng": fields.get("Ngân hàng", UNKNOWN),
        "Loại thẻ": fields.get("Loại thẻ", UNKNOWN),
        "Ngày sinh": fields.get("Ngày sinh", UNKNOWN),
        "Giới tính": fields.get("Giới tính", UNKNOWN),
        "Quê quán": fields.get("Quê quán", UNKNOWN),
        "Nơi thường trú": fields.get("Nơi thường trú", UNKNOWN),
        "Hạng": fields.get("Hạng", UNKNOWN),
    }
    return "\n".join([f"{k}: {v}" for k, v in out.items()] + ["<END>"])


def make_sample(rng: random.Random, doc_type: str, stage: str) -> Dict:
    input_text, fields, label_doc_type = generate_raw_input(rng, doc_type, stage)
    output_text = build_output(label_doc_type, fields)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"INPUT:\n{input_text}\n"},
            {"role": "assistant", "content": output_text},
        ],
        "task": "ocr_entity_extraction",
        "doc_type": label_doc_type,
        "noise_stage": stage,
    }


def split_counts(n: int) -> Tuple[int, int, int]:
    n_clean = int(n * 0.70)
    n_medium = int(n * 0.20)
    n_hard = n - n_clean - n_medium
    return n_clean, n_medium, n_hard


def save_jsonl(samples: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def split_train_val_test(samples: List[Dict], train_ratio=0.9, val_ratio=0.05):
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]
    return train, val, test


def preview(samples: List[Dict], k=3):
    for i in range(min(k, len(samples))):
        s = samples[i]
        print(f"===== SAMPLE {i+1} =====")
        print("STAGE:", s["noise_stage"])
        print("DOC:", s["doc_type"])
        print("INPUT:")
        print(s["messages"][1]["content"])
        print("OUTPUT:")
        print(s["messages"][2]["content"])
        print()
