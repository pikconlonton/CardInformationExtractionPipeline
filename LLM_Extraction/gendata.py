import argparse
import random
from pathlib import Path

import modal

from common import DOC_TYPES, make_sample, preview, save_jsonl, split_counts, split_train_val_test

app = modal.App("ocr-data-gen")
DATA_VOLUME = modal.Volume.from_name("ocr-data", create_if_missing=True)
DATA_DIR = Path("/data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_file("common.py", remote_path="/root/common.py")
)

@app.function(
    image=image,
    volumes={str(DATA_DIR): DATA_VOLUME},
)
def generate_dataset(n_samples: int, seed: int, out_dir: str = "/data") -> dict:
    rng = random.Random(seed)
    n_clean, n_medium, n_hard = split_counts(n_samples)

    samples = []
    doc_weights = [0.40, 0.35, 0.25]

    for _ in range(n_clean):
        doc_type = rng.choices(DOC_TYPES, weights=doc_weights, k=1)[0]
        samples.append(make_sample(rng, doc_type, "clean"))

    for _ in range(n_medium):
        doc_type = rng.choices(DOC_TYPES, weights=doc_weights, k=1)[0]
        samples.append(make_sample(rng, doc_type, "medium"))

    for _ in range(n_hard):
        doc_type = rng.choices(DOC_TYPES, weights=doc_weights, k=1)[0]
        samples.append(make_sample(rng, doc_type, "hard"))

    rng.shuffle(samples)

    train, val, test = split_train_val_test(samples, train_ratio=0.9, val_ratio=0.05)
    out_path = Path(out_dir)

    save_jsonl(train, out_path / "train.jsonl")
    save_jsonl(val, out_path / "val.jsonl")
    save_jsonl(test, out_path / "test.jsonl")

    meta = {
        "seed": seed,
        "n_samples": n_samples,
        "counts": {"clean": n_clean, "medium": n_medium, "hard": n_hard},
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
    }
    (out_path / "meta.json").write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    DATA_VOLUME.commit()
    print("Saved to:", out_path.resolve())
    print("Counts:", meta["counts"])
    print("Train/Val/Test:", len(train), len(val), len(test))
    print()
    preview(samples, k=3)
    return meta


@app.local_entrypoint()
def main(
    n_samples: int = 10000,
    seed: int = 42,
    out_dir: str = "/data",
):
    result = generate_dataset.remote(n_samples=n_samples, seed=seed, out_dir=out_dir)
    print(result)
