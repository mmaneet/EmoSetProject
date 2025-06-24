"""
diffusiondb_master_annotator.py
────────────────────────────────
End-to-end pipeline that:

1. Downloads a chosen subset of DiffusionDB.
2. Saves every image to a local directory (progress bar).
3. Runs an image-emotion classifier on those images (progress bar).
4. Runs a GoEmotions-based text classifier on every prompt (prints subprocess output).
5. Writes a CSV with:
      path, prompt,
      img_amusement … img_sadness,
      text_amusement … text_sadness
"""

import csv
import json
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from datasets import load_dataset
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
)

# ─────────────────────────── USER CONFIG ──────────────────────────── #

SUBSET = "2m_random_1k"                   # DiffusionDB subset
IMAGES_DIR = Path(f"diffusiondb_images_{SUBSET}")   # folder to save images (unique per subset)
IMAGE_MODEL_CKPT = "checkpoint-3640"      # fine-tuned image-emotion model

PROMPT_REPO = Path("Demux-MEmo-master")   # repo containing annotate.py
PROMPT_MODEL_CKPT = Path("goemotions")    # GoEmotions checkpoint
EMOTION_CONFIG = PROMPT_REPO / "emotion_configs/goemotions.json"

PROMPTS_CSV  = Path(f"temp_prompts_{SUBSET}.csv")   # temp CSV per subset
PROMPT_JSONL = Path(
    f"diffusiondb-{SUBSET}-prompt-emotion-distributions-FULL.jsonl"
)
OUTPUT_CSV   = Path(f"diffusiondb-{SUBSET}-master-annotations.csv")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ─────────────────────────── LABEL SETS ──────────────────────────── #

IMG_IDX2LABEL = {
    0: "amusement",
    1: "awe",
    2: "contentment",
    3: "excitement",
    4: "anger",
    5: "disgust",
    6: "fear",
    7: "sadness",
}

TEXT_TARGET_EMOTIONS = [
    "amusement",
    "surprise",
    "joy",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
]

# ──────────────────────────── UTILITIES ───────────────────────────── #


def timestamp(msg: str):
    """Print a timestamped message to the console."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────── functions ────────────────────────────── #


def save_images(dataset, images_path: Path):
    """
    Save each PIL image from the HuggingFace dataset to disk.

    Returns two lists (file_paths, prompts) preserving original order.
    """
    images_path.mkdir(exist_ok=True)
    file_paths, prompts = [], []

    tbar = tqdm(total=len(dataset), desc="Saving images", leave=False)
    for idx, example in enumerate(dataset):
        img = example["image"]
        path = images_path / f"{idx}.png"
        img.save(path)
        file_paths.append(str(path))
        prompts.append(example["prompt"])
        tbar.update(1)
    tbar.close()
    return file_paths, prompts


def predict_image_emotions(image_paths):
    """Run the image-emotion classifier over all saved images."""
    timestamp("Running image-based emotion classifier …")
    processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_CKPT, use_fast=True)
    model = (
        AutoModelForImageClassification.from_pretrained(IMAGE_MODEL_CKPT)
        .to(DEVICE)
        .eval()
    )

    img_probs = [[] for _ in range(8)]  # 8 classes

    def load_img(p: str):
        from PIL import Image

        return Image.open(p).convert("RGB")

    batch_size = 8
    dataloader = DataLoader(image_paths, batch_size=batch_size, shuffle=False)

    pbar = tqdm(total=len(image_paths), desc="Image batches", leave=False)
    with torch.no_grad():
        for batch_paths in dataloader:
            imgs = [load_img(p) for p in batch_paths]
            inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
            logits = model(**inputs).logits
            probs = softmax(logits, dim=-1).cpu()

            for row in probs:
                for cls_idx, prob in enumerate(row):
                    img_probs[cls_idx].append(prob.item())

            pbar.update(len(batch_paths))
    pbar.close()
    return img_probs


def run_prompt_classifier(prompts):
    """Call annotate.py and parse its JSONL output into a dict."""
    timestamp("Running prompt-level classifier (annotate.py) …")

    # 1️⃣  write prompts CSV (id, prompt)
    pd.DataFrame({"id": range(len(prompts)), "prompt": prompts}).to_csv(
        PROMPTS_CSV, index=False
    )

    # 2️⃣  call annotate.py (shows its stdout live)
    subprocess.run(
        [
            "python",
            "annotate.py",
            "--pretrained-folder",
            str(PROMPT_MODEL_CKPT.resolve()),
            "--emotion-config",
            str(EMOTION_CONFIG.resolve()),
            "--domain",
            "twitter",
            "--input-filename",
            str(PROMPTS_CSV.resolve()),
            "--input-format",
            "csv",
            "--out",
            str(PROMPT_JSONL.resolve()),
            "--device",
            DEVICE,
            "--text-column",
            "prompt",
            "--id-column",
            "id",
        ],
        cwd=str(PROMPT_REPO),
        check=True,
    )

    # 3️⃣  parse jsonl with a progress bar
    text_emotions = {e: [] for e in TEXT_TARGET_EMOTIONS}
    num_lines = len(prompts)
    with open(PROMPT_JSONL, "r", encoding="utf-8") as jf, tqdm(
        total=num_lines, desc="Parsing prompt JSONL", leave=False
    ) as pbar:
        for line in jf:
            data = json.loads(line)
            emo_dict = data["emotions"]
            for e in TEXT_TARGET_EMOTIONS:
                text_emotions[e].append(emo_dict.get(e, 0.0))
            pbar.update(1)

    return text_emotions


# ───────────────────────────── MAIN PIPELINE ──────────────────────── #


def main():
    timestamp(f"Loading DiffusionDB subset: {SUBSET}")
    dataset = load_dataset("poloclub/diffusiondb", name=SUBSET, split="train")

    # 1. save images + prompts
    file_paths, prompts = save_images(dataset, IMAGES_DIR)
    timestamp(f"Saved {len(file_paths)} images to {IMAGES_DIR.resolve()}")

    # 2. image-emotion predictions
    img_probs = predict_image_emotions(file_paths)

    # 3. prompt-emotion predictions
    text_probs = run_prompt_classifier(prompts)

    # 4. write combined CSV
    timestamp("Writing master CSV …")
    header = (
        ["path", "prompt"]
        + [f"img_{IMG_IDX2LABEL[i]}" for i in range(8)]
        + [f"text_{e}" for e in TEXT_TARGET_EMOTIONS]
    )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i in range(len(file_paths)):
            row = [
                file_paths[i],
                prompts[i],
                *[img_probs[c][i] for c in range(8)],
                *[text_probs[e][i] for e in TEXT_TARGET_EMOTIONS],
            ]
            writer.writerow(row)

    timestamp("✔ Pipeline complete!")
    print(f"   • Images directory: {IMAGES_DIR.resolve()}")
    print(f"   • Master CSV file : {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()