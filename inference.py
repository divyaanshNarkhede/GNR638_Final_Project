#!/usr/bin/env python3
"""GNR Project 2 — Visual MCQ Solver.

Reads PNG images of deep-learning multiple-choice questions and writes
`submission.csv` (image_name, option) to the current working directory.

Scoring: +1 correct, -0.25 wrong, 0 if option == 5 (skip), -1 for any other
value. We therefore default to 5 whenever parsing fails or the model abstains.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

# Required for cuBLAS-side determinism when torch.use_deterministic_algorithms
# is enabled. Must be set before torch / cuBLAS initialises, hence before
# `import torch`.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# ---------- Config ----------
MODEL_DIR = os.environ.get(
    "GNR_MODEL_DIR", "./model_cache/Qwen2.5-VL-7B-Instruct"
)
MAX_NEW_TOKENS = 768
SKIP = 5
LETTER_TO_INT = {"A": 1, "B": 2, "C": 3, "D": 4}


# Qwen2.5-VL processes images dynamically at any resolution. The default
# max_pixels (~12.8M) is huge and blows up activation memory on T4-class
# GPUs. Typeset MCQ docs are readable at ~1000x1000 (1280*28*28 pixels);
# tune via env var if needed.
MIN_PIXELS = int(os.environ.get("GNR_MIN_PIXELS", 256 * 28 * 28))
MAX_PIXELS = int(os.environ.get("GNR_MAX_PIXELS", 1280 * 28 * 28))

# Set GNR_LOAD_IN_4BIT=1 to load the model with bitsandbytes nf4 quantisation
# (~4-5 GB). Useful on a single 16 GB T4. Not needed on L40s.
LOAD_IN_4BIT = os.environ.get("GNR_LOAD_IN_4BIT", "0") == "1"

PROMPT = (
    "You are an expert deep learning researcher with comprehensive knowledge "
    "of neural network architectures, PyTorch, NumPy, optimization, "
    "regularization, attention/transformers, and ML theory.\n\n"
    "The image shows a multiple-choice question about deep learning with "
    "EXACTLY four options labelled A, B, C, and D. Exactly one option is "
    "correct.\n\n"
    "Workflow:\n"
    "1. Read the entire question text from the image. Pay close attention to:\n"
    "   - Whether the question asks for the CORRECT option, the INCORRECT "
    "option, or which statement is NOT true / is FALSE / is an EXCEPTION.\n"
    "   - Exact numerical values (kernel size, stride, padding, dimensions, "
    "channels, learning rate, dropout p, etc.).\n"
    "   - The exact framework / API (PyTorch nn.Module, torch.nn.functional, "
    "NumPy, etc.).\n"
    "2. Read ALL FOUR options A, B, C, D carefully and verbatim from the "
    "image. Note subtle differences between similar-looking options (e.g. "
    "dim=0 vs dim=1, stride=1 vs stride=2, with vs without bias, "
    "log_softmax vs softmax, batch_first=True vs False).\n"
    "3. Reason step by step. For computational questions, write the formula "
    "and the arithmetic explicitly. Reference values:\n"
    "   - Conv2d output spatial size: floor((W - K + 2P) / S) + 1\n"
    "   - MaxPool2d output spatial size: floor((W - K) / S) + 1\n"
    "   - ConvTranspose2d output: (W - 1) * S - 2P + K\n"
    "   - Conv2d trainable params: out_ch * (in_ch * K * K + (1 if bias else 0))\n"
    "   - Linear params: in_features * out_features + (out_features if bias else 0)\n"
    "   - LSTM output shape (batch_first=True): (batch, seq_len, hidden_size)\n"
    "   - Embedding output: input_shape + (embedding_dim,)\n"
    "   - MultiheadAttention output: same shape as query\n"
    "4. Eliminate clearly wrong options first, then choose between the "
    "remaining ones. Re-check your arithmetic before committing.\n"
    "5. End your response with EXACTLY ONE final line, on its own line, "
    "with nothing after it, in this exact format:\n\n"
    "FINAL: X\n\n"
    "where X is a single letter: A, B, C, or D.\n\n"
    "Use FINAL: SKIP only as a true last resort if the image is unreadable. "
    "Otherwise ALWAYS commit to your best answer — an educated guess has "
    "positive expected value over skipping."
)


def parse_answer(text: str) -> int:
    """Map free-form model output to {1,2,3,4,5}. Defaults to 5 on failure."""
    m = re.search(
        r"FINAL\s*[:\-]\s*\*{0,2}\(?([A-D]|SKIP)\)?\*{0,2}",
        text,
        re.IGNORECASE,
    )
    if m:
        token = m.group(1).upper()
        return LETTER_TO_INT.get(token, SKIP)

    fallback_patterns = [
        r"answer\s+is\s*[:\-]?\s*\*{0,2}\(?([A-D])\)?",
        r"correct\s+(?:answer|option|choice)\s+is\s*[:\-]?\s*\*{0,2}\(?([A-D])\)?",
        r"option\s+\(?([A-D])\)?\s+is\s+correct",
        r"\boption\s*[:\-]\s*\(?([A-D])\)?",
    ]
    for pat in fallback_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return LETTER_TO_INT[m.group(1).upper()]

    return SKIP


def load_model():
    if not torch.cuda.is_available():
        print("[infer] WARNING: no CUDA device available; running on CPU will be very slow.", flush=True)
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }
    if LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # device_map="auto" handles placement; explicit dtype is ignored under bnb.
        load_kwargs.pop("torch_dtype", None)
        print("[infer] Loading model in 4-bit (nf4)", flush=True)

    print(
        f"[infer] Loading model from {MODEL_DIR} "
        f"(dtype={dtype}, max_pixels={MAX_PIXELS}, 4bit={LOAD_IN_4BIT})",
        flush=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, **load_kwargs
    )
    model.eval()
    # Force the slow image processor. The new "fast" Qwen2VLImageProcessor
    # default uses a different resampling kernel and produces subtly
    # different image tensors -- enough to flip borderline answers between
    # transformers patch versions. Pinning use_fast=False keeps decoding
    # reproducible across environments.
    processor = AutoProcessor.from_pretrained(
        MODEL_DIR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, use_fast=False
    )
    return model, processor


def predict_one(model, processor, image_path: Path) -> tuple[int, str]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    trimmed = gen[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return parse_answer(response), response


def _enable_determinism() -> None:
    """Best-effort bit-for-bit determinism across runs on the same GPU.

    Uses ``warn_only=True`` so any op without a deterministic kernel falls
    back gracefully instead of raising -- we never want a hard crash here,
    since on the auto-graded test system any failure scores 0.
    """
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def main() -> int:
    _enable_determinism()

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True, type=Path)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full model responses for each image.",
    )
    args = parser.parse_args()

    test_dir: Path = args.test_dir.resolve()
    test_csv = test_dir / "test.csv"
    image_dir = test_dir / "images"
    if not test_csv.is_file():
        print(f"[infer] ERROR: {test_csv} not found", file=sys.stderr)
        return 1

    image_names: list[str] = []
    with open(test_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        # Per TA: test.csv uses `id` as the column name (== image_name).
        # Fall back to `image_name` for compatibility with the older sample.
        if "id" in fieldnames:
            key = "id"
        elif "image_name" in fieldnames:
            key = "image_name"
        else:
            print(
                f"[infer] ERROR: {test_csv} must have an 'id' or 'image_name' column "
                f"(found {fieldnames})",
                file=sys.stderr,
            )
            return 1
        for row in reader:
            image_names.append(row[key].strip())
    print(f"[infer] {len(image_names)} test rows from {test_csv}", flush=True)

    model, processor = load_model()

    submission_path = Path.cwd() / "submission.csv"
    t0 = time.time()
    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Per TA: submission must be `id, image_name, option` with id == image_name.
        writer.writerow(["id", "image_name", "option"])
        for i, name in enumerate(image_names, 1):
            img_path = image_dir / f"{name}.png"
            if not img_path.is_file():
                print(f"[infer] [{i}/{len(image_names)}] {name}: image missing -> SKIP", flush=True)
                writer.writerow([name, name, SKIP])
                f.flush()
                continue
            try:
                ans, raw = predict_one(model, processor, img_path)
                msg = f"[infer] [{i}/{len(image_names)}] {name} -> {ans}"
                if args.debug:
                    msg += f"\n--- raw ---\n{raw}\n-----------"
                print(msg, flush=True)
            except Exception as e:
                print(f"[infer] [{i}/{len(image_names)}] {name} ERROR: {e!r} -> SKIP", flush=True)
                ans = SKIP
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            writer.writerow([name, name, ans])
            f.flush()

    print(f"[infer] Wrote {submission_path} ({time.time() - t0:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
