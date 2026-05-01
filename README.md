# GNR Project 2 — Visual MCQ Solver

An offline vision-language pipeline that reads PNG images of typeset
deep-learning multiple-choice questions and predicts the correct answer
(`A`, `B`, `C`, or `D`) for each one. Submission is a single
`submission.csv` file with the predicted option per image.

## Team

| Name | Roll Number |
|---|---|
| Divyaansh Narkhede | 24B0981 |
| Satwik Bhole | 24B2498 |

## Course

GNR638 — Machine Learning for Remote Sensing II, IIT Bombay (Spring 2026)

---

## Problem Recap

Each test sample is a PNG of a deep-learning MCQ — typeset text, formulas,
and code blocks on a clean white background. Exactly four options labelled
`A`–`D`, one correct. The system must run **fully offline** at inference
time and produce `submission.csv` mapping every image to an integer in
`{1=A, 2=B, 3=C, 4=D, 5=skip}`.

Constraints:
- Inference runtime budget: **< 1 hour** for up to 50 images
- Hardware: single NVIDIA L40s, 48 GB VRAM, CUDA 12.6, 16 GB system RAM
- No internet during inference; weights must be cached locally during setup
- Conda env name must be `gnr_project_env`, Python 3.11

---

## Approach

### Model: Qwen2.5-VL-7B-Instruct

We use [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct),
a 7B-parameter vision-language model from Alibaba's Qwen team.

Why this model:
- **Strong document-VQA performance** — Qwen2.5-VL is explicitly trained on
  document understanding, OCR, and image-grounded reasoning, which fits the
  typeset-MCQ task well.
- **LaTeX & code awareness** — handles inline math notation (formulas like
  `floor((W - K + 2P)/S) + 1`) and PyTorch code snippets that appear in the
  question options.
- **Memory headroom** — bf16 weights are ~15 GB, leaving plenty of the L40s'
  48 GB VRAM free for activations and the vision encoder.
- **Single-shot inference** — one forward pass per image; no retrieval, no
  fine-tuning, no extra OCR stage.

We considered the larger 32B AWQ variant for better accuracy on math-heavy
questions, but the 7B model gave reliable answers on our hand-built sanity
set (20 questions covering CNN shape propagation, PyTorch API, dropout,
attention, etc.) within a comfortable time budget.

### Inference strategy

1. **Chain-of-thought prompting.** A single deterministic system prompt
   instructs the model to:
   - read the question and all four options carefully
   - reason step-by-step (showing arithmetic for computational questions)
   - terminate with a strictly-formatted final line: `FINAL: <A|B|C|D|SKIP>`
2. **Greedy decoding.** `do_sample=False` makes outputs reproducible.
3. **Layered regex parser.** First match the canonical `FINAL: X` line; on
   failure fall back to common phrasings (`the answer is C`, `option (B) is
   correct`, etc.); on total failure default to `SKIP` (5).
4. **Skip-on-failure.** Any exception (OOM, malformed image, parse miss)
   writes `SKIP=5` rather than a hallucinated option.

### Vision encoder cap

Qwen2.5-VL processes images at any resolution. Its default `max_pixels`
budget (~12.8 M pixels) blows up activation memory on smaller GPUs and is
unnecessary for typeset documents that are perfectly legible at ~1000×1000.
We pin `max_pixels = 1280 × 28 × 28 ≈ 1.0 M` and `min_pixels = 256 × 28 × 28
≈ 0.2 M` via the processor. This keeps the vision-tower attention buffer
~1.5 GB instead of the original ~22 GB and matches what the L40s budget
expects.

Both knobs are overridable through environment variables
(`GNR_MIN_PIXELS`, `GNR_MAX_PIXELS`) without code changes if the hidden
test set turns out to need higher resolution.

### Optional 4-bit quantisation (development only)

For testing on smaller GPUs (e.g. Kaggle's single 16 GB T4) the script
supports nf4 quantisation via bitsandbytes:

```bash
GNR_LOAD_IN_4BIT=1 python inference.py --test_dir <path>
```

This brings VRAM use to ~5 GB at a small accuracy cost. Disabled by
default; the L40s grading run uses bf16.

---

## Repository Layout

```
.
├── setup.bash          # one-shot env + deps + clone + weights download (run with internet)
├── inference.py        # offline inference script, --test_dir entry point
├── environment.yml     # conda env spec mirror of setup.bash
├── requirements.txt    # pip-only spec mirror of setup.bash
├── README.md           # this file
├── .gitignore          # excludes model_cache/, submission.csv, etc.
└── model_cache/        # populated by setup.bash; NOT committed (~15 GB)
```

`inference.py` and `setup.bash` always live at the same level so the
grader's `python inference.py --test_dir ...` resolves correctly without
any `cd`.

---

## Setup (internet required)

The setup script handles every step that needs the network — env
creation, dependency install, repo clone, and model weight download —
so that `inference.py` can later run fully offline.

```bash
bash setup.bash
```

What it does, in order:

1. **Locates conda** on the host (`command -v conda`, `~/miniconda3`,
   `~/anaconda3`, or `/opt/conda`) and sources the activation script.
2. **Creates** conda env `gnr_project_env` with Python 3.11 if missing.
3. **Activates** the env and upgrades `pip` and `wheel`.
4. **Installs PyTorch 2.6.0 + TorchVision 0.21.0** built for CUDA 12.6
   from the official PyTorch wheel index.
5. **Installs HF stack** — Transformers (≥4.49,<4.55), Accelerate,
   `qwen-vl-utils[decord]`, Pillow, `huggingface_hub`, and bitsandbytes.
6. **Clones this repo** (only if `inference.py` is not already present in
   the working directory) flat into the current directory — no nested
   subdir is created, so subsequent `python inference.py` works without
   any `cd`.
7. **Pre-downloads** Qwen2.5-VL-7B-Instruct (~15 GB) into
   `./model_cache/Qwen2.5-VL-7B-Instruct/` via
   `huggingface_hub.snapshot_download`.

The script uses `set -euo pipefail`, so any failure aborts immediately
with a non-zero exit code.

### Manual environment creation (alternative)

If you only want to recreate the Python environment without running the
weight download / clone steps:

```bash
conda env create -f environment.yml
conda activate gnr_project_env
```

Or with pip directly inside any Python 3.11 environment:

```bash
pip install -r requirements.txt
```

---

## Running Inference (no internet required)

After `setup.bash` finishes, the grader runs:

```bash
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
```

The `--test_dir` directory must contain:

```
<test_dir>/
├── images/
│   ├── image_1.png
│   ├── image_2.png
│   └── ...
├── test.csv               # one column: image_name (no extension)
└── sample_submission.csv  # ignored, only used by graders for format reference
```

`inference.py`:

1. Parses `--test_dir` via `argparse`.
2. Reads `test.csv` from `<test_dir>/test.csv` and collects every
   `image_name`.
3. Loads the model from `./model_cache/Qwen2.5-VL-7B-Instruct/` in bf16
   (with float16 fallback on non-bf16 GPUs).
4. For each image, builds a Qwen-VL chat message with the system prompt
   plus the image, runs greedy generation with `max_new_tokens=768`,
   trims the prompt tokens off the generation, and parses the final
   answer letter.
5. Writes `submission.csv` to the **current working directory** (not
   `--test_dir`), with columns `image_name,option`. The file is flushed
   after every row so partial results survive a crash.
6. Clears the CUDA cache between images to avoid fragmentation.

A `--debug` flag prints the full raw model response per image — useful
when iterating on the prompt.

### Expected runtime

| Hardware | Per-image | 50 images |
|---|---|---|
| L40s 48 GB (target) | ~15 s | ~13 min |
| 2× T4 16 GB (Kaggle dev) | ~38 s | ~32 min |

Both are well under the 1-hour grading limit.

### Output format

```
image_name,option
image_1,3
image_2,1
...
```

Where `option ∈ {1=A, 2=B, 3=C, 4=D, 5=skip}`. Skip is emitted only on
parse failure or when the model explicitly outputs `SKIP`.

---

## Grading Pipeline (for reference)

These commands are run automatically by the grader — every step has
been validated against this repo:

```bash
cd <unzipped_dir>                                    # Step 1: enter dir
bash setup.bash                                      # Step 2: setup (internet)
conda activate gnr_project_env                       # Step 3: activate env
python inference.py --test_dir <abs_test_path>       # Step 4: inference (no internet)
python <grading_script> --submission_file submission.csv   # Step 5: grade
conda remove --name gnr_project_env --all -y         # Step 6: cleanup
```

---

## Hardware & Software Targets

- **GPU**: NVIDIA L40s, 48 GB VRAM
- **CUDA**: 12.6
- **System RAM**: 16 GB
- **OS**: Linux (Ubuntu-based)
- **Python**: 3.11
- **PyTorch**: 2.6.0 + cu126
- **Transformers**: ≥4.49.0, <4.55

The pipeline was developed and end-to-end-tested on Kaggle (2× T4, 30 GB
combined VRAM) using the same code path; the L40s run is identical bar
the device-map placement, since the entire 7B model fits on a single
GPU.

---

## Failure-Mode Handling

| Scenario | Behaviour |
|---|---|
| Image file missing in `<test_dir>/images/` | Row written with `option=5` (SKIP), processing continues |
| CUDA OOM during generation | Caught, row written with `option=5`, CUDA cache cleared, processing continues |
| Model output cannot be parsed | `option=5` (SKIP) |
| `test.csv` missing | Script exits non-zero with explicit error message |
| Conda not on PATH during setup | `setup.bash` exits non-zero with explicit error message |

Every CSV row is flushed immediately after writing, so a mid-run crash
still leaves a partial valid `submission.csv`.

---

## Citations

- **Qwen2.5-VL Technical Report** — Bai et al., 2025
  https://github.com/QwenLM/Qwen2.5-VL
  (Model: `Qwen/Qwen2.5-VL-7B-Instruct` on HuggingFace Hub)
- **Hugging Face Transformers** — Wolf et al.
  https://github.com/huggingface/transformers
- **qwen-vl-utils** — official Qwen helper for vision-language input prep
  https://pypi.org/project/qwen-vl-utils/
- **PyTorch** — Paszke et al.
  https://pytorch.org/
- **bitsandbytes** — Dettmers et al., used optionally for nf4 quantisation
  https://github.com/TimDettmers/bitsandbytes
- **Hugging Face Hub** — used to download model weights during setup
  https://github.com/huggingface/huggingface_hub
