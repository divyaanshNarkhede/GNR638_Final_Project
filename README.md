# GNR Project 2 — Visual MCQ Solver

Offline vision-language pipeline that reads PNG images of deep-learning
multiple-choice questions and outputs the correct option (1=A, 2=B, 3=C, 4=D)
or 5 (skip) into `submission.csv`.

## Approach
- **Model:** [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) — strong document/text-in-image understanding, fits in ~15 GB at fp16.
- **Strategy:** zero-shot with chain-of-thought prompting; deterministic
  decoding (`do_sample=False`); strict `FINAL: <A|B|C|D|SKIP>` answer line that
  is parsed with a layered regex.
- **Risk control:** parser defaults to `5` (skip) on any failure, since the
  scoring rubric penalises hallucinated values (-1) more than a skip (0).

## Setup (one command, internet required)
```bash
bash setup.bash
```
This creates the conda environment `gnr_project_env` (Python 3.11), installs
PyTorch (CUDA 12.6 wheels), Transformers, qwen-vl-utils, etc., clones this repo
into the current directory, and pre-downloads the model weights to
`./model_cache/Qwen2.5-VL-7B-Instruct/`.

> Before submitting, edit `REPO_URL` near the top of `setup.bash` to point at
> your public GitHub fork.

## Inference (offline)
```bash
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
```
Writes `submission.csv` (columns `image_name,option`) to the current working
directory. Up to 50 images run in well under the 1-hour budget on the target
hardware.

`--debug` prints the raw model response for each image — useful for prompt
iteration; remove for final runs.

## Hardware
- Target: NVIDIA L40s, 48 GB VRAM, CUDA 12.6.
- Tested on Kaggle 2× T4 (30 GB combined) using fp16 with `device_map="auto"`.

## Layout
```
setup.bash         # creates env, installs deps, downloads weights
inference.py       # argparse entry point
environment.yml    # conda env spec (mirror of setup.bash installs)
requirements.txt   # pip-only spec
model_cache/       # populated by setup.bash; not committed
```

## Citations
- Bai et al., *Qwen2.5-VL Technical Report*, 2025 — https://github.com/QwenLM/Qwen2.5-VL
- Hugging Face `transformers` — https://github.com/huggingface/transformers
- `qwen-vl-utils` — https://pypi.org/project/qwen-vl-utils/
