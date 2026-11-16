# small-image-generator

How small (in GB on disk) can a prompt-to-image generator get, with cartoon style as the target? This repo is a minimal Python pipeline (prompt in, PNG out) plus a LoRA training script for trying that out hands-on. See [CLAUDE.md](CLAUDE.md) for the size breakdown and the list of ideas.

## Files

| File | Purpose |
|---|---|
| `generate.py` | CLI: prompt -> image, with a `--model` switch and optional `--lora` |
| `train_lora.py` | Trains a LoRA on `segmind/tiny-sd` (UNet attention layers only) |
| `pyproject.toml`, `uv.lock` | Dependencies, managed with [uv](https://docs.astral.sh/uv/) |

## Quick start

```bash
uv sync
uv run generate.py "a cartoon fox reading a book" --model bk-sdm-tiny --out fox.png
```

The first run downloads the model from huggingface.co into `~/.cache/huggingface/hub/`. The device is picked automatically (CUDA, then MPS, then CPU).

Options: `--model`, `--out`, `--steps`, `--seed`, `--size`, `--style` (suffix appended to the prompt, `''` to disable), `--lora repo_or_dir:file.safetensors`, `--lora-scale`.

## Models

First impressions from one prompt and one seed on an Apple-silicon Mac (MPS). Sizes are the Hugging Face cache size on disk.

| `--model` | Hugging Face repo | Cache size | Steps | Time | Notes |
|---|---|---|---|---|---|
| `bk-sdm-tiny` | `nota-ai/bk-sdm-tiny` | 1.06 GB (fp16) | 25 | ~10 s | Outlined cartoon look, some anatomy glitches |
| `tiny-sd` | `segmind/tiny-sd` | 1.06 GB | 25 | ~6 s | Cleanest flat-vector look, least detail |
| `sdxs` | `IDKiro/sdxs-512-dreamshaper` | 2.1 GB (fp32; ~0.9 GB expected in fp16, not verified) | 1 | ~2 s | Most detailed, painterly rather than flat cartoon |

The safety checker is disabled in `generate.py` (it alone is ~600 MB). Keep that in mind before exposing this publicly. `tiny-sd` ships pickle `.bin` weights, so diffusers loads it with unsafe serialization.

### Why off-the-shelf LoRAs don't work

SD 1.5 LoRAs, such as `artificialguybr/cutecartoon-redmond-1-5v-...`, fail to load on all three models with `size mismatch` errors. The distilled UNets have blocks removed and different channel widths, so a LoRA has to be trained on the small model itself. That is what `train_lora.py` does.

## Training a cartoon LoRA

Dataset: [`Norod78/cartoon-blip-captions`](https://huggingface.co/datasets/Norod78/cartoon-blip-captions), 3,141 images (512x512) with BLIP captions, 190 MB.

- License is CC BY-NC-SA 4.0, so non-commercial only.
- The images are mostly frames and renders of existing TV and film characters.
- The captions are short and often weak ("a toy story character"). If results look muddy, re-captioning is the next improvement, before training longer.

`train_lora.py` loads the base model in fp32, freezes it, and trains rank-16 LoRA layers on `to_q`, `to_k`, `to_v` and `to_out` (1.73M parameters, ~7 MB adapter). Each caption gets `, cartoon style` appended.

Options: `--base`, `--dataset`, `--out`, `--steps`, `--batch-size`, `--lr`, `--rank`, `--size`, `--seed`, `--save-every`.

On an Apple-silicon Mac (MPS) a step at batch size 4 takes about 1.9 s. A 20-step smoke test works there. Full runs belong on the GB10.

## Runbook: training on the GB10

The GB10 (DGX Spark-class) is aarch64 Linux with a Blackwell GPU. The CUDA steps below have not been run on that hardware yet, so check each one as you go.

### 1. Get the project onto the GB10

```bash
git clone https://github.com/therealremco/small-image-generator.git small-image-generator && cd small-image-generator
```

Or copy it with `rsync` and exclude the large or local-only folders:

```bash
rsync -av --exclude .venv --exclude '*.png' --exclude lora-test ./ gb10:small-image-generator/
```

You need `pyproject.toml`, `uv.lock`, `generate.py` and `train_lora.py`.

### 2. Install and check CUDA

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output is a version, `True` and the GPU name. If it prints `False`, or warns that the GPU's compute capability is unsupported, install a torch build with a newer CUDA version for aarch64 (for example from the PyTorch `cu128` or `cu130` wheel index), then rerun the check.

### 3. Smoke test (about a minute)

```bash
uv run train_lora.py --steps 20 --out lora-test
```

This downloads `segmind/tiny-sd` and the dataset on first use. The log should show `device: cuda`, `trainable params: 1.73M` and a loss line every 10 steps.

### 4. Full training run

```bash
uv run train_lora.py --steps 3000 --batch-size 8 --out lora-cartoon
```

- Checkpoints are saved every 500 steps to `lora-cartoon/step-500/`, `step-1000/` and so on. The final adapter is `lora-cartoon/pytorch_lora_weights.safetensors`.
- With spare GPU memory, try `--batch-size 16 --lr 2e-4`.
- To keep it running after you log out: `nohup uv run train_lora.py --steps 3000 --batch-size 8 --out lora-cartoon > train.log 2>&1 &`, then `tail -f train.log`.
- Watch the loss. It is noisy, so judge by the images, not the number.

### 5. Compare checkpoints

Use the same prompt and seed for the base model and each checkpoint:

```bash
uv run generate.py "a cute fox reading a book" --model tiny-sd --style "cartoon style" --out base.png
for s in 500 1000 2000; do
  uv run generate.py "a cute fox reading a book" --model tiny-sd --style "cartoon style" \
    --lora lora-cartoon/step-$s:pytorch_lora_weights.safetensors --out step-$s.png
done
uv run generate.py "a cute fox reading a book" --model tiny-sd --style "cartoon style" \
  --lora lora-cartoon:pytorch_lora_weights.safetensors --out final.png
```

If later checkpoints look worse or start reproducing specific show characters, use an earlier one.

### 6. Copy the result back

```bash
scp gb10:small-image-generator/lora-cartoon/pytorch_lora_weights.safetensors ./lora-cartoon/
```

## Where files live

- **Models and datasets** are in `~/.cache/huggingface/hub/` (default Hugging Face cache; the real files are in `blobs/`, with symlinks under `snapshots/`). Set `HF_HOME` to move it.
- **LoRA outputs** are in `--out` directories. `.gitignore` excludes `.venv/`, `*.png` and `lora-test/`.

## Next steps

1. Judge the trained LoRA against the base `tiny-sd` and pick a checkpoint.
2. Try the same training on `bk-sdm-tiny` or `sdxs`.
3. Shrink further: fp16 or quantized weights, a smaller text encoder (~246 MB in fp16 in all three models), or cached prompt embeddings.
4. Try cleaner or re-captioned data.
