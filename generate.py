"""Minimal prompt -> image pipeline.

    uv run generate.py "a cartoon fox reading a book"
    uv run generate.py "a cartoon fox" --model sdxs --out fox.png
"""
import argparse
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

# Candidate models on huggingface.co. All are SD 1.5-architecture, so one loader works.
# steps/guidance are each model's recommended settings.
MODELS = {
    "bk-sdm-tiny": dict(repo="nota-ai/bk-sdm-tiny", steps=25, guidance=7.5, variant="fp16"),
    "tiny-sd": dict(repo="segmind/tiny-sd", steps=25, guidance=7.5, variant=None),
    "sdxs": dict(repo="IDKiro/sdxs-512-dreamshaper", steps=1, guidance=0.0, variant=None),
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--model", choices=MODELS, default="bk-sdm-tiny")
    ap.add_argument("--out", type=Path, default=Path("out.png"))
    ap.add_argument("--steps", type=int, help="override the model's default step count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--style", default="cartoon style, flat colors, bold outlines",
                    help="suffix appended to the prompt ('' to disable)")
    ap.add_argument("--lora", help="LoRA as 'repo_id:weight_file.safetensors' on huggingface.co")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    args = ap.parse_args()

    cfg = MODELS[args.model]
    device = pick_device()
    dtype = torch.float16 if device != "cpu" else torch.float32

    t0 = time.time()
    kwargs = dict(torch_dtype=dtype, safety_checker=None)  # safety checker alone is ~600 MB
    if cfg["variant"]:
        kwargs["variant"] = cfg["variant"]  # download only the fp16 weights
    pipe = StableDiffusionPipeline.from_pretrained(cfg["repo"], **kwargs).to(device)
    if args.lora:
        repo, _, weight_name = args.lora.partition(":")
        pipe.load_lora_weights(repo, weight_name=weight_name or None)
        pipe.fuse_lora(lora_scale=args.lora_scale)
    print(f"loaded {cfg['repo']} on {device} in {time.time() - t0:.1f}s")

    prompt = f"{args.prompt}, {args.style}" if args.style else args.prompt
    gen = torch.Generator("cpu").manual_seed(args.seed)
    t0 = time.time()
    image = pipe(
        prompt,
        num_inference_steps=args.steps or cfg["steps"],
        guidance_scale=cfg["guidance"],
        height=args.size,
        width=args.size,
        generator=gen,
    ).images[0]
    print(f"generated in {time.time() - t0:.1f}s")

    image.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
