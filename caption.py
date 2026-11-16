"""Caption a folder of images with Florence-2 (native transformers support, no remote code).

    uv run caption.py data/cartoon --limit 8        # try a few first
    uv run caption.py data/cartoon                  # whole folder

For each image X.png, writes X.txt. Existing .txt files are skipped, so an interrupted run can be
resumed (use --overwrite to redo them). The original BLIP captions (X.blip.txt) are left alone.
"""
import argparse
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def clean(text: str, max_sentences: int) -> str:
    """Drop the 'The image shows' boilerplate and the hallucinated trailing sentences."""
    text = re.sub(r"^The (image|picture|scene) (shows|features|depicts|is)\s+(an?\s+|the\s+)?", "", text.strip(), flags=re.I)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    text = " ".join(sentences[:max_sentences]) if max_sentences else " ".join(sentences)
    return text[:1].upper() + text[1:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--model", default="florence-community/Florence-2-large")
    ap.add_argument("--task", default="<DETAILED_CAPTION>",
                    choices=["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"],
                    help="short, medium or long captions")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--num-beams", type=int, default=3)
    ap.add_argument("--max-sentences", type=int, default=2,
                    help="keep only the first N sentences; later ones tend to be hallucinated (0 = keep all)")
    ap.add_argument("--limit", type=int, help="caption only the first N images")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    images = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in EXTS)
    if not args.overwrite:
        images = [p for p in images if not p.with_suffix(".txt").exists()]
    if args.limit:
        images = images[: args.limit]
    if not images:
        print("nothing to do")
        return

    device = pick_device()
    dtype = torch.float16 if device != "cpu" else torch.float32
    print(f"device: {device}, {len(images)} images, task {args.task}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = Florence2ForConditionalGeneration.from_pretrained(args.model, dtype=dtype).to(device).eval()

    t0 = time.time()
    for start in range(0, len(images), args.batch_size):
        paths = images[start : start + args.batch_size]
        pics = [Image.open(p).convert("RGB") for p in paths]
        inputs = processor(text=[args.task] * len(pics), images=pics, return_tensors="pt").to(device, dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 num_beams=args.num_beams, do_sample=False)
        texts = [clean(t, args.max_sentences) for t in processor.batch_decode(out, skip_special_tokens=True)]
        for p, text in zip(paths, texts):
            p.with_suffix(".txt").write_text(text.strip() + "\n")
        done = start + len(paths)
        print(f"{done}/{len(images)} ({(time.time() - t0) / done:.2f}s/img)  e.g. {texts[0].strip()[:100]}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
