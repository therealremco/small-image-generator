"""Export a Hugging Face image+caption dataset to a plain folder.

    uv run export_dataset.py --out data/cartoon

Writes 0000.png + 0000.blip.txt (the dataset's original caption) per image. Run caption.py next to
write better captions to 0000.txt.
"""
import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="Norod78/cartoon-blip-captions")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", type=Path, default=Path("data/cartoon"))
    ap.add_argument("--image-column", default="image")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--limit", type=int, help="export only the first N rows")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    n = min(len(ds), args.limit) if args.limit else len(ds)
    args.out.mkdir(parents=True, exist_ok=True)
    width = max(4, len(str(n - 1)))

    for i in range(n):
        row = ds[i]
        stem = f"{i:0{width}d}"
        row[args.image_column].convert("RGB").save(args.out / f"{stem}.png")
        (args.out / f"{stem}.blip.txt").write_text(row[args.text_column].strip() + "\n")
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"exported {i + 1}/{n}", flush=True)
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
