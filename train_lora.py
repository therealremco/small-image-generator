"""Train a LoRA on a small SD-1.5-architecture model (UNet attention layers only).

    uv run train_lora.py --steps 20 --out lora-test          # smoke test
    uv run train_lora.py --steps 3000 --batch-size 8 --out lora-cartoon

Then:  uv run generate.py "a fox" --model tiny-sd --lora lora-cartoon:pytorch_lora_weights.safetensors --style "cartoon style"
"""
import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

SUFFIX = ", cartoon style"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="segmind/tiny-sd")
    ap.add_argument("--dataset", default="Norod78/cartoon-blip-captions")
    ap.add_argument("--data-dir", type=Path, help="folder of X.png + X.txt pairs; overrides --dataset")
    ap.add_argument("--out", type=Path, default=Path("lora-cartoon"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=500)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    tokenizer = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer")
    f32 = dict(dtype=torch.float32)  # some repos store fp16 weights; train in fp32
    text_encoder = CLIPTextModel.from_pretrained(args.base, subfolder="text_encoder", **f32).to(device).eval()
    vae = AutoencoderKL.from_pretrained(args.base, subfolder="vae", **f32).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(args.base, subfolder="unet", **f32).to(device)
    scheduler = DDPMScheduler.from_pretrained(args.base, subfolder="scheduler")
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)

    unet.add_adapter(LoraConfig(
        r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    ))
    params = [p for p in unet.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)

    if args.data_dir:
        # Folder of X.png + X.txt pairs (see export_dataset.py / caption.py); images without a .txt are skipped.
        pairs = [(p, p.with_suffix(".txt")) for p in sorted(args.data_dir.glob("*.png"))]
        pairs = [(img, txt) for img, txt in pairs if txt.exists()]
        if not pairs:
            raise SystemExit(f"no image+caption pairs in {args.data_dir}")
        print(f"{len(pairs)} image+caption pairs from {args.data_dir}")

        def sample():
            img, txt = random.choice(pairs)
            return {"image": Image.open(img), "text": txt.read_text().strip()}
    else:
        ds = load_dataset(args.dataset, split="train")

        def sample():
            return ds[random.randrange(len(ds))]

    tf = transforms.Compose([
        transforms.Resize(args.size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def batch(n: int):
        rows = [sample() for _ in range(n)]
        pix = torch.stack([tf(r["image"].convert("RGB")) for r in rows]).to(device)
        ids = tokenizer([r["text"] + SUFFIX for r in rows], padding="max_length", truncation=True,
                        max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
        return pix, ids

    def save(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionPipeline.save_lora_weights(path, unet_lora_layers=sd)
        print(f"saved {path}")

    unet.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        pix, ids = batch(args.batch_size)
        with torch.no_grad():
            latents = vae.encode(pix).latent_dist.sample() * vae.config.scaling_factor
            cond = text_encoder(ids)[0]
        noise = torch.randn_like(latents)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
        noisy = scheduler.add_noise(latents, noise, t)
        pred = unet(noisy, t, cond).sample
        loss = F.mse_loss(pred.float(), noise.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 10 == 0 or step == 1:
            print(f"step {step}/{args.steps} loss {loss.item():.4f} ({(time.time() - t0) / step:.2f}s/step)", flush=True)
        if step % args.save_every == 0 and step != args.steps:
            save(args.out / f"step-{step}")
    save(args.out)


if __name__ == "__main__":
    main()
