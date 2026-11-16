# Context: smallest possible text-to-image app (cartoon style)

Goal: find out how small (in GB on disk) a prompt-to-image generator can get, with cartoon style as the target, then try things out hands-on.

## Where the size goes
- Text encoder (SD 1.5 uses CLIP ViT-L, ~123M params)
- Denoiser (SD 1.5 UNet, ~860M params), which is most of the size
- VAE decoder (~50–80M params)
- Size on disk ≈ params × bytes per weight (fp16 = 2 bytes; 8-bit = 1; 4-bit = 0.5)

## Size options (estimates as of mid-2026)
| Approach | Size on disk | Prompt freedom |
|---|---|---|
| SD 1.5, fp16 | ~2 GB | Open |
| SD 1.5, 4–6 bit (stable-diffusion.cpp, Apple Core ML palettization) | ~0.6–1 GB | Open |
| Distilled/mobile (BK-SDM-Tiny, Segmind tiny-sd/small-sd, SnapGen, MobileDiffusion), quantized | ~250–500 MB | Open |
| Custom cartoon-only diffusion model (50–150M params, 8-bit) | ~50–150 MB | Open-ish, one style |
| Small-vocabulary GAN with lookup-table "text encoder" | ~10–30 MB | Limited |

## Ideas to try
1. Baseline: run SD 1.5 quantized with stable-diffusion.cpp and measure size, speed and quality.
2. Try BK-SDM-Tiny / segmind tiny-sd, quantize them, and compare.
3. Fine-tune a small model (LoRA or full) on a cartoon dataset to get style consistency.
4. Shrink the text encoder or cache prompt embeddings to save another 100–250 MB.
5. Stretch goal: train a <100 MB cartoon-only model from scratch on the GB10. Most of the effort goes into building and captioning the dataset.

Hardware available: NVIDIA GB10 (DGX Spark-class) home lab machine.
