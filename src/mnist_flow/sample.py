"""Sampling command: python -m mnist_flow.sample --help."""

import argparse
import math
from pathlib import Path

import torch
from torchvision.utils import save_image

from .flow import sample
from .model import VelocityUNet
from .utils import seed_everything, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MNIST-like digits from a trained checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoint.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/samples.png"))
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--method", choices=("euler", "heun"), default="heun")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    seed_everything(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = VelocityUNet(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
    images = sample(model, args.num_samples, args.steps, device, method=args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(((images.clamp(-1, 1) + 1) / 2).cpu(), args.output, nrow=math.ceil(math.sqrt(args.num_samples)))
    print(f"Saved {args.num_samples} generated images to {args.output}")


if __name__ == "__main__":
    main()
