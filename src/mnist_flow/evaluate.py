"""Report held-out MNIST test-set flow matching error for a checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .model import VelocityUNet
from .train import evaluate as evaluate_loss
from .utils import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure CFM MSE on untouched MNIST test images")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/test_metrics.json"))
    parser.add_argument("--max-test-samples", type=int, default=0, help="0 means all 10,000 test images")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if args.max_test_samples < 0 or args.batch_size < 1:
        raise ValueError("max-test-samples must be nonnegative; batch-size must be positive")
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = VelocityUNet(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
    transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))))
    dataset = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    if args.max_test_samples:
        dataset = Subset(dataset, range(min(args.max_test_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size)
    mse = evaluate_loss(model, loader, device, args.seed)
    result = {"test_cfm_mse": mse, "test_samples": len(dataset),
              "seed": args.seed, "checkpoint_epoch": checkpoint["epoch"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
