"""Training command: python -m mnist_flow.train --help."""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from .flow import flow_matching_loss, sample
from .model import VelocityUNet
from .utils import save_checkpoint, seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an unconditional MNIST flow matching model")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs, including completed epochs when resuming")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--preview-steps", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--resume", type=Path, help="Resume from a trusted checkpoint created by this project")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.num_workers < 0:
        raise ValueError("epochs and batch-size must be positive, lr > 0, num-workers >= 0")
    if not 0 <= args.ema_decay < 1 or args.preview_steps < 1:
        raise ValueError("ema-decay must be in [0,1); preview-steps must be positive")
    device = select_device(args.device)
    seed_everything(args.seed)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if checkpoint:
        config = checkpoint["config"]
        if (args.channels, args.time_dim) != (config["channels"], config["time_dim"]):
            raise ValueError("--channels and --time-dim must match the checkpoint when resuming")
    model = VelocityUNet(args.channels, args.time_dim).to(device)
    ema = deepcopy(model).eval()
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        start_epoch = checkpoint["epoch"]
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))))
    dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; training images: {len(dataset)}; epochs: {start_epoch + 1}..{args.epochs}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        for images, _ in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            loss = flow_matching_loss(model, images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                for ema_param, param in zip(ema.parameters(), model.parameters()):
                    ema_param.lerp_(param, 1 - args.ema_decay)
                for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
                    ema_buffer.copy_(buffer)
            total_loss += loss.item() * images.shape[0]
        print(f"Epoch {epoch + 1}: mean flow matching MSE = {total_loss / len(dataset):.6f}")
        from torchvision.utils import save_image

        preview = sample(ema, count=16, steps=args.preview_steps, device=device)
        save_image(((preview.clamp(-1, 1) + 1) / 2).cpu(), args.output_dir / f"epoch_{epoch + 1:03d}.png", nrow=4)
        save_checkpoint(args.output_dir / "checkpoint.pt", {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {"channels": args.channels, "time_dim": args.time_dim},
            "rng_state": torch.get_rng_state(),
            **({"cuda_rng_state": torch.cuda.get_rng_state_all()} if device.type == "cuda" else {}),
        })
    print(f"Checkpoint: {args.output_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
