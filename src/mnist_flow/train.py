"""Training command: python -m mnist_flow.train --help."""

import argparse
import csv
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
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
    parser.add_argument("--preview-count", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means all 54,000 training-split images")
    parser.add_argument("--max-val-samples", type=int, default=1000, help="0 means all 6,000 validation images")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--resume", type=Path, help="Resume from a trusted checkpoint created by this project")
    return parser.parse_args()


@torch.inference_mode()
def evaluate(model, loader: DataLoader, device: torch.device, seed: int) -> float:
    """Use the same independent noise and time pairs every epoch for comparison."""
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    total = 0.0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        noise = torch.randn(images.shape, generator=generator).to(device)
        t = torch.rand(images.shape[0], generator=generator).to(device)
        loss = flow_matching_loss(model, images, noise=noise, t=t)
        total += loss.item() * images.shape[0]
    return total / len(loader.dataset)


def record_metrics(path: Path, epoch: int, train_mse: float, val_mse: float) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(("epoch", "train_mse", "val_ema_mse"))
        writer.writerow((epoch, f"{train_mse:.8f}", f"{val_mse:.8f}"))


def prepare_metrics(path: Path, completed_epochs: int) -> None:
    """Discard rows newer than a resumed checkpoint to avoid duplicated epochs."""
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if int(row["epoch"]) <= completed_epochs]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("epoch", "train_mse", "val_ema_mse"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.num_workers < 0:
        raise ValueError("epochs and batch-size must be positive, lr > 0, num-workers >= 0")
    if not 0 <= args.ema_decay < 1 or args.preview_steps < 1 or args.preview_count < 1:
        raise ValueError("ema-decay must be in [0,1); preview-steps and preview-count must be positive")
    if args.max_train_samples < 0 or args.max_val_samples < 0:
        raise ValueError("dataset sample limits must be nonnegative")
    if args.resume and args.resume.parent.resolve() != args.output_dir.resolve():
        raise ValueError("--output-dir must be the checkpoint's directory when resuming")
    device = select_device(args.device)
    seed_everything(args.seed)
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if checkpoint:
        config = checkpoint["config"]
        if (args.channels, args.time_dim) != (config["channels"], config["time_dim"]):
            raise ValueError("--channels and --time-dim must match the checkpoint when resuming")
        previous = checkpoint.get("run_config")
        current = {"seed": args.seed, "batch_size": args.batch_size,
                   "max_train_samples": args.max_train_samples, "max_val_samples": args.max_val_samples}
        if previous and previous != current:
            raise ValueError("--seed, --batch-size and dataset limits must match the checkpoint when resuming")
    model = VelocityUNet(args.channels, args.time_dim).to(device)
    ema = deepcopy(model).eval()
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader_generator = torch.Generator().manual_seed(args.seed)
    start_epoch = 0
    best_val = float("inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        start_epoch = checkpoint["epoch"]
        best_val = checkpoint.get("best_val", best_val)
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        if "data_rng_state" in checkpoint:
            loader_generator.set_state(checkpoint["data_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))))
    dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    if len(dataset) < 2:
        raise ValueError("MNIST training set needs at least two images for train/validation split")
    cutoff = int(len(dataset) * 0.9)
    training = Subset(dataset, range(cutoff))
    validation = Subset(dataset, range(cutoff, len(dataset)))
    if args.max_train_samples:
        training = Subset(training, range(min(args.max_train_samples, len(training))))
    if args.max_val_samples:
        validation = Subset(validation, range(min(args.max_val_samples, len(validation))))
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", generator=loader_generator)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    prepare_metrics(metrics_path, start_epoch if checkpoint else 0)
    preview_noise = torch.randn((args.preview_count, 1, 28, 28),
                                generator=torch.Generator().manual_seed(args.seed + 2)).to(device)
    print(f"Device: {device}; train: {len(training)}; validation: {len(validation)}; "
          f"epochs: {start_epoch + 1}..{args.epochs}")
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
        train_mse = total_loss / len(training)
        val_mse = evaluate(ema, val_loader, device, args.seed + 1)
        improved = val_mse < best_val
        best_val = min(best_val, val_mse)
        print(f"Epoch {epoch + 1}: train MSE = {train_mse:.6f}, validation EMA MSE = {val_mse:.6f}")
        preview = sample(ema, count=args.preview_count, steps=args.preview_steps, device=device,
                         noise=preview_noise)
        save_image(((preview.clamp(-1, 1) + 1) / 2).cpu(),
                   args.output_dir / f"epoch_{epoch + 1:03d}.png", nrow=min(4, args.preview_count))
        state = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {"channels": args.channels, "time_dim": args.time_dim},
            "run_config": {"seed": args.seed, "batch_size": args.batch_size,
                           "max_train_samples": args.max_train_samples, "max_val_samples": args.max_val_samples},
            "best_val": best_val,
            "rng_state": torch.get_rng_state(),
            "data_rng_state": loader_generator.get_state(),
            **({"cuda_rng_state": torch.cuda.get_rng_state_all()} if device.type == "cuda" else {}),
        }
        record_metrics(metrics_path, epoch + 1, train_mse, val_mse)
        save_checkpoint(args.output_dir / "checkpoint.pt", state)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", state)
    print(f"Latest: {args.output_dir / 'checkpoint.pt'}; best: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
