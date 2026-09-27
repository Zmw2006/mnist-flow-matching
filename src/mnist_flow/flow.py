"""Straight-path conditional flow matching and ODE sampling."""

import torch
from torch.nn import functional as F


def flow_matching_loss(
    model, images: torch.Tensor, noise: torch.Tensor | None = None, t: torch.Tensor | None = None
) -> torch.Tensor:
    """x_t=(1-t)x_0+t*x_1, with target velocity x_1-x_0."""
    noise = torch.randn_like(images) if noise is None else noise
    t = torch.rand(images.shape[0], device=images.device, dtype=images.dtype) if t is None else t
    if noise.shape != images.shape or t.shape != (images.shape[0],):
        raise ValueError("noise must match images; t must have shape [batch]")
    time = t[:, None, None, None]
    x_t = (1 - time) * noise + time * images
    return F.mse_loss(model(x_t, t), images - noise)


@torch.inference_mode()
def sample(
    model, count: int, steps: int, device: torch.device, method: str = "heun",
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate dx/dt=v_theta(x,t) from t=0 to t=1."""
    if count < 1 or steps < 1 or method not in {"euler", "heun"}:
        raise ValueError("count and steps must be positive; method must be euler or heun")
    if noise is not None and (noise.shape != (count, 1, 28, 28) or noise.device != device):
        raise ValueError("noise must have shape [count, 1, 28, 28] on the selected device")
    x = torch.randn(count, 1, 28, 28, device=device) if noise is None else noise.clone()
    dt = 1.0 / steps
    was_training = model.training
    model.eval()
    try:
        for i in range(steps):
            t = torch.full((count,), i * dt, device=device)
            velocity = model(x, t)
            if method == "euler":
                x = x + dt * velocity
            else:
                predicted = x + dt * velocity
                t_next = torch.full_like(t, (i + 1) * dt)
                x = x + 0.5 * dt * (velocity + model(predicted, t_next))
    finally:
        model.train(was_training)
    return x
