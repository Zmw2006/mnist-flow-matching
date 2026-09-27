"""CPU smoke tests for the vector field, training objective and ODE integrators."""

import torch

from mnist_flow.flow import flow_matching_loss, sample
from mnist_flow.model import VelocityUNet


def test_unet_output_shape_and_backward():
    model = VelocityUNet(channels=8, time_dim=32)
    images = torch.randn(2, 1, 28, 28)
    loss = flow_matching_loss(model, images)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert model.stem.weight.grad is not None
    assert model.stem.weight.grad.abs().sum() > 0


class ConstantVelocity(torch.nn.Module):
    def forward(self, x, t):
        assert t.shape == (x.shape[0],)
        return torch.ones_like(x)


def test_solvers_integrate_constant_velocity():
    for method in ("euler", "heun"):
        torch.manual_seed(0)
        expected = torch.randn(3, 1, 28, 28) + 1
        torch.manual_seed(0)
        actual = sample(ConstantVelocity(), 3, 4, torch.device("cpu"), method)
        torch.testing.assert_close(actual, expected)


def test_invalid_arguments():
    model = ConstantVelocity()
    for count, steps, method in ((0, 5, "euler"), (1, 0, "euler"), (1, 5, "invalid")):
        try:
            sample(model, count, steps, torch.device("cpu"), method)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid sample parameters should raise ValueError")
