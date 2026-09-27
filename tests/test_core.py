"""CPU smoke tests for the vector field, training objective and ODE integrators."""

import torch
import csv
import sys
from torch.utils.data import TensorDataset

from mnist_flow.flow import flow_matching_loss, sample
from mnist_flow.model import VelocityUNet
from mnist_flow import train
from mnist_flow import sample as sample_command
from mnist_flow import evaluate as evaluate_command


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


def test_fixed_sampling_noise_and_validation_inputs():
    initial = torch.zeros(2, 1, 28, 28)
    result = sample(ConstantVelocity(), 2, 4, torch.device("cpu"), noise=initial)
    torch.testing.assert_close(result, torch.ones_like(initial))
    torch.testing.assert_close(initial, torch.zeros_like(initial))
    images = torch.zeros(2, 1, 28, 28)
    loss = flow_matching_loss(ConstantVelocity(), images, noise=initial, t=torch.zeros(2))
    assert loss.item() == 1.0


def test_train_resume_and_sample_cli(monkeypatch, tmp_path):
    training = TensorDataset(torch.zeros(2, 1, 28, 28), torch.zeros(2, dtype=torch.long))
    monkeypatch.setattr(train.datasets, "MNIST", lambda _root, train, **_kwargs: training)
    output = tmp_path / "run"
    args = ["train", "--epochs", "1", "--batch-size", "2", "--channels", "8",
            "--time-dim", "32", "--preview-count", "2", "--preview-steps", "1",
            "--max-train-samples", "2", "--max-val-samples", "2", "--device", "cpu",
            "--output-dir", str(output)]
    monkeypatch.setattr(sys, "argv", args)
    train.main()
    assert (output / "checkpoint.pt").exists()
    assert (output / "best.pt").exists()
    assert (output / "epoch_001.png").exists()
    monkeypatch.setattr(sys, "argv", args[:2] + ["2"] + args[3:] +
                        ["--resume", str(output / "checkpoint.pt")])
    train.main()
    with (output / "metrics.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
    assert checkpoint["epoch"] == 2
    monkeypatch.setattr(sys, "argv", ["sample", "--checkpoint", str(output / "best.pt"),
                                      "--output", str(output / "generated.png"),
                                      "--num-samples", "2", "--steps", "1", "--device", "cpu"])
    sample_command.main()
    assert (output / "generated.png").exists()
    monkeypatch.setattr(evaluate_command.datasets, "MNIST", lambda _root, train, **_kwargs: training)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--checkpoint", str(output / "best.pt"),
                                      "--output", str(output / "test_metrics.json"),
                                      "--max-test-samples", "2", "--batch-size", "2", "--device", "cpu"])
    evaluate_command.main()
    assert (output / "test_metrics.json").exists()
