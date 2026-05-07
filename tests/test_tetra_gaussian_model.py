import torch

from scene.tetra_gaussian_model import (
    tetra_signed_volumes,
    tetra_to_gaussian_moment_matching,
)


def test_tetra_to_gaussian_moment_matching_shapes_and_finiteness():
    if not torch.cuda.is_available():
        return
    tetra = torch.tensor(
        [[[0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]]],
        device="cuda",
        requires_grad=True,
    )

    means, scales, rotations, cov = tetra_to_gaussian_moment_matching(tetra, eta=1.0, eps=1e-4)

    assert means.shape == (1, 3)
    assert scales.shape == (1, 3)
    assert rotations.shape == (1, 4)
    assert cov.shape == (1, 3, 3)
    assert torch.isfinite(means).all()
    assert torch.isfinite(scales).all()
    assert torch.isfinite(rotations).all()
    assert (scales > 0).all()
    assert torch.allclose(torch.linalg.norm(rotations, dim=1), torch.ones(1, device="cuda"), atol=1e-5)

    loss = means.square().sum() + scales.sum() + rotations.square().sum()
    loss.backward()
    assert tetra.grad is not None
    assert torch.isfinite(tetra.grad).all()


def test_tetra_signed_volume_detects_orientation():
    if not torch.cuda.is_available():
        return
    tetra = torch.tensor(
        [[[0.0, 0.0, 0.0],
          [1.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]],
         [[1.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]]],
        device="cuda",
    )

    volumes = tetra_signed_volumes(tetra)
    assert volumes[0] > 0
    assert volumes[1] < 0
