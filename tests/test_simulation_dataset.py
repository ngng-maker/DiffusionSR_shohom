from pathlib import Path

import numpy as np

from diffusionsr.datasets.dataset import SimulationXZDataset


def _write_sample(root: Path, split: str, hr: np.ndarray, lr: np.ndarray, upscaled_lr: np.ndarray) -> None:
    name = "sample_power100velocity200_1.0.npy"
    hr_dir = root / split / "HR"
    lr_dir = root / split / "LR" / "direct" / "1x"
    upscaled_dir = root / split / "LR" / "direct" / "2x"

    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    upscaled_dir.mkdir(parents=True, exist_ok=True)

    np.save(hr_dir / name, hr)
    np.save(lr_dir / name, lr)
    np.save(upscaled_dir / name, upscaled_lr)


def _make_single_field_root(tmp_path: Path) -> Path:
    root = tmp_path / "mock_ss316l_single_field"
    hr = np.arange(36, dtype=float).reshape(6, 6) + 293
    lr = np.arange(9, dtype=float).reshape(3, 3) + 293
    upscaled_lr = np.arange(36, dtype=float).reshape(6, 6) + 293
    _write_sample(root, "train", hr=hr, lr=lr, upscaled_lr=upscaled_lr)
    return root


def _make_multi_field_root(tmp_path: Path) -> Path:
    root = tmp_path / "mock_ss316l_multi_field"
    hr = np.stack(
        [np.arange(36, dtype=float).reshape(6, 6) + 293, np.ones((6, 6), dtype=float)],
        axis=-1,
    )
    lr = np.stack(
        [np.arange(9, dtype=float).reshape(3, 3) + 293, np.ones((3, 3), dtype=float)],
        axis=-1,
    )
    upscaled_lr = np.stack(
        [np.arange(36, dtype=float).reshape(6, 6) + 293, np.ones((6, 6), dtype=float)],
        axis=-1,
    )
    _write_sample(root, "train", hr=hr, lr=lr, upscaled_lr=upscaled_lr)
    return root


def test_single_field_dataset_keeps_original_2d_shapes(tmp_path: Path) -> None:
    root = _make_single_field_root(tmp_path)

    dataset = SimulationXZDataset(
        downscale_method="direct",
        root_folder=str(root),
        split="train",
        normalize="standardize",
        field_names=["temperature"],
    )

    residual, hr, lr, upscaled_lr = dataset[0]

    assert dataset.factor == 2
    assert dataset.num_fields == 1
    assert residual.shape == (1, 6, 6)
    assert hr.shape == (1, 6, 6)
    assert lr.shape == (1, 3, 3)
    assert upscaled_lr.shape == (1, 6, 6)


def test_ss316l_dataset_defaults_to_all_fields(tmp_path: Path) -> None:
    root = _make_multi_field_root(tmp_path)

    dataset = SimulationXZDataset(
        downscale_method="direct",
        root_folder=str(root),
        split="train",
        normalize="standardize",
    )

    residual, hr, lr, upscaled_lr = dataset[0]

    assert dataset.num_fields == 2
    assert residual.shape == (2, 6, 6)
    assert hr.shape == (2, 6, 6)
    assert lr.shape == (2, 3, 3)
    assert upscaled_lr.shape == (2, 6, 6)


def test_inflate_dim_expands_single_field_into_fake_depth(tmp_path: Path) -> None:
    root = _make_single_field_root(tmp_path)

    dataset = SimulationXZDataset(
        downscale_method="direct",
        root_folder=str(root),
        split="train",
        normalize="standardize",
        field_names=["temperature"],
        inflate_dim=4,
        inflate_method="repeat",
    )

    residual, hr, lr, upscaled_lr = dataset[0]

    assert dataset.num_fields == 4
    assert residual.shape == (4, 6, 6)
    assert hr.shape == (4, 6, 6)
    assert lr.shape == (4, 3, 3)
    assert upscaled_lr.shape == (4, 6, 6)
    np.testing.assert_allclose(hr[0], hr[1])
    np.testing.assert_allclose(hr[1], hr[2])
