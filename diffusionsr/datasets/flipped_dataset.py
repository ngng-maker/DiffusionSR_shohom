"""
Dataset wrapper that applies horizontal flip (along the x / laser-direction axis)
BEFORE normalization, fixing the ordering bug in train_rrdn_encoder.py.

Bug in the original code (train_rrdn_encoder.py:146):
  flip was applied to already-normalized tensors, causing a spatially-varying bias
  because the pixel-wise statistics (mean_hr, std_hr) are not symmetric along x.

Fix here:
  1. Get the normalized tensors from the parent dataset.
  2. With probability p_flip, unscale ALL four tensors to physical units.
  3. Flip along dim=1 (H = x / laser direction, NOT z / depth).
  4. Rescale back to normalized units.
  5. Recompute residual = hr_norm - upscaled_lr_norm (so it is always consistent).

Usage:
    from diffusionsr.datasets.flipped_dataset import FlippedDataset
    ds = FlippedDataset(p_flip=0.2, downscale_method='direct', root_folder=..., split='train', ...)
"""
import random

import numpy as np
import torch

from diffusionsr.datasets.dataset import SimulationXZDataset


class FlippedDataset(SimulationXZDataset):
    """
    Drop-in replacement for SimulationXZDataset that applies a physically correct
    pre-normalization horizontal flip with probability p_flip.

    All public attributes (mean_hr, std_hr, field_names, factor, img_shape, etc.)
    and methods (unscale_data, rescale_data) are inherited unchanged.
    Only __getitem__ is overridden.
    """

    def __init__(self, *args, p_flip: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_flip = p_flip

        # Cache stats as float32 tensors with a batch dim so broadcasting is trivial.
        # Shape: (1, C, H, W) — the leading 1 broadcasts over the batch in train_coupled.
        # These are stored on CPU; train_coupled.ulr_to_hr() moves them to the right device.
        std_min = 1.0   # minimum std in Kelvin, prevents division by ~0

        def _t(arr):
            return torch.tensor(np.array(arr, dtype=np.float32))

        self._mean_hr_t   = _t(self.mean_hr)           # (C, H, W)
        self._std_hr_t    = _t(self.std_hr).clamp(min=std_min)
        self._mean_lr_t   = _t(self.mean_lr)
        self._std_lr_t    = _t(self.std_lr).clamp(min=std_min)
        self._mean_ulr_t  = _t(self.mean_upscaled_lr)
        self._std_ulr_t   = _t(self.std_upscaled_lr).clamp(min=std_min)
        # Residual stats (may not exist on all datasets — fall back to HR stats)
        _mean_res = getattr(self, 'mean_residual', self.mean_hr)
        _std_res  = getattr(self, 'std_residual',  self.std_hr)
        self._mean_res_t  = _t(_mean_res)
        self._std_res_t   = _t(_std_res).clamp(min=std_min)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unscale(norm_t: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
        """Undo standardization: x_phys = x_norm * std + mean."""
        return norm_t * std_t + mean_t

    @staticmethod
    def _rescale(phys_t: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor) -> torch.Tensor:
        """Apply standardization: x_norm = (x_phys - mean) / std."""
        return (phys_t - mean_t) / std_t

    # ------------------------------------------------------------------
    # __getitem__ override
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        res, hr, lr, ulr = super().__getitem__(idx)

        # Parent may return numpy arrays; convert so stat-tensor arithmetic works.
        def _to_t(x):
            return x if isinstance(x, torch.Tensor) else torch.tensor(np.array(x, dtype=np.float32))
        res, hr, lr, ulr = _to_t(res), _to_t(hr), _to_t(lr), _to_t(ulr)

        if random.random() >= self.p_flip:
            return res, hr, lr, ulr

        # --- convert to physical units ---
        hr_p  = self._unscale(hr,  self._mean_hr_t,  self._std_hr_t)
        lr_p  = self._unscale(lr,  self._mean_lr_t,  self._std_lr_t)
        ulr_p = self._unscale(ulr, self._mean_ulr_t, self._std_ulr_t)

        # --- flip along dim=1 (H = x / laser direction) ---
        hr_p  = torch.flip(hr_p,  dims=[1])
        lr_p  = torch.flip(lr_p,  dims=[1])
        ulr_p = torch.flip(ulr_p, dims=[1])

        # --- normalize back ---
        hr_n  = self._rescale(hr_p,  self._mean_hr_t,  self._std_hr_t)
        lr_n  = self._rescale(lr_p,  self._mean_lr_t,  self._std_lr_t)
        ulr_n = self._rescale(ulr_p, self._mean_ulr_t, self._std_ulr_t)

        # --- recompute residual from the now-consistent flipped tensors ---
        # Residual is defined in the parent as hr_norm - upscaled_lr_norm.
        res_n = hr_n - ulr_n

        return res_n, hr_n, lr_n, ulr_n
