"""Optimizer helper.

torch.optim.Adam's default CPU path (`_single_tensor_adam`) computes sqrt through Intel MKL's
vector math library, which can select an AVX-512 kernel (mkl_vml_kernel_sSqrt_Z0...) that crashes
with "Illegal instruction" on some WSL / virtualized CPUs, even with MKL_ENABLE_INSTRUCTIONS set.
The fused implementation is a single ATen kernel that does not call MKL, and it is also faster.
"""
import torch


def adam(params, lr: float = 1e-3) -> torch.optim.Optimizer:
    try:
        return torch.optim.Adam(params, lr=lr, fused=True)
    except (RuntimeError, TypeError, ValueError):     # older torch or unsupported device
        return torch.optim.Adam(params, lr=lr, foreach=False)
