from __future__ import annotations

from typing import Iterable, Literal

import torch
import torch.nn.functional as F


def _to_nchw(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 2:
        return image.unsqueeze(0).unsqueeze(0)
    if image.dim() == 3:
        return image.unsqueeze(0)
    if image.dim() == 4:
        return image
    raise ValueError(f"Unsupported image dim for Frangi: {image.dim()}")


def _from_nchw(filtered: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if ref.dim() == 2:
        return filtered[0, 0]
    if ref.dim() == 3:
        return filtered[0]
    return filtered


def _conv2d_depthwise(image_nchw: torch.Tensor, kernel_1x1khkw: torch.Tensor) -> torch.Tensor:
    channels = image_nchw.shape[1]
    k = kernel_1x1khkw.shape[-1]
    weight = kernel_1x1khkw.to(device=image_nchw.device, dtype=image_nchw.dtype).expand(channels, 1, k, k)
    return F.conv2d(image_nchw, weight, padding=k // 2, groups=channels)


def frangi_vesselness(
    image: torch.Tensor,
    sigmas: Iterable[float] = (1.0, 2.0, 3.0),
    beta: float = 0.5,
    gamma: float = 15.0,
    black_ridges: bool = False,
    fusion: Literal["max", "soft"] = "soft",
    eps: float = 1e-6,
) -> torch.Tensor:
    image_nchw = _to_nchw(image)
    kernels: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for sigma in sigmas:
        if sigma <= 0:
            continue

        radius = max(1, int(round(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=image_nchw.device, dtype=image_nchw.dtype)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        sigma2 = sigma * sigma
        gaussian = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma2)) / (2.0 * torch.pi * sigma2)

        dxx = ((xx * xx - sigma2) / (sigma2 * sigma2)) * gaussian
        dyy = ((yy * yy - sigma2) / (sigma2 * sigma2)) * gaussian
        dxy = ((xx * yy) / (sigma2 * sigma2 * sigma2)) * gaussian

        dxx = (sigma2 * dxx).unsqueeze(0).unsqueeze(0)
        dyy = (sigma2 * dyy).unsqueeze(0).unsqueeze(0)
        dxy = (sigma2 * dxy).unsqueeze(0).unsqueeze(0)
        kernels.append((dxx, dyy, dxy))

    if len(kernels) == 0:
        return _from_nchw(torch.zeros_like(image_nchw), image)

    beta2 = max(beta * beta, eps)
    gamma2 = max(gamma * gamma, eps)

    responses: list[torch.Tensor] = []
    for dxx_kernel, dyy_kernel, dxy_kernel in kernels:
        dxx = _conv2d_depthwise(image_nchw, dxx_kernel)
        dyy = _conv2d_depthwise(image_nchw, dyy_kernel)
        dxy = _conv2d_depthwise(image_nchw, dxy_kernel)

        trace = dxx + dyy
        det_term = torch.sqrt((dxx - dyy) * (dxx - dyy) + 4.0 * dxy * dxy + eps)
        lambda1 = 0.5 * (trace + det_term)
        lambda2 = 0.5 * (trace - det_term)

        swap_mask = lambda1.abs() > lambda2.abs()
        lambda1_sorted = torch.where(swap_mask, lambda2, lambda1)
        lambda2_sorted = torch.where(swap_mask, lambda1, lambda2)

        rb = lambda1_sorted.abs() / (lambda2_sorted.abs() + eps)
        s2 = lambda1_sorted * lambda1_sorted + lambda2_sorted * lambda2_sorted
        vesselness = torch.exp(-(rb * rb) / (2.0 * beta2)) * (1.0 - torch.exp(-s2 / (2.0 * gamma2)))

        if black_ridges:
            vesselness = torch.where(lambda2_sorted < 0, torch.zeros_like(vesselness), vesselness)
        else:
            vesselness = torch.where(lambda2_sorted > 0, torch.zeros_like(vesselness), vesselness)

        responses.append(vesselness)

    if fusion == "soft":
        combined = torch.stack(responses, dim=0).mean(dim=0)
    else:
        combined, _ = torch.stack(responses, dim=0).max(dim=0)

    return _from_nchw(combined, image)


def frangi_mask(
    vesselness: torch.Tensor,
    threshold: float = 0.2,
    dilation_radius: int = 0,
    closing_radius: int = 0,
) -> torch.Tensor:
    vesselness = vesselness.float()
    max_vesselness = vesselness.max()
    if max_vesselness <= 0:
        mask = torch.zeros_like(vesselness, dtype=torch.bool)
    else:
        mask = vesselness >= (max_vesselness * threshold)

    # 1) Morphological closing — bridges small gaps between vessel fragments
    if closing_radius > 0:
        k = 2 * int(closing_radius) + 1
        m = _to_nchw(mask.float())
        # dilate
        m = F.max_pool2d(m, kernel_size=k, stride=1, padding=closing_radius)
        # erode (max-pool of negative, then negate)
        m = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=closing_radius)
        mask = _from_nchw(m > 0.5, mask)

    # 2) Outward dilation — expands mask beyond the vessel centreline
    if dilation_radius > 0:
        mask_nchw = _to_nchw(mask.float())
        kernel_size = 2 * int(dilation_radius) + 1
        mask_nchw = F.max_pool2d(mask_nchw, kernel_size=kernel_size, stride=1, padding=dilation_radius)
        mask = _from_nchw(mask_nchw > 0, mask)

    return mask