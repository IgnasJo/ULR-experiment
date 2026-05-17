"""
SR (Super-Resolution) quality metrics: PSNR, SSIM, LPIPS, FID.

These require ground-truth HR images alongside the SR outputs and match
the metrics reported in Table IV of the ULR2SS paper.

All tensor inputs are expected in [0, 1] float range, shape [1,C,H,W] or [C,H,W].
"""

import warnings
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Per-image metrics
# ---------------------------------------------------------------------------

def compute_psnr(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio for [0,1] float tensors.

    PSNR = 10 * log10(1 / MSE)

    Returns float('inf') if the images are identical.
    """
    mse = torch.mean((sr.float() - hr.float()) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(sr: torch.Tensor, hr: torch.Tensor) -> float:
    """
    Structural Similarity Index for [0,1] float tensors.

    Uses skimage.metrics.structural_similarity with channel_axis=-1.
    """
    from skimage.metrics import structural_similarity

    sr_np = sr.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    hr_np = hr.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    return float(structural_similarity(sr_np, hr_np, channel_axis=-1, data_range=1.0))


# ---------------------------------------------------------------------------
# FID helpers
# ---------------------------------------------------------------------------

def _extract_inception_features(images: list, device: str = 'cpu') -> np.ndarray:
    """
    Extract InceptionV3 pool3 (2048-D) features from a list of [C,H,W] tensors.

    The classifier head is replaced with Identity so the 2048-D avg-pool
    output is returned directly, matching the standard FID definition.
    """
    import torchvision.models as models
    import torchvision.transforms as T

    try:
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False,
        )
    except AttributeError:
        # Older torchvision fallback
        inception = models.inception_v3(pretrained=True, transform_input=False)

    inception.fc = torch.nn.Identity()
    inception.eval()
    inception = inception.to(device)

    resize = T.Resize((299, 299), antialias=True)

    feats = []
    with torch.no_grad():
        for img in images:
            img_t = resize(img.unsqueeze(0).to(device))
            feat = inception(img_t)          # [1, 2048]
            feats.append(feat.squeeze(0).cpu().numpy())

    return np.stack(feats)


def compute_fid(sr_features: np.ndarray, hr_features: np.ndarray) -> float:
    """
    Fréchet Inception Distance given pre-extracted features.

    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2·sqrt(Σ_r·Σ_g))

    Adds a small eps·I regularisation to covariances to handle rank deficiency
    (critical when N ≪ 2048, e.g. on the 4-image overfit test set).
    Complex output from sqrtm is discarded (imaginary artefact from near-singular
    matrices) — a warning is issued when the imaginary part is non-trivial.

    Returns float('nan') when called with fewer than 2 samples.
    """
    from scipy.linalg import sqrtm

    n = sr_features.shape[0]
    if n < 2:
        warnings.warn("FID requires ≥2 samples; returning NaN.", stacklevel=2)
        return float('nan')
    if n < 50:
        warnings.warn(
            f"FID computed from only {n} samples — result is not statistically meaningful.",
            stacklevel=2,
        )

    mu1 = np.mean(sr_features, axis=0)
    mu2 = np.mean(hr_features, axis=0)

    sigma1 = np.cov(sr_features, rowvar=False)
    sigma2 = np.cov(hr_features, rowvar=False)

    # Regularise to avoid singular covariance matrices
    eps = 1e-6
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.imag(covmean), 0, atol=1e-3):
            warnings.warn(
                "sqrtm produced a significant imaginary component; FID may be unreliable.",
                stacklevel=2,
            )
        covmean = np.real(covmean)

    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return fid


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

class SRMetricsAccumulator:
    """
    Accumulates per-image SR quality metrics and computes dataset-level FID.

    Usage::

        acc = SRMetricsAccumulator(device='cuda')
        for sr_img, hr_img in pairs:       # tensors in [0,1]
            acc.update(sr_img, hr_img)
        results = acc.summary()            # dict with PSNR, SSIM, LPIPS, FID
    """

    def __init__(self, device: str = 'cpu', compute_lpips: bool = True):
        self.device = device
        self.psnr_values: list = []
        self.ssim_values: list = []
        self.lpips_values: list = []
        self._sr_images: list = []
        self._hr_images: list = []

        self._lpips_fn = None
        if compute_lpips:
            try:
                import lpips
                self._lpips_fn = lpips.LPIPS(net='alex').to(device)
                self._lpips_fn.eval()
            except ImportError:
                pass  # LPIPS optional; skip silently if not available

    def update(self, sr: torch.Tensor, hr: torch.Tensor) -> None:
        """
        Accumulate metrics for one (sr, hr) image pair.

        Args:
            sr: [1,C,H,W] or [C,H,W] float tensor in [0,1]
            hr: same shape/dtype
        """
        sr = sr.float().cpu()
        hr = hr.float().cpu()

        # Normalise to 4-D
        if sr.dim() == 3:
            sr = sr.unsqueeze(0)
        if hr.dim() == 3:
            hr = hr.unsqueeze(0)

        self.psnr_values.append(compute_psnr(sr, hr))
        self.ssim_values.append(compute_ssim(sr, hr))

        if self._lpips_fn is not None:
            # LPIPS expects [-1, 1]
            with torch.no_grad():
                sr_lp = (sr * 2 - 1).to(self.device)
                hr_lp = (hr * 2 - 1).to(self.device)
                lp_val = self._lpips_fn(sr_lp, hr_lp)
                self.lpips_values.append(float(lp_val.item()))

        # Store full images for FID (kept as [C,H,W] CPU tensors)
        self._sr_images.append(sr.squeeze(0))
        self._hr_images.append(hr.squeeze(0))

    def get_fid(self) -> float:
        """Compute FID over all accumulated images (runs InceptionV3)."""
        if len(self._sr_images) < 2:
            return float('nan')
        sr_feats = _extract_inception_features(self._sr_images, device=self.device)
        hr_feats = _extract_inception_features(self._hr_images, device=self.device)
        return compute_fid(sr_feats, hr_feats)

    def summary(self, compute_fid_score: bool = True) -> dict:
        """
        Return dict of all SR metrics.

        Args:
            compute_fid_score: When True, also run FID (requires InceptionV3 forward
                passes over the full accumulated set — may be slow on CPU).

        Returns:
            Dict with keys: PSNR, SSIM, LPIPS (if available), FID (if requested).
        """
        if not self.psnr_values:
            return {}

        finite_psnr = [p for p in self.psnr_values if np.isfinite(p)]
        result = {
            'PSNR': float(np.mean(finite_psnr)) if finite_psnr else float('nan'),
            'SSIM': float(np.mean(self.ssim_values)),
        }

        if self.lpips_values:
            result['LPIPS'] = float(np.mean(self.lpips_values))

        if compute_fid_score:
            result['FID'] = self.get_fid()

        return result

    def reset(self) -> None:
        """Clear all accumulated state."""
        self.psnr_values.clear()
        self.ssim_values.clear()
        self.lpips_values.clear()
        self._sr_images.clear()
        self._hr_images.clear()
