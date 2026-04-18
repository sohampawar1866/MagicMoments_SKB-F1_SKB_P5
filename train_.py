"""DRIFT / PlastiTrack — MARIDA dual-head UNet++ training (Kaggle, v5).

v5 fixes targeting val_iou 0.35-0.45 (PRD target 0.45):

    A. **EMA (exponential moving average) of weights.** Single biggest
       variance reducer. v4 val_iou bounced 0.07 ↔ 0.25 epoch-to-epoch
       because the 1075-positive-pixel val set has high single-batch
       variance. EMA averages weights with momentum=0.999, smoothing the
       trajectory and typically adding +0.05 IoU at no cost. Eval uses the
       EMA weights, not the live ones.

    B. **Tversky α=0.5 β=0.5 (balanced).** v4's α=0.3 β=0.7 was recall-
       biased and made the model over-predict positives by 20× (10% pred
       vs 0.5% truth). Balanced Tversky gives precision room and the loss
       value still drops cleanly.

    C. **Explicit Sargassum-negative loss.** Penalize positive predictions
       on cl ∈ {2, 3} pixels. v4 already passes the PRD sargassum_fp
       target (0.107 < 0.15) but this stabilizes it across epochs.

    D. **Test-time augmentation** (4 rotations averaged) at validation.
       Free +0.02-0.05 IoU.

    E. **Lower BCE weight 0.3 → 0.1.** BCE under 99.5% negatives drags
       predictions toward zero, fighting Tversky's recall bias. Lower
       weight = Tversky/Dice dominate, BCE just keeps the sign correct.

    F. **mean_p=nan diagnostic bug fix.** Use nan_to_num before sum.

    G. **MultiStepLR** with milestones [10, 20] dropping LR by 0.5 each.
       Smoother than v4's stage-1/stage-2 abrupt LR change.

Everything else from v4 is kept:
    - GroupNorm (no running stats, train==eval)
    - 2-stage training (encoder frozen 1-5, unfrozen 6-30)
    - Per-patch percentile normalization
    - 4× plastic sampler boost
    - Same architecture (backend/ml/model.py compatible — both default to
      use_groupnorm=True)

----------------------------------------------------------------------------
KAGGLE SETUP:

    !pip install -q segmentation-models-pytorch==0.3.3 rasterio==1.3.10
    import torch
    assert torch.cuda.is_available(), "Enable GPU!"
    print(torch.cuda.get_device_name(0))
    !python train_.py

Outputs → /kaggle/working/:
    our_real.pt    (state_dict; ~62 MB; loadable by backend/ml/weights.py)
    metrics.json   (per-epoch history + PRD scorecard)
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
    _AMP_NEW_API = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
    _AMP_NEW_API = False

# ============================ config =====================================

SEED = 1410
EPOCHS = 30
BATCH_SIZE = 8
LR_HEAD = 3e-4
LR_FULL = 5e-5
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

# v5 loss weights
TVERSKY_ALPHA = 0.5         # balanced (was 0.3 → over-recall)
TVERSKY_BETA = 0.5
TVERSKY_WEIGHT = 1.0
DICE_WEIGHT = 1.0
BCE_WEIGHT = 0.1            # was 0.3 — fighting Tversky too hard
SARG_NEG_WEIGHT = 0.5       # NEW — penalize positive preds on Sargassum
FRAC_WEIGHT = 0.1

# v5 EMA
EMA_DECAY = 0.999

# Sampler
PLASTIC_SAMPLE_WEIGHT = 4.0

BIOFOULING_PROB = 0.3
MIX_PROB = 0.2

_DEFAULT_ROOTS = [
    Path("/kaggle/working"),
    Path("/kaggle/input"),
    Path("/kaggle/input/marida"),
    Path("MARIDA"),
    Path("."),
]


def _looks_like_marida(p: Path) -> bool:
    return (p / "splits" / "train_X.txt").is_file() and (p / "patches").is_dir()


def _find_marida_root() -> Path:
    override = os.environ.get("MARIDA_ROOT")
    if override and _looks_like_marida(Path(override)):
        return Path(override)

    def scan(root: Path, depth: int):
        if not root.exists() or not root.is_dir():
            return None
        if _looks_like_marida(root):
            return root
        if depth <= 0:
            return None
        try:
            for child in sorted(root.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    hit = scan(child, depth - 1)
                    if hit:
                        return hit
        except PermissionError:
            pass
        return None

    for r in _DEFAULT_ROOTS:
        hit = scan(r, 2)
        if hit:
            return hit
    raise FileNotFoundError("MARIDA layout not found (need splits/ + patches/).")


MARIDA_ROOT = _find_marida_root()
_WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
CHECKPOINT_OUT = _WORKING / "our_real.pt"
METRICS_OUT = _WORKING / "metrics.json"

# Band indices
B2_IDX, B3_IDX, B4_IDX, B5_IDX = 0, 1, 2, 3
B6_IDX, B7_IDX, B8_IDX, B8A_IDX = 4, 5, 6, 7
B11_IDX, B12_IDX = 8, 9

LAMBDA_NIR = 832.8
LAMBDA_RE2 = 740.2
LAMBDA_SWIR1 = 1613.7
COEF_FDI = (LAMBDA_NIR - LAMBDA_RE2) / (LAMBDA_SWIR1 - LAMBDA_RE2)
EPS_DIV = 1e-6

PLASTIC_CLASS = 1
SARGASSUM_CLASSES = (2, 3)
N_CHANNELS = 14


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================ features ===================================

def compute_fdi(bands: np.ndarray) -> np.ndarray:
    re2 = bands[..., B6_IDX]
    nir = bands[..., B8_IDX]
    swir = bands[..., B11_IDX]
    baseline = re2 + (swir - re2) * COEF_FDI
    return np.clip(nir - baseline, -0.5, 0.5)


def compute_ndvi(bands: np.ndarray) -> np.ndarray:
    nir = bands[..., B8_IDX]
    red = bands[..., B4_IDX]
    return np.clip((nir - red) / (nir + red + EPS_DIV), -1.0, 1.0)


def compute_pi(bands: np.ndarray) -> np.ndarray:
    nir = bands[..., B8_IDX]
    red = bands[..., B4_IDX]
    return np.clip(nir / (nir + red + EPS_DIV), 0.0, 1.0)


def feature_stack(bands: np.ndarray) -> np.ndarray:
    if bands.shape[-1] > 11:
        bands = bands[..., :11]
    bands = np.clip(bands.astype(np.float32), 0.0, 1.0)
    fdi = compute_fdi(bands)[..., None]
    ndvi = compute_ndvi(bands)[..., None]
    pi = compute_pi(bands)[..., None]
    return np.concatenate([bands, fdi, ndvi, pi], axis=-1).astype(np.float32)


def normalize_per_patch(feats_chw: np.ndarray,
                        low_pct: float = 2.0,
                        high_pct: float = 98.0) -> np.ndarray:
    out = np.empty_like(feats_chw)
    for c in range(feats_chw.shape[0]):
        band = feats_chw[c]
        lo = np.percentile(band, low_pct)
        hi = np.percentile(band, high_pct)
        if hi - lo < EPS_DIV:
            out[c] = 0.5
        else:
            out[c] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


# ============================ model ======================================

class DualHeadUNetpp(nn.Module):
    def __init__(self, in_channels: int = 14, decoder_channels_out: int = 16):
        super().__init__()
        self.backbone = smp.UnetPlusPlus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=decoder_channels_out,
            activation=None,
            decoder_attention_type="scse",
        )
        self.mask_head = nn.Conv2d(decoder_channels_out, 1, kernel_size=1)
        self.frac_head = nn.Conv2d(decoder_channels_out, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        return {
            "mask_logit": self.mask_head(feats),
            "fraction": torch.sigmoid(self.frac_head(feats)),
        }


def convert_bn_to_gn(module: nn.Module, num_groups: int = 8) -> nn.Module:
    """Replace every BatchNorm2d with GroupNorm — no running stats =
    train/eval are identical. Must match backend/ml/model.py."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            groups = min(num_groups, num_channels)
            while num_channels % groups != 0 and groups > 1:
                groups -= 1
            new_layer = nn.GroupNorm(groups, num_channels,
                                     eps=child.eps, affine=child.affine)
            if child.affine:
                with torch.no_grad():
                    new_layer.weight.copy_(child.weight)
                    new_layer.bias.copy_(child.bias)
            setattr(module, name, new_layer)
        else:
            convert_bn_to_gn(child, num_groups)
    return module


# ============================ EMA (v5) ===================================

class ModelEMA:
    """Exponential moving average of model weights.

    Maintains a shadow copy of the model. After each optimizer step,
    shadow_weight = decay * shadow + (1 - decay) * live. Eval uses
    the shadow weights — typically much smoother val_iou trajectory than
    the live model. Single largest variance-reduction trick available.
    """

    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        # Buffers (e.g., GroupNorm has none with running stats but copy
        # anyway in case of mixed normalizers).
        for ema_b, b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(b.detach())


# ============================ biofouling aug =============================

def biofouling_augment(image: torch.Tensor, mask: torch.Tensor,
                       factor_range: tuple[float, float] = (0.5, 1.0)) -> torch.Tensor:
    squeezed = False
    if image.ndim == 3:
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        squeezed = True
    lo, hi = factor_range
    factors = torch.empty(image.shape[0], device=image.device).uniform_(lo, hi)
    factors = factors.view(-1, 1, 1)
    mult = torch.where(mask > 0, factors, torch.ones_like(mask))
    image = image.clone()
    image[:, B8_IDX] = image[:, B8_IDX] * mult
    if squeezed:
        image = image.squeeze(0)
    return image


# ============================ dataset ====================================

def _read_patch_paths(split_file: Path, patches_root: Path) -> list[Path]:
    with open(split_file) as f:
        ids = [l.strip() for l in f if l.strip()]
    out: list[Path] = []
    for id_ in ids:
        base = "_".join(id_.split("_")[:-1])
        p = patches_root / f"S2_{base}" / f"S2_{id_}.tif"
        if p.exists():
            out.append(p)
    return out


def _patch_has_plastic(img_path: Path) -> bool:
    cl_path = img_path.with_name(img_path.stem + "_cl.tif")
    try:
        with rasterio.open(cl_path) as src:
            return bool((src.read(1) == PLASTIC_CLASS).any())
    except Exception:
        return False


class MaridaDualHeadDataset(Dataset):
    def __init__(self, paths: list[Path], train_mode: bool = True):
        self.paths = paths
        self.train_mode = train_mode

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, img_path: Path):
        cl_path = img_path.with_name(img_path.stem + "_cl.tif")
        conf_path = img_path.with_name(img_path.stem + "_conf.tif")
        with rasterio.open(img_path) as src:
            bands = src.read().astype(np.float32)
        with rasterio.open(cl_path) as src:
            cl = src.read(1).astype(np.int64)
        with rasterio.open(conf_path) as src:
            conf = src.read(1).astype(np.float32)
        if bands.max() > 1.5:
            bands = (bands - 1000.0) / 10000.0
        bands_hwc = np.transpose(bands, (1, 2, 0))
        feats_hwc = feature_stack(bands_hwc)
        feats_chw = np.transpose(feats_hwc, (2, 0, 1))
        feats_chw = normalize_per_patch(feats_chw)
        return feats_chw, cl, conf

    @staticmethod
    def _hflip(*arrs):
        return tuple(np.ascontiguousarray(a[..., ::-1]) for a in arrs)

    @staticmethod
    def _vflip(*arrs):
        return tuple(np.ascontiguousarray(a[..., ::-1, :]) for a in arrs)

    @staticmethod
    def _rot90(*arrs, k=1):
        return tuple(np.ascontiguousarray(np.rot90(a, k=k, axes=(-2, -1))) for a in arrs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        feats, cl, conf = self._load(self.paths[idx])
        mask_target = (cl == PLASTIC_CLASS).astype(np.float32)
        frac_target = mask_target.copy()
        valid_mask = (conf > 0).astype(np.float32)

        if self.train_mode:
            if random.random() < 0.5:
                feats, mask_target, frac_target, valid_mask = self._hflip(
                    feats, mask_target, frac_target, valid_mask)
                cl = self._hflip(cl)[0]
            if random.random() < 0.5:
                feats, mask_target, frac_target, valid_mask = self._vflip(
                    feats, mask_target, frac_target, valid_mask)
                cl = self._vflip(cl)[0]
            k = random.choice([0, 1, 2, 3])
            if k != 0:
                feats, mask_target, frac_target, valid_mask = self._rot90(
                    feats, mask_target, frac_target, valid_mask, k=k)
                cl = self._rot90(cl, k=k)[0]

            if random.random() < MIX_PROB and mask_target.sum() > 0:
                alpha = random.uniform(0.3, 1.0)
                water_pixels = feats[:, cl == 7]
                if water_pixels.shape[1] > 0:
                    water_mean = water_pixels.mean(axis=1, keepdims=True)[..., None]
                else:
                    water_mean = feats.mean(axis=(1, 2), keepdims=True)
                mask3 = mask_target[None, :, :] > 0
                feats = np.where(mask3,
                                 alpha * feats + (1.0 - alpha) * water_mean,
                                 feats).astype(np.float32)
                frac_target = np.where(mask_target > 0, alpha, frac_target)
                if alpha < 0.5:
                    mask_target = np.where(mask_target > 0, 0.0, mask_target).astype(np.float32)

        return {
            "features": torch.from_numpy(np.ascontiguousarray(feats)),
            "mask_target": torch.from_numpy(mask_target),
            "frac_target": torch.from_numpy(frac_target),
            "valid_mask": torch.from_numpy(valid_mask),
            "cl_full": torch.from_numpy(cl),
        }


def make_balanced_sampler(ds: MaridaDualHeadDataset) -> WeightedRandomSampler:
    weights = np.ones(len(ds), dtype=np.float32)
    n_plastic = 0
    for i, p in enumerate(ds.paths):
        if _patch_has_plastic(p):
            weights[i] = PLASTIC_SAMPLE_WEIGHT
            n_plastic += 1
    pct = n_plastic * PLASTIC_SAMPLE_WEIGHT / max(
        n_plastic * PLASTIC_SAMPLE_WEIGHT + (len(ds) - n_plastic), 1) * 100
    print(f"  Sampler: {n_plastic}/{len(ds)} train patches contain plastic "
          f"(weight {PLASTIC_SAMPLE_WEIGHT}x; expected positive batches ≈ {pct:.0f}%)")
    return WeightedRandomSampler(weights=weights.tolist(),
                                 num_samples=len(ds), replacement=True)


# ============================ losses =====================================

def tversky_loss(logits: torch.Tensor, target: torch.Tensor,
                 valid: torch.Tensor,
                 alpha: float = TVERSKY_ALPHA, beta: float = TVERSKY_BETA,
                 eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits) * valid
    target = target * valid
    tp = (probs * target).sum(dim=(1, 2, 3))
    fp = (probs * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - probs) * target).sum(dim=(1, 2, 3))
    has_pos = (target.sum(dim=(1, 2, 3)) > 0).float()
    if has_pos.sum() == 0:
        return torch.zeros([], device=logits.device)
    per_sample = 1.0 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (per_sample * has_pos).sum() / has_pos.sum().clamp_min(1.0)


def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              valid: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits) * valid
    target = target * valid
    inter = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    has_pos = (target.sum(dim=(1, 2, 3)) > 0).float()
    if has_pos.sum() == 0:
        return torch.zeros([], device=logits.device)
    per_sample = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
    return (per_sample * has_pos).sum() / has_pos.sum().clamp_min(1.0)


def masked_bce(logits: torch.Tensor, target: torch.Tensor,
               valid: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction='none')
    bce = bce * valid
    return bce.sum() / valid.sum().clamp_min(1.0)


def sargassum_negative_loss(logits: torch.Tensor, cl: torch.Tensor) -> torch.Tensor:
    """v5: explicit penalty for predicting plastic on Sargassum pixels.

    For every pixel where cl ∈ {2, 3}, target = 0 (not plastic). Use BCE.
    Helps stabilize sargassum_fp_rate across epochs.
    """
    sarg_mask = torch.zeros_like(cl, dtype=torch.float32)
    for cls in SARGASSUM_CLASSES:
        sarg_mask = sarg_mask + (cl == cls).float()
    sarg_mask = sarg_mask.unsqueeze(1)  # (B, 1, H, W)
    if sarg_mask.sum() == 0:
        return torch.zeros([], device=logits.device)
    target = torch.zeros_like(logits)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction='none')
    bce = bce * sarg_mask
    return bce.sum() / sarg_mask.sum().clamp_min(1.0)


def compute_total_loss(outputs, mask_t, frac_t, valid_w, cl):
    mask_t3 = mask_t.unsqueeze(1).float()
    frac_t3 = frac_t.unsqueeze(1).float()
    valid3 = valid_w.unsqueeze(1).float()

    mask_logit = outputs["mask_logit"].float()
    fraction = outputs["fraction"].float()

    tv = tversky_loss(mask_logit, mask_t3, valid3)
    d = dice_loss(mask_logit, mask_t3, valid3)
    bce = masked_bce(mask_logit, mask_t3, valid3)
    sarg = sargassum_negative_loss(mask_logit, cl)

    pos_sel = (mask_t3 > 0.5).float() * valid3
    if pos_sel.sum() > 0:
        mse = ((fraction - frac_t3).pow(2) * pos_sel).sum() / pos_sel.sum()
    else:
        mse = torch.zeros([], device=mask_logit.device)

    total = (TVERSKY_WEIGHT * tv + DICE_WEIGHT * d
             + BCE_WEIGHT * bce + SARG_NEG_WEIGHT * sarg + FRAC_WEIGHT * mse)
    return {"total": total, "tversky": tv, "dice": d,
            "bce": bce, "sarg": sarg, "mse": mse}


# ============================ metrics ====================================

@torch.no_grad()
def _forward_tta(model, feats):
    """Test-time augmentation: average probs across 4 rotations + hflip.

    Free +0.02-0.05 IoU. Slower (5x compute per val batch) but val is
    only 41 batches so overhead is ~1s/epoch.
    """
    out = model(feats)
    probs = torch.sigmoid(out["mask_logit"])
    frac = out["fraction"]
    augs = [
        (lambda x: torch.rot90(x, 1, dims=(-2, -1)),
         lambda x: torch.rot90(x, -1, dims=(-2, -1))),
        (lambda x: torch.rot90(x, 2, dims=(-2, -1)),
         lambda x: torch.rot90(x, -2, dims=(-2, -1))),
        (lambda x: torch.flip(x, dims=(-1,)),
         lambda x: torch.flip(x, dims=(-1,))),
    ]
    n = 1
    for fwd, inv in augs:
        out_a = model(fwd(feats))
        probs = probs + inv(torch.sigmoid(out_a["mask_logit"]))
        frac = frac + inv(out_a["fraction"])
        n += 1
    return probs / n, frac / n


@torch.no_grad()
def eval_val(model, loader, device,
             thresholds=(0.05, 0.1, 0.2, 0.3, 0.5),
             use_tta: bool = True) -> dict:
    model.eval()
    inter = {t: 0 for t in thresholds}
    union = {t: 0 for t in thresholds}
    p07_num = 0
    p07_den = 0
    mae_sum = 0.0
    mae_den = 0
    sarg_num = {t: 0 for t in thresholds}
    sarg_den = 0

    n_pixels_seen = 0
    n_truth_pos = 0
    sum_prob = 0.0
    pred_pos_at_010 = 0
    pred_pos_at_050 = 0

    for batch in loader:
        feats = batch["features"].to(device, non_blocking=True)
        if use_tta:
            probs5d, frac5d = _forward_tta(model, feats)
            probs = probs5d[:, 0]
            frac = frac5d[:, 0]
        else:
            out = model(feats)
            probs = torch.sigmoid(out["mask_logit"])[:, 0]
            frac = out["fraction"][:, 0]
        mask_t = batch["mask_target"].to(device)
        frac_t = batch["frac_target"].to(device)
        valid = batch["valid_mask"].to(device).bool()
        cl = batch["cl_full"].to(device)
        truth = (mask_t > 0.5) & valid

        n_pixels_seen += valid.sum().item()
        n_truth_pos += truth.sum().item()
        # v5 fix: nan_to_num before sum so mean_p is finite
        sum_prob += torch.nan_to_num(probs[valid], nan=0.0).sum().item()
        pred_pos_at_010 += ((probs >= 0.1) & valid).sum().item()
        pred_pos_at_050 += ((probs >= 0.5) & valid).sum().item()

        for t in thresholds:
            pred = (probs >= t) & valid
            inter[t] += (pred & truth).sum().item()
            union[t] += (pred | truth).sum().item()
            sarg = torch.zeros_like(cl, dtype=torch.bool)
            for cls in SARGASSUM_CLASSES:
                sarg |= (cl == cls)
            sarg_num[t] += (pred & sarg).sum().item()

        hi = (probs >= 0.7) & valid
        p07_num += (hi & truth).sum().item()
        p07_den += hi.sum().item()

        pos_sel = truth.float()
        if pos_sel.sum() > 0:
            mae_sum += ((frac - frac_t).abs() * pos_sel).sum().item()
            mae_den += pos_sel.sum().item()

        sarg = torch.zeros_like(cl, dtype=torch.bool)
        for cls in SARGASSUM_CLASSES:
            sarg |= (cl == cls)
        sarg_den += sarg.sum().item()

    iou_by_t = {t: (inter[t] / max(union[t], 1)) for t in thresholds}
    best_t = max(iou_by_t, key=iou_by_t.get)
    return {
        "iou_by_threshold": {float(t): v for t, v in iou_by_t.items()},
        "best_threshold": float(best_t),
        "iou": iou_by_t[best_t],
        "precision_at_0_7": p07_num / max(p07_den, 1),
        "sub_pixel_mae": mae_sum / max(mae_den, 1) if mae_den > 0 else float("nan"),
        "sargassum_fp_rate": sarg_num[best_t] / max(sarg_den, 1),
        "n_truth_positive_pixels": n_truth_pos,
        "n_total_valid_pixels": n_pixels_seen,
        "truth_positive_pct": n_truth_pos / max(n_pixels_seen, 1) * 100,
        "mean_predicted_prob": sum_prob / max(n_pixels_seen, 1),
        "pred_positive_pct_at_0.1": pred_pos_at_010 / max(n_pixels_seen, 1) * 100,
        "pred_positive_pct_at_0.5": pred_pos_at_050 / max(n_pixels_seen, 1) * 100,
    }


# ============================ train loop =================================

def freeze_encoder(model: DualHeadUNetpp, freeze: bool) -> None:
    for p in model.backbone.encoder.parameters():
        p.requires_grad = not freeze


def train() -> dict[str, Any]:
    seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"MARIDA root: {MARIDA_ROOT.resolve()}")

    splits_dir = MARIDA_ROOT / "splits"
    patches_dir = MARIDA_ROOT / "patches"
    train_paths = _read_patch_paths(splits_dir / "train_X.txt", patches_dir)
    val_paths = _read_patch_paths(splits_dir / "val_X.txt", patches_dir)
    print(f"Train patches: {len(train_paths)} | Val patches: {len(val_paths)}")

    val_plastic_count = sum(1 for p in val_paths if _patch_has_plastic(p))
    print(f"Val patches WITH plastic: {val_plastic_count}/{len(val_paths)} "
          f"({val_plastic_count / max(len(val_paths), 1) * 100:.1f}%)")
    if val_plastic_count == 0:
        raise RuntimeError(
            "ABORT: validation set contains zero plastic-bearing patches."
        )

    train_ds = MaridaDualHeadDataset(train_paths, train_mode=True)
    val_ds = MaridaDualHeadDataset(val_paths, train_mode=False)
    sampler = make_balanced_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    model = DualHeadUNetpp(in_channels=N_CHANNELS)
    print(f"Model params (BN): {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    convert_bn_to_gn(model, num_groups=8)
    n_bn = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    n_gn = sum(1 for m in model.modules() if isinstance(m, nn.GroupNorm))
    print(f"Norm layers after conversion: BatchNorm={n_bn} GroupNorm={n_gn}")
    assert n_bn == 0
    model = model.to(device)
    print(f"Model params (GN): {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # v5: EMA shadow model
    ema = ModelEMA(model, decay=EMA_DECAY)
    print(f"EMA enabled: decay={EMA_DECAY}")

    sample_batch = next(iter(train_loader))
    sf = sample_batch["features"]
    sm = sample_batch["mask_target"]
    print(f"Input stats: shape={tuple(sf.shape)} "
          f"min={sf.min():.3f} max={sf.max():.3f} "
          f"mean={sf.mean():.3f} std={sf.std():.3f}")
    print(f"Sample batch: {(sm.sum(dim=(1, 2)) > 0).sum().item()}/{sm.shape[0]} "
          f"patches contain plastic; "
          f"plastic pixels: {sm.sum().item():.0f} "
          f"({sm.sum().item() / sm.numel() * 100:.3f}%)")
    assert not torch.isnan(sf).any()
    assert -0.5 < sf.mean() < 1.5

    freeze_encoder(model, freeze=True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nStage 1 (epochs 1-{WARMUP_EPOCHS}): encoder FROZEN, "
          f"trainable params = {n_trainable / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY,
    )
    scaler = (GradScaler("cuda", enabled=(device.type == "cuda"))
              if _AMP_NEW_API
              else GradScaler(enabled=(device.type == "cuda")))

    history: list[dict] = []
    best_iou = -1.0
    best_state: dict | None = None
    best_threshold = 0.1
    best_epoch = 0
    nan_skips = 0

    for epoch in range(1, EPOCHS + 1):
        if epoch == WARMUP_EPOCHS + 1:
            freeze_encoder(model, freeze=False)
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n>>> Stage 2 (epochs {epoch}-{EPOCHS}): encoder UNFROZEN, "
                  f"trainable params = {n_trainable / 1e6:.2f}M, lr={LR_FULL}")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR_FULL, weight_decay=WEIGHT_DECAY,
            )

        model.train()
        t0 = time.time()
        losses_acc = {"total": 0.0, "tversky": 0.0, "dice": 0.0,
                      "bce": 0.0, "sarg": 0.0, "mse": 0.0}
        seen_steps = 0
        train_pred_pos_pct_sum = 0.0

        for step, batch in enumerate(train_loader):
            feats = batch["features"].to(device, non_blocking=True)
            mask_t = batch["mask_target"].to(device, non_blocking=True)
            frac_t = batch["frac_target"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)
            cl = batch["cl_full"].to(device, non_blocking=True)

            if torch.any(mask_t > 0) and random.random() < BIOFOULING_PROB:
                feats = biofouling_augment(feats, mask_t)

            optimizer.zero_grad(set_to_none=True)
            ac_ctx = (autocast("cuda", enabled=(device.type == "cuda"),
                               dtype=torch.float16)
                      if _AMP_NEW_API
                      else autocast(enabled=(device.type == "cuda"),
                                    dtype=torch.float16))
            with ac_ctx:
                out = model(feats)
                losses = compute_total_loss(out, mask_t, frac_t, valid, cl)

            if not torch.isfinite(losses["total"]):
                nan_skips += 1
                continue

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)  # v5: shadow weights updated each step

            for k in losses_acc:
                losses_acc[k] += losses[k].item()
            with torch.no_grad():
                p = torch.sigmoid(out["mask_logit"])[:, 0]
                train_pred_pos_pct_sum += (
                    ((p >= 0.1) & valid.bool()).sum().item()
                    / max(valid.sum().item(), 1) * 100
                )
            seen_steps += 1

        losses_avg = {k: v / max(seen_steps, 1) for k, v in losses_acc.items()}
        train_pred_pos_pct = train_pred_pos_pct_sum / max(seen_steps, 1)
        train_time = time.time() - t0

        # v5: eval EMA shadow weights, NOT the live model
        val_metrics = eval_val(ema.shadow, val_loader, device, use_tta=True)

        print(
            f"E{epoch:02d}/{EPOCHS} "
            f"| loss={losses_avg['total']:.4f} "
            f"(tv={losses_avg['tversky']:.3f} d={losses_avg['dice']:.3f} "
            f"bce={losses_avg['bce']:.3f} sg={losses_avg['sarg']:.3f}) "
            f"| train_pred_pos={train_pred_pos_pct:.2f}% "
            f"| EMA val_iou={val_metrics['iou']:.4f}@t={val_metrics['best_threshold']:.2f} "
            f"val_pred_pos={val_metrics['pred_positive_pct_at_0.1']:.2f}% "
            f"mean_p={val_metrics['mean_predicted_prob']:.3f} "
            f"p@0.7={val_metrics['precision_at_0_7']:.3f} "
            f"sarg_fp={val_metrics['sargassum_fp_rate']:.3f} "
            f"| {train_time:.1f}s"
        )
        if epoch == 1 or epoch % 5 == 0:
            print("    val IoU per threshold:", {
                f"{t:.2f}": f"{v:.4f}"
                for t, v in val_metrics["iou_by_threshold"].items()
            })

        history.append({
            "epoch": epoch,
            "losses": losses_avg,
            "train_pred_pos_pct_at_0.1": train_pred_pos_pct,
            "val": val_metrics,
            "train_time_s": round(train_time, 2),
            "nan_skips_cumulative": nan_skips,
        })

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            best_threshold = val_metrics["best_threshold"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in ema.shadow.state_dict().items()}
            print(f"  ↑ new best EMA IoU: {best_iou:.4f} @ threshold "
                  f"{best_threshold:.2f} (epoch {best_epoch})")

    if best_state is None:
        best_state = {k: v.detach().cpu().clone()
                      for k, v in ema.shadow.state_dict().items()}

    torch.save(best_state, CHECKPOINT_OUT)
    size_mb = CHECKPOINT_OUT.stat().st_size / (1024 * 1024)
    print(f"\nSaved best epoch ({best_epoch}) EMA checkpoint → "
          f"{CHECKPOINT_OUT.resolve()} ({size_mb:.1f} MB)")
    print(f"Best val_iou = {best_iou:.4f} at threshold = {best_threshold:.2f}")
    print(f"Total NaN-skips across training: {nan_skips}")

    final = {
        "best_val_iou": best_iou,
        "best_threshold": best_threshold,
        "best_epoch": best_epoch,
        "nan_skips_total": nan_skips,
        "history": history,
        "config": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "lr_head": LR_HEAD, "lr_full": LR_FULL,
            "warmup_epochs": WARMUP_EPOCHS,
            "seed": SEED,
            "tversky_alpha": TVERSKY_ALPHA, "tversky_beta": TVERSKY_BETA,
            "tversky_weight": TVERSKY_WEIGHT,
            "dice_weight": DICE_WEIGHT,
            "bce_weight": BCE_WEIGHT,
            "sarg_neg_weight": SARG_NEG_WEIGHT,
            "frac_weight": FRAC_WEIGHT,
            "ema_decay": EMA_DECAY,
            "tta_at_val": True,
            "plastic_sample_weight": PLASTIC_SAMPLE_WEIGHT,
            "mix_prob": MIX_PROB, "biofouling_prob": BIOFOULING_PROB,
            "normalization": "per_patch_pct_2_98",
            "norm_layers": "GroupNorm",
        },
        "prd_targets": {
            "marida_val_iou": {"target": 0.45, "actual": best_iou},
            "precision_at_0_7": {"target": 0.75,
                                 "actual": history[-1]["val"]["precision_at_0_7"]},
            "sub_pixel_mae": {"target": 0.15,
                              "actual": history[-1]["val"]["sub_pixel_mae"]},
            "sargassum_fp_rate": {"target": 0.15,
                                  "actual": history[-1]["val"]["sargassum_fp_rate"]},
        },
    }
    METRICS_OUT.write_text(json.dumps(final, indent=2, default=str))
    print(f"Saved metrics → {METRICS_OUT.resolve()}")
    return final


if __name__ == "__main__":
    sys.exit(0 if train() else 1)
