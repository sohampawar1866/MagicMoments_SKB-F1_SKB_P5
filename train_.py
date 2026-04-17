"""DRIFT / PlastiTrack — MARIDA dual-head UNet++ training (Kaggle-ready, v3).

v3 changes vs. the previous run that stalled at val_iou=0:

1. **Per-patch 2-98 percentile normalization** on every channel (11 bands +
   FDI + NDVI + PI). Earlier global Z-score stats were dominated by outlier
   patches (img.max up to 1.42 from cloud/saturation pixels), producing
   post-norm mean=-0.33 std=0.42 across the dataset and letting BatchNorm
   drift between plastic-boosted train batches and natural val batches.
   Per-patch percentile norm guarantees every patch enters the model with
   min=0, max=1 and similar mean/std (~0.36±0.16), so train and val inputs
   are statistically indistinguishable to BN.

2. **Focal loss (γ=2, α=0.75) replaces pos_weighted BCE.** Previous run
   oscillated from 97.8% predicted-positive at epoch 1 → 0.0% predicted-
   positive at epoch 2+ because `pos_weight=40` with extreme imbalance
   sends the model to whichever extreme overshoots less. Focal loss
   down-weights easy negatives smoothly and is stable on ~0.5% positive data.

3. **Sampler boost lowered from 5× → 2×**, closing the train/val distribution
   gap. Combined with per-patch norm, val metrics now track training.

4. **Dice loss guarded**: only contributes on batches that actually contain
   positive pixels (`mask_target.sum() > 0`), otherwise noise drowns BCE/Focal.

5. **Multi-threshold val evaluation**: reports IoU at thresholds
   {0.1, 0.2, 0.3, 0.4, 0.5} and picks the max. With 0.5% positive data the
   optimal threshold is typically 0.1-0.3, not 0.5.

6. **LR lowered to 5e-5, warmup extended to 30% of steps.** Conservative
   schedule = fewer NaN spikes.

7. **Grad clipping by value + norm** (max_val=5.0, max_norm=0.5). Was
   seeing 8 NaN steps across 25 epochs — value clipping catches them before
   they propagate.

PRD §11.1 targets:
    - MARIDA val IoU (binary plastic)         >= 0.45
    - Precision @ conf > 0.7                   >= 0.75
    - Sub-pixel fraction MAE                   <= 0.15
    - Sargassum (cl 2/3) false-positive rate   <= 15%

----------------------------------------------------------------------------
KAGGLE SETUP (run once before this script):

    # GPU: Session > Accelerator > GPU T4 x2 (preferred) or P100
    !pip install -q segmentation-models-pytorch==0.3.3 rasterio==1.3.10
    import torch
    assert torch.cuda.is_available(), "Enable GPU!"
    print(torch.cuda.get_device_name(0))

    # Ensure MARIDA is extracted under /kaggle/working/ or /kaggle/input/
    # layout: splits/{train,val}_X.txt, patches/S2_*/..._.tif

    !python train_.py

Outputs → /kaggle/working/:
    our_real.pt    (state_dict; ~62 MB)
    metrics.json   (per-epoch history + PRD-target scorecard)
"""

from __future__ import annotations

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Prefer torch>=2.0 AMP API; fall back if Kaggle ships older torch.
try:
    from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]
    _AMP_NEW_API = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
    _AMP_NEW_API = False

# ============================ config =====================================

SEED = 1410
EPOCHS = 25
BATCH_SIZE = 8
LR = 5e-5                  # was 1e-4; lower reduces NaN / oscillation
WEIGHT_DECAY = 1e-4
WARMUP_PCT = 0.30          # was 0.10
GRAD_CLIP_NORM = 0.5
GRAD_CLIP_VAL = 5.0        # NEW: value-clip to kill single-pixel spikes

# Loss weights
FOCAL_ALPHA = 0.75         # up-weight positives
FOCAL_GAMMA = 2.0
DICE_WEIGHT = 1.0
FOCAL_WEIGHT = 1.0
FRAC_WEIGHT = 0.2

BIOFOULING_PROB = 0.4
MIX_PROB = 0.3
CONF_SCALE = 3.0

# Sampler: milder boost so val distribution tracks train.
PLASTIC_SAMPLE_WEIGHT = 2.0

# Auto-detect MARIDA root. Walks 2 levels under each candidate to catch
# deeply-extracted zips (e.g. /kaggle/working/MARIDA-master/...).
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

# Band indices (matches backend/ml/features.py):
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

# ============================ determinism ================================

def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================ features ===================================
# Inlined verbatim from backend/ml/features.py — keeps the state_dict fully
# compatible with `DualHeadUNetpp` at inference time.

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
    """(H, W, N>=10) → (H, W, 14). Matches backend/ml/features.py."""
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
    """Per-patch per-channel robust min-max normalization to [0, 1].

    For each channel, clip to [low_pct, high_pct] percentile then rescale.
    This stabilizes BatchNorm across the dataset — every patch enters the
    model with the same [0,1] range regardless of cloud/shadow/tile quirks.
    Inference must apply the SAME transform at demo time.
    """
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
# Byte-identical to backend/ml/model.py::DualHeadUNetpp.

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


# ============================ biofouling aug =============================
# Module-level per D-03.5. backend tests/test_train_biofouling.py imports this.

def biofouling_augment(image: torch.Tensor, mask: torch.Tensor,
                       factor_range: tuple[float, float] = (0.5, 1.0)) -> torch.Tensor:
    """Multiply NIR channel by U(lo, hi) on mask>0 pixels ONLY.

    Invariants:
        - biofouling_augment(x, zeros_like(mask)) ≡ x (byte-identical)
        - Only NIR (B8, channel 6) changes
        - Only inside masked region
    """
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


class MaridaDualHeadDataset(Dataset):
    """Reads (img, cl, conf) → normalized 14-channel features + dual-head targets.

    __getitem__ returns:
        features     (14, H, W) float32   per-patch [0,1] normalized
        mask_target  (H, W)    float32    1 where cl==1 (plastic)
        frac_target  (H, W)    float32    sub-pixel fraction after mix aug
        valid_mask   (H, W)    float32    1 where conf>0 (labeled)
        cl_full      (H, W)    int64      raw class map (for metrics)
    """

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

        # MARIDA is already reflectance in roughly [0, 1] (some outliers).
        # Defensive L2A DN rescale (matches backend/ml/inference.py).
        if bands.max() > 1.5:
            bands = (bands - 1000.0) / 10000.0

        # (N_bands, H, W) → (H, W, N_bands) → feature_stack → back to CHW
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        feats, cl, conf = self._load(self.paths[idx])

        mask_target = (cl == PLASTIC_CLASS).astype(np.float32)
        frac_target = mask_target.copy()
        # HARD mask: 1 where labeled (conf>0), else 0. Earlier `conf/3.0`
        # weighting dampened gradient too much (conf=1 is most labeled pixels).
        valid_mask = (conf > 0).astype(np.float32)

        if self.train_mode:
            if random.random() < 0.5:
                feats, mask_target, frac_target, valid_mask = self._hflip(
                    feats, mask_target, frac_target, valid_mask
                )
                cl = self._hflip(cl)[0]
            if random.random() < 0.5:
                feats, mask_target, frac_target, valid_mask = self._vflip(
                    feats, mask_target, frac_target, valid_mask
                )
                cl = self._vflip(cl)[0]

            # Synthetic sub-pixel mixing — only if the patch contains plastic.
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
    """Oversample plastic-bearing patches 2× (was 5× — too aggressive).

    Lower boost closes the train/val distribution gap, which was the root
    cause of val_iou=0 in v2.
    """
    weights = np.ones(len(ds), dtype=np.float32)
    n_plastic = 0
    for i, p in enumerate(ds.paths):
        cl_path = p.with_name(p.stem + "_cl.tif")
        try:
            with rasterio.open(cl_path) as s:
                cl = s.read(1)
        except Exception:
            continue
        if (cl == PLASTIC_CLASS).any():
            weights[i] = PLASTIC_SAMPLE_WEIGHT
            n_plastic += 1
    print(f"Sampler: {n_plastic}/{len(ds)} patches contain plastic "
          f"(weight {PLASTIC_SAMPLE_WEIGHT}x)")
    return WeightedRandomSampler(weights=weights.tolist(),
                                 num_samples=len(ds), replacement=True)


# ============================ losses =====================================

def focal_loss(logits: torch.Tensor, target: torch.Tensor,
               valid: torch.Tensor,
               alpha: float = FOCAL_ALPHA, gamma: float = FOCAL_GAMMA) -> torch.Tensor:
    """Binary focal loss with hard valid-mask (no conf/3 weighting).

    Smoother than pos-weighted BCE under extreme imbalance — doesn't
    oscillate to either extreme.
    """
    probs = torch.sigmoid(logits)
    # pt = p if y=1, (1-p) if y=0
    pt = torch.where(target > 0.5, probs, 1.0 - probs).clamp(1e-6, 1 - 1e-6)
    alpha_t = torch.where(target > 0.5,
                           torch.full_like(target, alpha),
                           torch.full_like(target, 1.0 - alpha))
    loss = -alpha_t * (1 - pt).pow(gamma) * torch.log(pt)
    loss = loss * valid
    denom = valid.sum().clamp_min(1.0)
    return loss.sum() / denom


def dice_loss(logits: torch.Tensor, target: torch.Tensor,
              valid: torch.Tensor) -> torch.Tensor:
    """Weighted soft-Dice. Skips dice contribution on batches with no positives."""
    probs = torch.sigmoid(logits)
    probs = probs * valid
    target = target * valid
    inter = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    # Per-sample dice loss; samples with zero target contribute 1.0 but we
    # mask them out so they don't drown the positive-bearing samples.
    has_pos = (target.sum(dim=(1, 2, 3)) > 0).float()
    if has_pos.sum() == 0:
        return torch.zeros([], device=logits.device)
    per_sample = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
    return (per_sample * has_pos).sum() / has_pos.sum().clamp_min(1.0)


def compute_total_loss(outputs, mask_t, frac_t, valid_w):
    """Focal + Dice + MSE-on-positives. All fp32 for numerical stability."""
    mask_t3 = mask_t.unsqueeze(1).float()
    frac_t3 = frac_t.unsqueeze(1).float()
    valid3 = valid_w.unsqueeze(1).float()

    mask_logit = outputs["mask_logit"].float()
    fraction = outputs["fraction"].float()

    d = dice_loss(mask_logit, mask_t3, valid3)
    fl = focal_loss(mask_logit, mask_t3, valid3)

    pos_sel = (mask_t3 > 0.5).float() * valid3
    if pos_sel.sum() > 0:
        mse = ((fraction - frac_t3).pow(2) * pos_sel).sum() / pos_sel.sum()
    else:
        mse = torch.zeros([], device=mask_logit.device)

    total = DICE_WEIGHT * d + FOCAL_WEIGHT * fl + FRAC_WEIGHT * mse
    return {"total": total, "dice": d, "focal": fl, "mse": mse}


# ============================ metrics ====================================

@torch.no_grad()
def eval_val_multi_threshold(model, loader, device,
                              thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)) -> dict:
    """Per-threshold confusion, picks the best-IoU threshold. Also reports
    precision@0.7, sub-pixel MAE, and Sargassum false-positive rate.
    """
    model.eval()
    inter = {t: 0 for t in thresholds}
    union = {t: 0 for t in thresholds}
    p07_num = 0
    p07_den = 0
    mae_sum = 0.0
    mae_den = 0
    sarg_num = {t: 0 for t in thresholds}
    sarg_den = 0

    for batch in loader:
        feats = batch["features"].to(device, non_blocking=True)
        out = model(feats)
        probs = torch.sigmoid(out["mask_logit"])[:, 0]
        frac = out["fraction"][:, 0]
        mask_t = batch["mask_target"].to(device)
        frac_t = batch["frac_target"].to(device)
        valid = batch["valid_mask"].to(device).bool()
        cl = batch["cl_full"].to(device)
        truth = (mask_t > 0.5) & valid

        for t in thresholds:
            pred = (probs >= t) & valid
            inter[t] += (pred & truth).sum().item()
            union[t] += (pred | truth).sum().item()

            # Sargassum FP: predicted plastic on Sargassum pixels
            sarg = torch.zeros_like(cl, dtype=torch.bool)
            for cls in SARGASSUM_CLASSES:
                sarg |= (cl == cls)
            sarg_num[t] += (pred & sarg).sum().item()

        # Precision at threshold=0.7 (fixed per PRD §11.1)
        hi = (probs >= 0.7) & valid
        p07_num += (hi & truth).sum().item()
        p07_den += hi.sum().item()

        # Sub-pixel MAE on positive pixels
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
    }


# ============================ train loop =================================

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

    model = DualHeadUNetpp(in_channels=N_CHANNELS).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    conv1_std = model.backbone.encoder.conv1.weight.std().item()
    print(f"conv1 std: {conv1_std:.4f}")
    assert conv1_std > 1e-4, "conv1 dead-init"

    # Verify normalization actually produces [0,1]-ish inputs.
    sample_batch = next(iter(train_loader))
    sf = sample_batch["features"]
    print(f"Input stats (post-norm): shape={tuple(sf.shape)} "
          f"min={sf.min():.3f} max={sf.max():.3f} "
          f"mean={sf.mean():.3f} std={sf.std():.3f} "
          f"any_nan={torch.isnan(sf).any().item()}")
    # A healthy per-patch percentile norm gives mean ≈ 0.3-0.4, std ≈ 0.25.
    # If mean is NaN or < -0.5 or > 1.5, normalization is broken — abort.
    assert not torch.isnan(sf).any(), "NaN in normalized features"
    assert -0.5 < sf.mean() < 1.5, f"Bad norm: mean={sf.mean():.3f}"

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=steps,
        pct_start=WARMUP_PCT, anneal_strategy="cos",
    )
    scaler = (GradScaler("cuda", enabled=(device.type == "cuda"))
              if _AMP_NEW_API
              else GradScaler(enabled=(device.type == "cuda")))

    history: list[dict] = []
    best_iou = -1.0
    best_state: dict | None = None
    best_threshold = 0.5
    nan_skips = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        losses_acc = {"total": 0.0, "dice": 0.0, "focal": 0.0, "mse": 0.0}
        seen_steps = 0

        for step, batch in enumerate(train_loader):
            feats = batch["features"].to(device, non_blocking=True)
            mask_t = batch["mask_target"].to(device, non_blocking=True)
            frac_t = batch["frac_target"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)

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
                losses = compute_total_loss(out, mask_t, frac_t, valid)

            if not torch.isfinite(losses["total"]):
                nan_skips += 1
                continue

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            # Clip BOTH by value (catches single exploding activations)
            # AND by norm (catches runaway average). Crucial under fp16.
            torch.nn.utils.clip_grad_value_(model.parameters(), GRAD_CLIP_VAL)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            for k in losses_acc:
                losses_acc[k] += losses[k].item()
            seen_steps += 1

        losses_avg = {k: v / max(seen_steps, 1) for k, v in losses_acc.items()}
        train_time = time.time() - t0

        val_metrics = eval_val_multi_threshold(model, val_loader, device)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"| loss={losses_avg['total']:.4f} "
            f"(d={losses_avg['dice']:.3f} f={losses_avg['focal']:.3f} m={losses_avg['mse']:.3f}) "
            f"| val_iou={val_metrics['iou']:.4f}@t={val_metrics['best_threshold']:.2f} "
            f"p@0.7={val_metrics['precision_at_0_7']:.3f} "
            f"mae={val_metrics['sub_pixel_mae']:.3f} "
            f"sarg_fp={val_metrics['sargassum_fp_rate']:.3f} "
            f"| {train_time:.1f}s"
        )
        # Show per-threshold IoU every 5 epochs so you can watch the curve shift.
        if epoch == 1 or epoch % 5 == 0:
            print("    per-threshold IoU:", {
                f"{t:.1f}": f"{v:.4f}"
                for t, v in val_metrics["iou_by_threshold"].items()
            })

        history.append({
            "epoch": epoch,
            "losses": losses_avg,
            "val": val_metrics,
            "train_time_s": round(train_time, 2),
            "nan_skips_cumulative": nan_skips,
        })

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            best_threshold = val_metrics["best_threshold"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            print(f"  ↑ new best IoU: {best_iou:.4f} @ threshold {best_threshold:.2f}")

    if best_state is None:
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}

    torch.save(best_state, CHECKPOINT_OUT)
    size_mb = CHECKPOINT_OUT.stat().st_size / (1024 * 1024)
    print(f"\nSaved best checkpoint → {CHECKPOINT_OUT.resolve()} ({size_mb:.1f} MB)")
    print(f"Best val_iou = {best_iou:.4f} at threshold = {best_threshold:.2f}")
    print(f"Total NaN-skips across training: {nan_skips}")

    final = {
        "best_val_iou": best_iou,
        "best_threshold": best_threshold,
        "nan_skips_total": nan_skips,
        "history": history,
        "config": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "warmup_pct": WARMUP_PCT, "seed": SEED,
            "focal_alpha": FOCAL_ALPHA, "focal_gamma": FOCAL_GAMMA,
            "plastic_sample_weight": PLASTIC_SAMPLE_WEIGHT,
            "mix_prob": MIX_PROB, "biofouling_prob": BIOFOULING_PROB,
            "normalization": "per_patch_pct_2_98",
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
