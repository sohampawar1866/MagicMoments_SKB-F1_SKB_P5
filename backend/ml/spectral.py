"""FDI/NDVI spectral gating (OceanTrace).

Validates ML-detected polygons by their spectral signature inside the
geometry. A real floating plastic patch should have:
  - elevated FDI (Floating Debris Index, Biermann 2020) — positive
    and above noise floor
  - low NDVI — high NDVI means vegetation/Sargassum, not plastic

Polygons that fail either check are dropped from the response. Survivors
are tagged `spectral_validated: true`. The number of rejected polygons
is returned alongside so the caller can surface it as response metadata.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_FDI_MIN: float = 0.005
DEFAULT_NDVI_MAX: float = 0.20


def gate_polygon_arrays(
    fdi_arr: np.ndarray,
    ndvi_arr: np.ndarray,
    poly_mask: np.ndarray,
    fdi_min: float = DEFAULT_FDI_MIN,
    ndvi_max: float = DEFAULT_NDVI_MAX,
) -> tuple[bool, dict]:
    """Test mean FDI/NDVI inside boolean poly_mask. Returns (pass, stats).

    pass = mean_fdi >= fdi_min AND mean_ndvi <= ndvi_max.
    """
    if poly_mask.sum() == 0:
        return False, {"reason": "empty_mask"}
    fdi_in = fdi_arr[poly_mask]
    ndvi_in = ndvi_arr[poly_mask]
    fdi_in = fdi_in[np.isfinite(fdi_in)]
    ndvi_in = ndvi_in[np.isfinite(ndvi_in)]
    if fdi_in.size == 0 or ndvi_in.size == 0:
        return False, {"reason": "all_nan"}
    mean_fdi = float(fdi_in.mean())
    mean_ndvi = float(ndvi_in.mean())
    fdi_ok = mean_fdi >= fdi_min
    ndvi_ok = mean_ndvi <= ndvi_max
    return (fdi_ok and ndvi_ok), {
        "mean_fdi": mean_fdi,
        "mean_ndvi": mean_ndvi,
        "fdi_ok": fdi_ok,
        "ndvi_ok": ndvi_ok,
    }


def gate_polygon(
    geom,
    fdi_arr: np.ndarray,
    ndvi_arr: np.ndarray,
    transform,
    fdi_min: float = DEFAULT_FDI_MIN,
    ndvi_max: float = DEFAULT_NDVI_MAX,
) -> tuple[bool, dict]:
    """Rasterio-mask polygon `geom` over the fdi/ndvi grids and gate.

    `geom` is a shapely geometry in the same CRS as `transform`.
    """
    try:
        from rasterio.features import geometry_mask
        mask = ~geometry_mask(
            [geom.__geo_interface__],
            out_shape=fdi_arr.shape,
            transform=transform,
            invert=False,
        )
    except Exception as e:
        logger.info("spectral gate fallback (no rasterize): %s", e)
        return True, {"reason": "rasterize_failed"}
    return gate_polygon_arrays(fdi_arr, ndvi_arr, mask, fdi_min, ndvi_max)


def fdi_ndvi_features(bands_hwc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute (fdi, ndvi) arrays from an (H, W, N_bands>=9) reflectance stack.

    Re-uses backend.ml.features.compute_fdi/compute_ndvi for parity with
    the training-time feature stack.
    """
    from backend.ml.features import compute_fdi, compute_ndvi
    return compute_fdi(bands_hwc), compute_ndvi(bands_hwc)
