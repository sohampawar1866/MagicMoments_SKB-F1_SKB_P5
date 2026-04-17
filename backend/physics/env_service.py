"""Dynamic environmental data ingestion (OceanTrace).

Bbox-aware fetcher for CMEMS currents + chlorophyll, ERA5 winds + SST.
Disk-caches NetCDF subsets keyed by hash(bbox + t0 + horizon) so repeated
forecasts for the same AOI don't re-hit the upstream APIs.

Auth: env vars `COPERNICUSMARINE_USERNAME`, `COPERNICUSMARINE_PASSWORD`,
`CDSAPI_URL`, `CDSAPI_KEY`. If unset, raises `EnvCredentialsMissing` —
service callers must catch and silently fall back to the prebaked static
NetCDFs already used by `env_data.load_env_stack`.

Returns an `EnvBundle` exposing the four xarray Datasets plus a
`.to_envstack()` adapter so the existing tracker code path needs no
edits — only `density_polygons`/landfall code that wants Chl/SST reaches
into the new fields.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import xarray as xr

from backend.physics.env_data import EnvStack, _finalize

logger = logging.getLogger(__name__)

CMEMS_CURRENTS_DATASET = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"
CMEMS_BGC_DATASET = "cmems_mod_glo_bgc_anfc_0.25deg_P1D-m"
ERA5_DATASET = "reanalysis-era5-single-levels"
ERA5_VARS = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "sea_surface_temperature",
)


class EnvCredentialsMissing(RuntimeError):
    """Raised when one of CMEMS/CDS credentials is unset.

    Caller (services layer) catches and silently falls back to prebaked
    static NetCDFs so the demo never crashes on missing keys.
    """


@dataclass
class EnvBundle:
    """Four xarray datasets bundled per-AOI per-time-window.

    `currents_ds` and `winds_ds` are compatible with the existing
    `EnvStack` shape via `.to_envstack()`. `sst_ds` and `chl_ds` are new
    — bio_fouling.py + dashboard endpoints reach into them via
    `.interp(longitude=..., latitude=..., time=...)`.
    """
    currents_ds: xr.Dataset
    winds_ds: xr.Dataset
    sst_ds: xr.Dataset
    chl_ds: xr.Dataset
    bbox: tuple[float, float, float, float]
    t0: str

    def to_envstack(self, horizon_hours: int = 72) -> EnvStack:
        return _finalize(self.currents_ds, self.winds_ds, horizon_hours)

    def sample_chl(self, lon: float, lat: float, t_iso: str | None = None) -> float:
        try:
            sel = {"longitude": lon, "latitude": lat}
            if t_iso is not None and "time" in self.chl_ds.coords:
                sel["time"] = t_iso
            v = self.chl_ds["chl"].interp(**sel, method="linear").values
            return float(v) if v == v else 0.3
        except Exception:
            return 0.3

    def sample_sst(self, lon: float, lat: float, t_iso: str | None = None) -> float:
        try:
            sel = {"longitude": lon, "latitude": lat}
            if t_iso is not None and "time" in self.sst_ds.coords:
                sel["time"] = t_iso
            v = self.sst_ds["sst"].interp(**sel, method="linear").values
            return float(v) - 273.15 if v > 100 else float(v)
        except Exception:
            return 20.0


def _cache_key(bbox: tuple[float, float, float, float], t0: str, horizon_days: int) -> str:
    raw = f"{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}_{t0}_{horizon_days}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _check_creds() -> None:
    missing = []
    if not os.environ.get("COPERNICUSMARINE_USERNAME"):
        missing.append("COPERNICUSMARINE_USERNAME")
    if not os.environ.get("COPERNICUSMARINE_PASSWORD"):
        missing.append("COPERNICUSMARINE_PASSWORD")
    if not os.environ.get("CDSAPI_KEY") and not Path.home().joinpath(".cdsapirc").exists():
        missing.append("CDSAPI_KEY (or ~/.cdsapirc)")
    if missing:
        raise EnvCredentialsMissing(
            f"missing env vars: {', '.join(missing)} — silent fallback expected"
        )


def _fetch_cmems_currents(bbox, t0_iso, t1_iso, out_path: Path) -> xr.Dataset:
    import copernicusmarine
    copernicusmarine.subset(
        dataset_id=CMEMS_CURRENTS_DATASET,
        variables=["uo", "vo"],
        minimum_longitude=bbox[0], maximum_longitude=bbox[2],
        minimum_latitude=bbox[1], maximum_latitude=bbox[3],
        start_datetime=t0_iso, end_datetime=t1_iso,
        minimum_depth=0, maximum_depth=1,
        output_filename=str(out_path), force_download=True,
    )
    return xr.open_dataset(out_path, decode_times=True)


def _fetch_cmems_chl(bbox, t0_iso, t1_iso, out_path: Path) -> xr.Dataset:
    import copernicusmarine
    copernicusmarine.subset(
        dataset_id=CMEMS_BGC_DATASET,
        variables=["chl"],
        minimum_longitude=bbox[0], maximum_longitude=bbox[2],
        minimum_latitude=bbox[1], maximum_latitude=bbox[3],
        start_datetime=t0_iso, end_datetime=t1_iso,
        minimum_depth=0, maximum_depth=1,
        output_filename=str(out_path), force_download=True,
    )
    return xr.open_dataset(out_path, decode_times=True)


def _fetch_era5(bbox, t0: datetime, horizon_hours: int, out_path: Path) -> xr.Dataset:
    import cdsapi
    days = []
    cur = t0
    end = t0 + timedelta(hours=horizon_hours)
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    c = cdsapi.Client()
    c.retrieve(
        ERA5_DATASET,
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": list(ERA5_VARS),
            "year": sorted({d.strftime("%Y") for d in days}),
            "month": sorted({d.strftime("%m") for d in days}),
            "day": sorted({d.strftime("%d") for d in days}),
            "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
            "area": [bbox[3], bbox[0], bbox[1], bbox[2]],
        },
        str(out_path),
    )
    ds = xr.open_dataset(out_path, decode_times=True)
    rename = {}
    if "u10" not in ds and "10m_u_component_of_wind" in ds:
        rename["10m_u_component_of_wind"] = "u10"
    if "v10" not in ds and "10m_v_component_of_wind" in ds:
        rename["10m_v_component_of_wind"] = "v10"
    if "sst" not in ds and "sea_surface_temperature" in ds:
        rename["sea_surface_temperature"] = "sst"
    if rename:
        ds = ds.rename(rename)
    return ds


def fetch_env_for_bbox(
    bbox: tuple[float, float, float, float],
    t0: str,
    horizon_days: int = 7,
    cache_dir: str | Path = "data/env_cache",
) -> EnvBundle:
    """Fetch (or load from cache) currents+chl+winds+sst for bbox over [t0, t0+horizon_days].

    bbox = (min_lon, min_lat, max_lon, max_lat). t0 ISO date string.
    Raises EnvCredentialsMissing if API credentials absent — caller is
    expected to silently fall back to prebaked static NetCDFs.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(bbox, t0, horizon_days)

    currents_p = cache_dir / f"{key}_currents.nc"
    chl_p = cache_dir / f"{key}_chl.nc"
    era5_p = cache_dir / f"{key}_era5.nc"

    cache_hit = currents_p.exists() and chl_p.exists() and era5_p.exists()
    if not cache_hit:
        _check_creds()

    t0_dt = datetime.fromisoformat(t0)
    t1_dt = t0_dt + timedelta(days=horizon_days)
    t0_iso = t0_dt.strftime("%Y-%m-%dT%H:%M:%S")
    t1_iso = t1_dt.strftime("%Y-%m-%dT%H:%M:%S")

    if cache_hit:
        logger.info("env_service: cache hit %s for bbox=%s t0=%s", key, bbox, t0)
        currents = xr.open_dataset(currents_p, decode_times=True)
        chl = xr.open_dataset(chl_p, decode_times=True)
        era5 = xr.open_dataset(era5_p, decode_times=True)
    else:
        logger.info("env_service: cache miss %s — fetching upstream", key)
        currents = _fetch_cmems_currents(bbox, t0_iso, t1_iso, currents_p)
        chl = _fetch_cmems_chl(bbox, t0_iso, t1_iso, chl_p)
        era5 = _fetch_era5(bbox, t0_dt, horizon_days * 24, era5_p)

    winds = era5[["u10", "v10"]] if "u10" in era5 else era5
    sst = era5[["sst"]] if "sst" in era5 else era5
    return EnvBundle(
        currents_ds=currents,
        winds_ds=winds,
        sst_ds=sst,
        chl_ds=chl,
        bbox=bbox,
        t0=t0,
    )
