import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Iterable
from scipy.special import gamma
import lmoments3 as lm

# ---------- Settings ----------
MIN_YEARS: int = 5
N_LMOMENTS: int = 3
NO_DATA_VALUE: float = 0.0
DEFAULT_CRS: str = "EPSG:4326"
# -------------------------------

@dataclass
class GEVParams:
    """Container for GEV parameters (μ, σ, ξ)."""
    theta1: np.ndarray  # location μ
    theta2: np.ndarray  # scale σ (>0)
    theta3: np.ndarray  # shape ξ

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {"theta1": self.theta1, "theta2": self.theta2, "theta3": self.theta3}

# ---- Core formulas ----
def _gev_from_lmoments(l1: np.ndarray, l2: np.ndarray, tau3: np.ndarray) -> GEVParams:
    """Hosking (1990) approximation: (L1,L2,τ3) -> (μ,σ,ξ)."""
    c = (2.0 / (3.0 + tau3)) - (np.log(2.0) / np.log(3.0))
    xi = 7.8590 * c + 2.9554 * (c ** 2)
    sigma = (l2 * xi) / ((1.0 - 2.0 ** (-xi)) * gamma(1.0 + xi))
    mu = l1 - (sigma / xi) * (1.0 - gamma(1.0 + xi))
    return GEVParams(theta1=mu, theta2=sigma, theta3=xi)

def quantile_from_gev(T: np.ndarray, params: GEVParams) -> np.ndarray:
    """Return design quantiles Q_T for return periods T."""
    T = np.asarray(T, dtype=float)
    P = 1.0 - (1.0 / T)
    xi, mu, sigma = params.theta3, params.theta1, params.theta2
    return np.where(
        np.isclose(xi, 0.0),
        mu - sigma * np.log(-np.log(P)),
        mu + (sigma / xi) * (1.0 - (-np.log(P)) ** xi),
    )

def return_period_from_gev(Q: np.ndarray, params: GEVParams) -> np.ndarray:
    """Return period T for observed/design value Q."""
    xi, mu, sigma = params.theta3, params.theta1, params.theta2
    inner = 1.0 - xi * ((Q - mu) / sigma)
    inner = np.maximum(inner, 0.0)
    y = np.where(np.isclose(xi, 0.0), (Q - mu) / sigma, (-1.0 / xi) * np.log(inner))
    P = np.exp(-np.exp(-y))
    return 1.0 / (1.0 - P)

# ---- Fitting ----
def fit_gev_from_dataframe(df) -> Tuple[GEVParams, np.ndarray]:
    """Fit GEV from each column of a DataFrame of annual maxima."""
    import pandas as pd
    cols = list(df.columns)
    data = [np.asarray(df[c].dropna(), dtype=float) for c in cols]
    valid_counts = np.array([np.count_nonzero(np.isfinite(v)) for v in data])

    l1 = np.full(len(cols), np.nan)
    l2 = np.full(len(cols), np.nan)
    tau3 = np.full(len(cols), np.nan)
    for i, x in enumerate(data):
        if x.size >= MIN_YEARS:
            L = lm.lmom_ratios(x, nmom=N_LMOMENTS)
            l1[i], l2[i], tau3[i] = L[0], L[1], L[2]

    return _gev_from_lmoments(l1, l2, tau3), valid_counts

def fit_gev_from_annualmax_cube(annual_max, mask: Optional[np.ndarray]=None):
    """Fit GEV from a cube (time, y, x) of annual maxima."""
    arr = np.asarray(annual_max, dtype=float)
    valid_counts = np.sum(np.isfinite(arr), axis=0)

    L = np.apply_along_axis(
        lambda v: lm.lmom_ratios(v[np.isfinite(v)], nmom=N_LMOMENTS)
        if np.count_nonzero(np.isfinite(v)) >= MIN_YEARS
        else np.array([np.nan, np.nan, np.nan]),
        axis=0, arr=arr
    )
    l1, l2, tau3 = L[0], L[1], L[2]
    if mask is not None:
        l1, l2, tau3 = np.where(mask, l1, np.nan), np.where(mask, l2, np.nan), np.where(mask, tau3, np.nan)

    return _gev_from_lmoments(l1, l2, tau3), valid_counts

# ---- Export GeoTIFF ----
def write_param_geotiffs(params: GEVParams, x: np.ndarray, y: np.ndarray,
                         out_folder: str, basename: str,
                         crs: str = DEFAULT_CRS, nodata: float = NO_DATA_VALUE):
    """Write μ/σ/ξ as GeoTIFFs using rioxarray."""
    import os, xarray as xr, rioxarray  # noqa: F401
    os.makedirs(out_folder, exist_ok=True)
    for key, arr in params.as_dict().items():
        da = xr.DataArray(arr.astype("float32"), dims=["y", "x"], coords={"y": y, "x": x})
        da = da.rio.write_crs(crs).rio.write_nodata(nodata)
        da.rio.to_raster(os.path.join(out_folder, f"{basename}_{key}.tif"))

# ---- Export table ----
def write_param_table(params: GEVParams,
                      series_names: Optional[Iterable[str]] = None,
                      outfile: Optional[str] = None,
                      fmt: str = "csv"):
    """Write μ/σ/ξ as a table (rows=params, cols=series)."""
    import pandas as pd
    mu, sigma, xi = params.theta1, params.theta2, params.theta3
    if mu.ndim != 1:
        raise ValueError("write_param_table expects 1D parameter arrays.")
    n = mu.size
    series_names = [f"series_{i}" for i in range(n)] if series_names is None else list(series_names)
    df = pd.DataFrame(np.vstack([mu, sigma, xi]),
                      index=["theta1", "theta2", "theta3"],
                      columns=series_names)
    if outfile:
        if fmt.lower() == "csv":
            df.to_csv(outfile)
        elif fmt.lower() == "parquet":
            df.to_parquet(outfile)
    return df
