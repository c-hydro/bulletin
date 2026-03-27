import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr


class GridsHandler:
    """
    Utilities for gridded datasets (NetCDF) and point extraction.

    This module intentionally contains "data logic" (aggregation, decumulation, quantiles)
    and NOT file system I/O. File reading/writing stays in IOHandler.
    """

    @staticmethod
    def infer_coord_name(ds: xr.Dataset, candidates: list[str]) -> str | None:
        """
        Infer a coordinate/dimension name among candidates.
        """
        for name in candidates:
            if name in ds.coords:
                return name
            if name in ds.dims:
                return name
            if name in ds.variables:
                return name
        return None

    @staticmethod
    def infer_lat_lon_names(ds: xr.Dataset) -> tuple[str, str]:
        """
        Infer latitude and longitude names in a Dataset.
        """
        lat_name = GridsHandler.infer_coord_name(ds, ["lat", "latitude", "y", "YLAT", "nav_lat"])
        lon_name = GridsHandler.infer_coord_name(ds, ["lon", "longitude", "x", "XLONG", "nav_lon"])
        if lat_name is None or lon_name is None:
            raise ValueError(f"Could not infer lat/lon names. Found coords: {list(ds.coords)}")
        return lat_name, lon_name

    @staticmethod
    def infer_time_step_hours(time_index: pd.DatetimeIndex) -> float:
        """
        Infer timestep (hours) from a time index.
        """
        if len(time_index) < 2:
            raise ValueError("Need at least 2 time steps to infer timestep.")
        diffs = (time_index[1:] - time_index[:-1]).total_seconds() / 3600.0
        return float(np.median(diffs))

    @staticmethod
    def decumulate_timeseries(ts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert cumulative rainfall (mm) to incremental rainfall (mm/step).

        Uses diff and clips negatives to 0.
        """
        ts = ts_df.diff()
        ts = ts.clip(lower=0.0)
        return ts

    @staticmethod
    def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
        """
        Weighted quantile for 1D arrays (no interpolation).

        Returns the first value where cumulative weight >= quantile * total_weight.
        """
        if values.size == 0:
            return float("nan")

        q = float(quantile)
        if q <= 0.0:
            return float(np.nanmin(values))
        if q >= 1.0:
            return float(np.nanmax(values))

        v = np.asarray(values, dtype=float)
        w = np.asarray(weights, dtype=float)

        mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
        v = v[mask]
        w = w[mask]
        if v.size == 0:
            return float("nan")

        order = np.argsort(v)
        v_sorted = v[order]
        w_sorted = w[order]

        cum = np.cumsum(w_sorted)
        cutoff = q * float(cum[-1])

        idx = int(np.searchsorted(cum, cutoff, side="left"))
        idx = min(max(idx, 0), v_sorted.size - 1)
        return float(v_sorted[idx])

    @staticmethod
    def get_3x3_indices_1d(coord_values: np.ndarray, target: float) -> tuple[int, int]:
        """
        Get nearest index on 1D coordinate array, and return (start, end) slice indices
        for a radius-1 neighborhood. end is exclusive.
        """
        arr = np.asarray(coord_values, dtype=float)
        i = int(np.argmin(np.abs(arr - float(target))))
        i0 = max(i - 1, 0)
        i1 = min(i + 2, arr.size)  # exclusive
        return i0, i1

    @staticmethod
    def extract_timeseries_at_points(
        ds: xr.Dataset,
        varname: str,
        points_gdf: gpd.GeoDataFrame,
        point_id_field: str,
        method: str = "nearest",
        spatial_aggregation: dict | None = None,
    ) -> pd.DataFrame:
        """
        Extract a time series for each point from a gridded dataset.

        Default behavior: nearest cell (xarray .sel with method='nearest').

        Optional spatial aggregation:
            spatial_aggregation = {
                "method": "weighted_quantile",
                "quantile": 0.8,
                "central_weight": 0.4
            }

        For weighted_quantile:
          - a 3x3 neighborhood is extracted around the nearest grid cell
          - the per-timestep value is computed as a weighted quantile (no interpolation)
          - the central cell weight is "central_weight"
          - the remaining weight (1-central_weight) is split equally among the neighbors
        """
        if varname not in ds:
            raise KeyError(f"Variable '{varname}' not found. Available: {list(ds.data_vars)}")

        if point_id_field not in points_gdf.columns:
            raise KeyError(f"point_id_field '{point_id_field}' not in points shapefile columns")

        lat_name, lon_name = GridsHandler.infer_lat_lon_names(ds)
        da = ds[varname]

        if "time" not in da.dims and "time" not in da.coords:
            raise ValueError(f"Variable '{varname}' has no 'time' dimension/coord. Dims: {da.dims}")

        agg = spatial_aggregation or {}
        agg_method = str(agg.get("method", "nearest")).lower()
        if agg_method not in ["nearest", "weighted_quantile"]:
            raise ValueError("pixel_spatial_aggregation.method must be 'nearest' or 'weighted_quantile'")

        # Prepare aggregation parameters (if needed)
        if agg_method == "weighted_quantile":
            q = float(agg.get("quantile", 0.8))
            cw = float(agg.get("central_weight", 0.4))
            if not (0.0 < q < 1.0):
                raise ValueError("pixel_spatial_aggregation.quantile must be between 0 and 1")
            if not (0.0 < cw < 1.0):
                raise ValueError("pixel_spatial_aggregation.central_weight must be between 0 and 1")

            # Only supports 1D lat/lon coordinates (typical rectilinear grids)
            if ds[lat_name].ndim != 1 or ds[lon_name].ndim != 1:
                raise ValueError("weighted_quantile aggregation supports only 1D lat/lon grids")

        out: dict[str, pd.Series] = {}
        for _, row in points_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if geom.geom_type != "Point":
                raise ValueError("Points GeoDataFrame must contain only Point geometries")

            pid = str(row[point_id_field])
            x = float(geom.x)
            y = float(geom.y)

            if agg_method == "nearest":
                sel = da.sel({lat_name: y, lon_name: x}, method=method)
                t = pd.to_datetime(sel["time"].values)
                out[pid] = pd.Series(sel.values, index=t)
                continue

            # weighted_quantile: 3x3 neighborhood around nearest indices
            q = float(agg.get("quantile", 0.8))
            cw = float(agg.get("central_weight", 0.4))

            lat_vals = ds[lat_name].values
            lon_vals = ds[lon_name].values
            i0, i1 = GridsHandler.get_3x3_indices_1d(lat_vals, y)
            j0, j1 = GridsHandler.get_3x3_indices_1d(lon_vals, x)

            block = da.isel({lat_name: slice(i0, i1), lon_name: slice(j0, j1)})

            # Ensure time is first axis
            block_vals = block.values
            if block_vals.ndim != 3:
                raise ValueError("Unexpected neighborhood block dimensions")

            dims = list(block.dims)
            if "time" not in dims:
                raise ValueError("Could not find time dimension in neighborhood block")

            time_axis = dims.index("time")
            if time_axis != 0:
                block_vals = np.moveaxis(block_vals, time_axis, 0)  # (time, ny, nx)

            ny = block_vals.shape[1]
            nx = block_vals.shape[2]

            # Build weights for this block size (edge-safe)
            n_cells = ny * nx
            if n_cells == 1:
                weights_flat = np.array([1.0], dtype=float)
            else:
                w_other = (1.0 - cw) / float(n_cells - 1)
                weights_block = np.full((ny, nx), w_other, dtype=float)

                # Central cell is the nearest one within the extracted window.
                # For a full 3x3 block, it is [1,1]. At borders it is clamped.
                cy = min(1, ny - 1)
                cx = min(1, nx - 1)
                weights_block[cy, cx] = cw
                weights_flat = weights_block.ravel()

            series_vals: list[float] = []
            for k in range(block_vals.shape[0]):
                vals_flat = block_vals[k, :, :].ravel()
                series_vals.append(GridsHandler.weighted_quantile(vals_flat, weights_flat, q))

            t = pd.to_datetime(block["time"].values)
            out[pid] = pd.Series(series_vals, index=t)

        if len(out) == 0:
            raise ValueError("No valid point time series extracted (check geometries/IDs)")

        df = pd.DataFrame(out).sort_index()
        return df
