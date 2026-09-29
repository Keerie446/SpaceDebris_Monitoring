from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DensityGrid:
    altitude_edges_km: np.ndarray
    inclination_edges_deg: np.ndarray
    counts: np.ndarray


def compute_density_grid(
    df: pd.DataFrame,
    *,
    altitude_bins: int = 30,
    inclination_bins: int = 30,
) -> tuple[pd.DataFrame, DensityGrid]:
    """
    Adds `crowding` per object based on local (altitude,inclination) bin counts.
    """
    altitude = df["altitude_km"].to_numpy()
    inclination = df["inclination_deg"].to_numpy()

    alt_min, alt_max = float(np.nanmin(altitude)), float(np.nanmax(altitude))
    inc_min, inc_max = float(np.nanmin(inclination)), float(np.nanmax(inclination))

    # Ensure non-zero ranges.
    if alt_max <= alt_min:
        alt_max = alt_min + 1.0
    if inc_max <= inc_min:
        inc_max = inc_min + 1.0

    alt_edges = np.linspace(alt_min, alt_max, altitude_bins + 1)
    inc_edges = np.linspace(inc_min, inc_max, inclination_bins + 1)

    alt_idx = np.digitize(altitude, alt_edges, right=False) - 1
    inc_idx = np.digitize(inclination, inc_edges, right=False) - 1

    # Clamp to valid indices.
    alt_idx = np.clip(alt_idx, 0, altitude_bins - 1)
    inc_idx = np.clip(inc_idx, 0, inclination_bins - 1)

    counts = np.zeros((altitude_bins, inclination_bins), dtype=np.float64)
    # Count points per bin.
    np.add.at(counts, (alt_idx, inc_idx), 1.0)

    max_count = float(np.max(counts)) if np.max(counts) > 0 else 1.0
    local_density = counts[alt_idx, inc_idx] / max_count

    out = df.copy()
    out["crowding"] = local_density
    return out, DensityGrid(alt_edges, inc_edges, counts)


def compute_altitude_density(
    df: pd.DataFrame,
    *,
    altitude_bins: int = 30,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    1D density estimate over altitude bins. Used for risk prediction when
    the user doesn't provide inclination explicitly.
    """
    altitude = df["altitude_km"].to_numpy()
    alt_min, alt_max = float(np.nanmin(altitude)), float(np.nanmax(altitude))
    if alt_max <= alt_min:
        alt_max = alt_min + 1.0

    edges = np.linspace(alt_min, alt_max, altitude_bins + 1)
    idx = np.digitize(altitude, edges, right=False) - 1
    idx = np.clip(idx, 0, altitude_bins - 1)

    counts = np.zeros((altitude_bins,), dtype=np.float64)
    np.add.at(counts, idx, 1.0)
    max_count = float(np.max(counts)) if np.max(counts) > 0 else 1.0
    local_density = counts[idx] / max_count

    out = df.copy()
    out["alt_crowding"] = local_density
    return out, edges, counts

