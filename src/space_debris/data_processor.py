from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from space_debris.config import DEFAULT_ORBIT_THRESHOLDS, OrbitThresholds


@dataclass(frozen=True)
class ColumnMapping:
    altitude: str | None
    velocity: str | None
    inclination: str | None
    size: str | None
    orbit_type: str | None
    launch_year: str | None
    decay_status: str | None


def _normalize_colname(s: str) -> str:
    return "".join(ch for ch in s.strip().lower() if ch.isalnum())


def infer_column_mapping(
    df: pd.DataFrame,
    *,
    thresholds: OrbitThresholds = DEFAULT_ORBIT_THRESHOLDS,
) -> ColumnMapping:
    # Heuristic inference based on column name patterns.
    cols = list(df.columns)
    norm = {_normalize_colname(c): c for c in cols}

    def pick(*names: str) -> str | None:
        for n in names:
            key = _normalize_colname(n)
            if key in norm:
                return norm[key]
        return None

    altitude = pick("altitude", "alt", "h", "apogee", "perigee", "heightkm", "height")
    velocity = pick("velocity", "vel", "speed", "v", "impactspeed", "v_impact")
    inclination = pick("inclination", "incl", "i", "orbitinclination", "inc")
    size = pick("objectsize", "size", "diameter", "d", "object_size", "diameter_m", "diameter_km")
    orbit_type = pick("orbittype", "orbit_type", "orbits", "orbitregion", "class", "objecttype")
    launch_year = pick("launchyear", "launch_year", "year", "launchtime", "launch_date", "launchtime")
    decay_status = pick("decaystatus", "decay_status", "decayed", "isdecayed", "decay")

    # If user dataset has no recognizable altitude, keep mapping empty and let UI drive it.
    return ColumnMapping(
        altitude=altitude,
        velocity=velocity,
        inclination=inclination,
        size=size,
        orbit_type=orbit_type,
        launch_year=launch_year,
        decay_status=decay_status,
    )


def _coerce_numeric(series: pd.Series) -> pd.Series:
    # Convert to numeric; non-convertible becomes NaN.
    if series.dtype.kind in ("i", "u", "f"):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def _normalize_velocity_units(v_k: pd.Series) -> pd.Series:
    # Heuristic: if values look like m/s (hundreds to thousands), convert to km/s.
    v = _coerce_numeric(v_k)
    v_median = v.dropna().median()
    if pd.isna(v_median):
        return v
    if v_median > 100:  # m/s typical for orbital speeds
        return v / 1000.0
    return v


def _parse_year(col: pd.Series) -> pd.Series:
    # Try direct int coercion first; then fall back to parsing dates.
    s = col.astype(str)
    year_numeric = pd.to_numeric(s, errors="coerce")
    if year_numeric.notna().mean() >= 0.5:
        return year_numeric
    parsed = pd.to_datetime(col, errors="coerce", utc=True)
    return parsed.dt.year


def _parse_decay_status(col: pd.Series) -> pd.Series:
    s = col.astype(str).str.strip().str.lower()
    truthy = s.str.contains("true|yes|decay|decayed|expired", regex=True, na=False)
    falsy = s.str.contains("false|no|notdecay|not_decayed|active|intact", regex=True, na=False)
    out = pd.Series(np.nan, index=col.index, dtype="float64")
    out[truthy] = 1.0
    out[falsy] = 0.0
    return out


def _map_orbit_type(value: Any, altitude_km: float, thresholds: OrbitThresholds) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        # Infer from altitude.
        if altitude_km < thresholds.leo_max_km:
            return "LEO"
        if altitude_km >= thresholds.geo_min_km:
            return "GEO"
        return "MEO"

    s = str(value).strip().upper()
    # Common aliases
    if "LEO" in s:
        return "LEO"
    if "GEO" in s:
        return "GEO"
    if "MEO" in s:
        return "MEO"
    if s.startswith("L"):
        return "LEO"
    if s.startswith("G"):
        return "GEO"

    # Fallback to altitude inference.
    if altitude_km < thresholds.leo_max_km:
        return "LEO"
    if altitude_km >= thresholds.geo_min_km:
        return "GEO"
    return "MEO"


def estimate_velocity_from_altitude(altitude_km: pd.Series) -> pd.Series:
    # Circular orbit approximation: v = sqrt(mu / r)
    # mu ~ 398600 km^3/s^2, Earth radius ~ 6371 km
    mu_km3_s2 = 398600.0
    r_km = 6371.0 + _coerce_numeric(altitude_km)
    v = np.sqrt(mu_km3_s2 / r_km)
    return v


def prepare_features(
    df: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    thresholds: OrbitThresholds = DEFAULT_ORBIT_THRESHOLDS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if mapping.altitude is None:
        raise ValueError("Altitude column is required for feature preparation.")

    out = pd.DataFrame(index=df.index)
    out["altitude_km"] = _coerce_numeric(df[mapping.altitude])
    out = out.dropna(subset=["altitude_km"])

    # Inclination
    if mapping.inclination is not None:
        out["inclination_deg"] = _coerce_numeric(df.loc[out.index, mapping.inclination])
    else:
        # If the dataset doesn't include inclination, use a neutral default.
        out["inclination_deg"] = 0.0

    # Velocity
    if mapping.velocity is not None:
        out["velocity_km_s"] = _normalize_velocity_units(df.loc[out.index, mapping.velocity])
    else:
        out["velocity_km_s"] = estimate_velocity_from_altitude(out["altitude_km"])

    # Size
    if mapping.size is not None:
        out["object_size"] = _coerce_numeric(df.loc[out.index, mapping.size])
    else:
        out["object_size"] = np.nan

    if out["object_size"].notna().any():
        out["object_size"] = out["object_size"].fillna(out["object_size"].median())
    else:
        out["object_size"] = 1.0

    # Launch year
    meta: dict[str, Any] = {"has_year": False}
    if mapping.launch_year is not None:
        year = _parse_year(df.loc[out.index, mapping.launch_year])
        out["launch_year"] = year
        if out["launch_year"].notna().mean() >= 0.5:
            out["launch_year"] = out["launch_year"].fillna(out["launch_year"].median()).astype(int)
            meta["has_year"] = True
        else:
            out["launch_year"] = 0
    else:
        out["launch_year"] = 0

    # Decay status
    if mapping.decay_status is not None:
        decay = _parse_decay_status(df.loc[out.index, mapping.decay_status])
        out["decay_status"] = decay
    else:
        out["decay_status"] = np.nan
    if out["decay_status"].notna().any():
        out["decay_status"] = out["decay_status"].fillna(out["decay_status"].median())
    else:
        out["decay_status"] = 0.0

    # Orbit type (categorical)
    if mapping.orbit_type is not None:
        raw_orbit = df.loc[out.index, mapping.orbit_type]
        out["orbit_type"] = [
            _map_orbit_type(v, float(a), thresholds) for v, a in zip(raw_orbit.tolist(), out["altitude_km"].tolist())
        ]
    else:
        out["orbit_type"] = [
            _map_orbit_type(None, float(a), thresholds) for a in out["altitude_km"].tolist()
        ]

    # Basic cleaning
    out["inclination_deg"] = out["inclination_deg"].fillna(out["inclination_deg"].median())
    out["velocity_km_s"] = out["velocity_km_s"].fillna(out["velocity_km_s"].median())
    out["object_size"] = out["object_size"].replace([np.inf, -np.inf], np.nan).fillna(out["object_size"].median())

    return out.reset_index(drop=True), meta

