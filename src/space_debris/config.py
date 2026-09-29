from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrbitThresholds:
    # Altitude in km above Earth's surface.
    leo_max_km: float = 2000.0
    # GEO is around 35,786 km altitude above Earth surface.
    geo_min_km: float = 35786.0


DEFAULT_ORBIT_THRESHOLDS = OrbitThresholds()

