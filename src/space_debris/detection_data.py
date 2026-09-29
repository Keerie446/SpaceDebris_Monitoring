from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectionMeta:
    orbit_quantiles: tuple[float, float]  # (q1, q2) for altitude_km proxy -> LEO/MEO/GEO
    used_rows: int
    used_boxes: int


def parse_bboxes(bboxes_cell: Any) -> list[list[float]]:
    """
    Parse `bboxes` column entries like:
      '[[282, 308, 58, 86], [228, 256, 191, 218]]'
    into a list of boxes. Boxes are [x1,y1,x2,y2] or [x1,y1,x2,y2,conf].
    """
    if bboxes_cell is None or (isinstance(bboxes_cell, float) and np.isnan(bboxes_cell)):
        return []
    if isinstance(bboxes_cell, list):
        return [[float(v) for v in box] for box in bboxes_cell]
    s = str(bboxes_cell).strip()
    if not s:
        return []
    boxes = ast.literal_eval(s)
    out: list[list[float]] = []
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        out.append([float(v) for v in box])
    return out


def _orbit_type_from_altitude(altitude_km: np.ndarray, q1: float, q2: float) -> np.ndarray:
    out = np.empty_like(altitude_km, dtype=object)
    out[altitude_km < q1] = "LEO"
    out[(altitude_km >= q1) & (altitude_km < q2)] = "MEO"
    out[altitude_km >= q2] = "GEO"
    return out


def prepare_detection_features(
    raw_df: pd.DataFrame,
    *,
    image_id_col: str = "ImageID",
    bboxes_col: str = "bboxes",
    max_boxes: int | None = 50000,
    seed: int = 42,
) -> tuple[pd.DataFrame, DetectionMeta]:
    """
    Converts (ImageID, bboxes[]) rows into per-bbox "object rows" with the schema
    expected by the rest of the app:

      altitude_km        -> y_center proxy
      inclination_deg    -> x_center proxy
      velocity_km_s      -> bbox width proxy
      object_size        -> bbox area proxy
      orbit_type         -> derived from altitude_km proxy into LEO/MEO/GEO bins
      launch_year        -> ImageID (ordered proxy for time)
      decay_status       -> bbox confidence proxy (1.0 if missing)
    """
    if image_id_col not in raw_df.columns or bboxes_col not in raw_df.columns:
        raise ValueError(f"Expected columns `{image_id_col}` and `{bboxes_col}` in detection dataset.")

    used_rows = 0
    records: list[dict[str, Any]] = []
    for _, row in raw_df.iterrows():
        image_id = row[image_id_col]
        boxes = parse_bboxes(row[bboxes_col])
        if not boxes:
            continue
        used_rows += 1
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            conf = box[4] if len(box) >= 5 else 1.0

            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            area = width * height

            records.append(
                {
                    "altitude_km": float(y_center),
                    "inclination_deg": float(x_center),
                    "velocity_km_s": float(width),
                    "object_size": float(area),
                    "launch_year": int(image_id),
                    "decay_status": float(conf),
                    # Helpful for mapping WOW feature to "image space"
                    "x_center_proxy": float(x_center),
                    "y_center_proxy": float(y_center),
                }
            )

    if not records:
        raise ValueError("No bounding boxes found after parsing the detection dataset.")

    df = pd.DataFrame.from_records(records)
    used_boxes = int(len(df))

    if max_boxes is not None and len(df) > max_boxes:
        df = df.sample(n=max_boxes, random_state=seed).reset_index(drop=True)

    # Derive orbit type bins from altitude proxy distribution.
    q1, q2 = np.quantile(df["altitude_km"].to_numpy(), [0.33, 0.66]).tolist()
    df["orbit_type"] = _orbit_type_from_altitude(df["altitude_km"].to_numpy(), q1=float(q1), q2=float(q2))

    meta = DetectionMeta(
        orbit_quantiles=(float(q1), float(q2)),
        used_rows=used_rows,
        used_boxes=used_boxes,
    )
    return df.reset_index(drop=True), meta


def load_detection_dataset_from_folder(dataset_root: Path) -> pd.DataFrame:
    """
    Expected structure:
      debris-detection 2/train.csv
      debris-detection 2/val.csv
    """
    train_path = dataset_root / "train.csv"
    val_path = dataset_root / "val.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Could not find `train.csv` and `val.csv` under: {dataset_root}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return pd.concat([train_df, val_df], ignore_index=True)

