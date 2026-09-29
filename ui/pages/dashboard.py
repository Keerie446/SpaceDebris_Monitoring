from __future__ import annotations

from typing import Any
import os
import glob

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from space_debris.data_processor import ColumnMapping, infer_column_mapping, prepare_features, estimate_velocity_from_altitude
from space_debris.density import compute_altitude_density, compute_density_grid
from space_debris.risk_model import RISK_LABELS, predict_risk, train_risk_models
from space_debris.config import DEFAULT_ORBIT_THRESHOLDS
from space_debris.viz import (
    plot_altitude_distribution,
    plot_clustering_scatter,
    plot_density_heatmap,
    plot_orbital_region_risk_map,
    plot_orbit_evolution_animation,
    plot_yearly_growth,
)
from space_debris.analytics import (
    debris_count,
    forecast_debris_growth,
    forecast_figure,
    orbit_type_distribution,
    risk_distribution,
)


def _pick_column_ui(
    *,
    st_label: str,
    df: pd.DataFrame,
    inferred: str | None,
    required: bool = False,
) -> str | None:
    choices = ["<Not provided>"] + [str(c) for c in df.columns]
    default_idx = 0
    if inferred is not None and inferred in df.columns:
        default_idx = 1 + list(df.columns).index(inferred)

    idx = st.selectbox(st_label, options=choices, index=default_idx)
    if idx == "<Not provided>":
        if required:
            st.warning(f"{st_label} is required. Choose a valid column.")
        return None
    return idx


def _number_or_default(x: Any, default: float) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        return float(pd.to_numeric(x, errors="coerce"))
    except Exception:
        return float(default)


def render_dashboard() -> None:
    st.markdown(
        """
        <div class="neon-card">
          <h2 style="margin-top:0;">Dashboard</h2>
          <p style="margin-bottom:0;">Upload a dataset to generate orbital density, clustering, risk prediction, and analytics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Let user choose between example datasets bundled with the project or upload
    data_source = st.radio("Data source", options=["Example dataset", "Upload dataset"], index=0, horizontal=True)

    raw_df = None
    uploaded_name = None

    if data_source == "Upload dataset":
        uploaded = st.file_uploader("Upload debris dataset (CSV or Parquet)", type=["csv", "parquet"])
        if uploaded is None:
            st.stop()

        # Read uploaded dataset
        try:
            if uploaded.name.lower().endswith(".csv"):
                raw_df = pd.read_csv(uploaded)
            else:
                raw_df = pd.read_parquet(uploaded)
            uploaded_name = uploaded.name
        except Exception as e:
            st.error(f"Failed to read dataset: {e}")
            st.stop()
    else:
        # Look for example datasets in project root folder 'debris-detection 2'
        example_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "debris-detection 2"))
        example_files = []
        if os.path.isdir(example_dir):
            for ext in ("csv", "parquet"):
                example_files.extend(glob.glob(os.path.join(example_dir, f"*.{ext}")))

        if not example_files:
            st.info("No example datasets found in project. Please switch to 'Upload dataset' and provide a file.")
            st.stop()

        # Present examples to pick
        example_files = sorted(example_files)
        example_names = [os.path.basename(p) for p in example_files]
        choice = st.selectbox("Choose example dataset", options=example_names)
        chosen_path = example_files[example_names.index(choice)]

        try:
            if chosen_path.lower().endswith(".csv"):
                raw_df = pd.read_csv(chosen_path)
            else:
                raw_df = pd.read_parquet(chosen_path)
            uploaded_name = os.path.basename(chosen_path)
            st.success(f"Loaded example dataset: {uploaded_name}")
        except Exception as e:
            st.error(f"Failed to read example dataset: {e}")
            st.stop()

    if raw_df.empty:
        st.error("Uploaded dataset is empty.")
        st.stop()

    # Reset cached artifacts if a different dataset is uploaded/selected.
    current_name = uploaded_name or "<unknown>"
    if st.session_state.get("_sd_upload_name") != current_name:
        for k in ["_sd_features_df", "_sd_alt_edges", "_sd_alt_counts", "_sd_dgrid", "_sd_artifacts", "_sd_df_train", "_sd_meta"]:
            st.session_state.pop(k, None)
        st.session_state["_sd_upload_name"] = current_name

    st.divider()
    with st.expander("Dataset preview", expanded=False):
        st.dataframe(raw_df.head(50), use_container_width=True)

    # Infer mapping
    inferred = infer_column_mapping(raw_df)

    def _synthesize_from_bboxes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Explode a DataFrame containing an `bboxes` column into object-level rows
        and synthesize orbital features so the pipeline can run on arbitrary inputs.
        """
        import ast
        import numpy as _np

        objs = []
        for _, row in df.iterrows():
            bstr = row.get("bboxes") if "bboxes" in row.index else None
            try:
                boxes = ast.literal_eval(bstr) if isinstance(bstr, str) else []
            except Exception:
                boxes = []
            for b in boxes:
                # b is [x1,x2,w,h] or similar; compute area if length>=4
                area = float(b[2] * b[3]) if len(b) >= 4 else 1.0
                objs.append({
                    "object_size": max(1.0, area / 100.0),
                })

        if not objs:
            # Fallback: create a small synthetic dataset
            objs = [{"object_size": 1.0} for _ in range(500)]

        out = pd.DataFrame(objs)
        n = len(out)
        # Synthetic orbital attributes (LEO-focused)
        rng = _np.random.default_rng(42)
        out["altitude_km"] = rng.uniform(200, 1200, size=n)
        out["inclination_deg"] = rng.uniform(0, 180, size=n)
        out["velocity_km_s"] = estimate_velocity_from_altitude(out["altitude_km"])
        out["decay_status"] = 0.0
        years = rng.integers(1990, 2024, size=n)
        out["launch_year"] = years
        # Infer orbit type
        def _map(a):
            if a < DEFAULT_ORBIT_THRESHOLDS.leo_max_km:
                return "LEO"
            if a >= DEFAULT_ORBIT_THRESHOLDS.geo_min_km:
                return "GEO"
            return "MEO"

        out["orbit_type"] = out["altitude_km"].apply(_map)

        # Compute crowding as altitude-bin normalized counts
        alt_edges = _np.linspace(out["altitude_km"].min(), out["altitude_km"].max(), num=40)
        counts, _ = _np.histogram(out["altitude_km"].to_numpy(), bins=alt_edges)
        # map each row to a crowding value
        idxs = _np.digitize(out["altitude_km"].to_numpy(), alt_edges, right=False) - 1
        idxs = _np.clip(idxs, 0, len(counts) - 1)
        maxc = counts.max() if counts.max() > 0 else 1
        out["crowding"] = counts[idxs] / float(maxc)

        meta = {"synthesized": True}
        return out.reset_index(drop=True), meta

    # Decide whether to auto-process: Example datasets are processed automatically.
    auto_process = data_source == "Example dataset"

    # If auto_process, synthesize features when no altitude column exists; otherwise allow manual mapping.
    features_df = None
    meta: dict = {}
    if auto_process:
        with st.spinner("Auto-processing dataset and training models..."):
            if inferred.altitude is not None:
                # Use inferred mapping where possible
                mapping = ColumnMapping(
                    altitude=inferred.altitude,
                    velocity=inferred.velocity,
                    inclination=inferred.inclination,
                    size=inferred.size,
                    orbit_type=inferred.orbit_type,
                    launch_year=inferred.launch_year,
                    decay_status=inferred.decay_status,
                )
                try:
                    features_df, meta = prepare_features(raw_df, mapping)
                except Exception:
                    features_df, meta = _synthesize_from_bboxes(raw_df)
            else:
                features_df, meta = _synthesize_from_bboxes(raw_df)

            # Build density grid and train models
            features_df, dgrid = compute_density_grid(features_df)
            features_df_1d, alt_edges, alt_counts = compute_altitude_density(features_df)
            artifacts, df_train = train_risk_models(features_df)

            # Store artifacts
            st.session_state["_sd_features_df"] = features_df
            st.session_state["_sd_alt_edges"] = alt_edges
            st.session_state["_sd_alt_counts"] = alt_counts
            st.session_state["_sd_dgrid"] = dgrid
            st.session_state["_sd_artifacts"] = artifacts
            st.session_state["_sd_df_train"] = df_train
            st.session_state["_sd_meta"] = meta

        st.success("Auto-processing complete. Visualizations ready.")
    else:
        # Manual mapping flow (kept for uploads with expected columns)
        st.write("Column mapping (auto-inferred, adjust if needed):")

        altitude_col = _pick_column_ui(st_label="Altitude column (km)", df=raw_df, inferred=inferred.altitude, required=True)
        velocity_col = _pick_column_ui(st_label="Velocity column (km/s or m/s)", df=raw_df, inferred=inferred.velocity)
        inclination_col = _pick_column_ui(st_label="Inclination column (deg)", df=raw_df, inferred=inferred.inclination)
        size_col = _pick_column_ui(st_label="Object size column", df=raw_df, inferred=inferred.size)
        orbit_type_col = _pick_column_ui(st_label="Orbit type column (LEO/MEO/GEO)", df=raw_df, inferred=inferred.orbit_type)
        launch_year_col = _pick_column_ui(st_label="Launch year / year column", df=raw_df, inferred=inferred.launch_year)
        decay_status_col = _pick_column_ui(st_label="Decay status column", df=raw_df, inferred=inferred.decay_status)

        mapping = ColumnMapping(
            altitude=altitude_col,
            velocity=velocity_col,
            inclination=inclination_col,
            size=size_col,
            orbit_type=orbit_type_col,
            launch_year=launch_year_col,
            decay_status=decay_status_col,
        )

        process_enabled = mapping.altitude is not None
        if mapping.altitude is None:
            st.error("Altitude column is required. Please select a valid column and rerun.")
            st.stop()

        if st.button("Process dataset & train models", disabled=not process_enabled):
            with st.spinner("Processing data, building density grid, and training models..."):
                features_df, meta = prepare_features(raw_df, mapping)
                features_df, dgrid = compute_density_grid(features_df)

                # Also compute 1D altitude density for prediction without inclination.
                features_df_1d, alt_edges, alt_counts = compute_altitude_density(features_df)

                artifacts, df_train = train_risk_models(features_df)

            st.success("Ready.")

            # Store in session state for subsequent predictions
            st.session_state["_sd_features_df"] = features_df
            st.session_state["_sd_alt_edges"] = alt_edges
            st.session_state["_sd_alt_counts"] = alt_counts
            st.session_state["_sd_dgrid"] = dgrid
            st.session_state["_sd_artifacts"] = artifacts
            st.session_state["_sd_df_train"] = df_train
            st.session_state["_sd_meta"] = meta

    # If already processed, render outputs.
    if "_sd_artifacts" in st.session_state and st.session_state.get("_sd_artifacts") is not None:
        artifacts = st.session_state["_sd_artifacts"]
        df_train = st.session_state["_sd_df_train"]
        features_df = st.session_state["_sd_features_df"]
        dgrid = st.session_state["_sd_dgrid"]
        alt_edges = st.session_state["_sd_alt_edges"]
        alt_counts = st.session_state["_sd_alt_counts"]
        meta = st.session_state.get("_sd_meta", {})

        df_train = df_train.copy()
        st.plotly_chart(plot_altitude_distribution(features_df), use_container_width=True)

        with st.expander("Space Overview Panel", expanded=True):
            total = debris_count(df_train)
            st.metric("Total debris count", total)

            rdist = risk_distribution(df_train)
            pie = px.pie(values=rdist.values, names=rdist.index, title="Risk Classification")
            st.plotly_chart(pie, use_container_width=True)

            odist = orbit_type_distribution(df_train)
            odist_df = odist.reset_index()
            odist_df.columns = ["orbit_type", "debris_count"]
            bar = px.bar(odist_df, x="orbit_type", y="debris_count", title="Orbit Type Distribution")
            st.plotly_chart(bar, use_container_width=True)

        st.divider()
        st.header("Orbit Visualization Panel")
        st.plotly_chart(plot_clustering_scatter(features_df, kmeans=artifacts.kmeans), use_container_width=True)
        st.plotly_chart(plot_density_heatmap(dgrid), use_container_width=True)

        st.divider()
        st.header("Risk Prediction Panel")

        dataset_min_alt = float(features_df["altitude_km"].min())
        dataset_max_alt = float(features_df["altitude_km"].max())
        dataset_min_vel = float(features_df["velocity_km_s"].min())
        dataset_max_vel = float(features_df["velocity_km_s"].max())
        dataset_min_size = float(features_df["object_size"].min())
        dataset_max_size = float(features_df["object_size"].max())

        default_alt = float(features_df["altitude_km"].median())
        default_vel = float(features_df["velocity_km_s"].median())
        default_size = float(features_df["object_size"].median())

        altitude = st.slider("Altitude (km)", min_value=max(0.0, dataset_min_alt), max_value=dataset_max_alt, value=default_alt)
        velocity = st.slider(
            "Velocity (km/s)", min_value=float(dataset_min_vel), max_value=float(dataset_max_vel), value=default_vel
        )
        size = st.slider(
            "Object size", min_value=float(dataset_min_size), max_value=float(dataset_max_size), value=default_size
        )

        # Optional fields to improve the prediction quality.
        use_inclination = st.toggle("Provide inclination (optional)", value=False)
        if use_inclination:
            incl_min = float(features_df["inclination_deg"].min())
            incl_max = float(features_df["inclination_deg"].max())
            inclination = st.slider("Inclination (deg)", min_value=incl_min, max_value=incl_max, value=float(features_df["inclination_deg"].median()))
        else:
            inclination = float(features_df["inclination_deg"].median())

        orbit_type_options = ["Auto/infer"] + sorted(features_df["orbit_type"].dropna().unique().tolist())
        orbit_type_choice = st.selectbox("Orbit type (optional)", orbit_type_options, index=0)
        orbit_type = None if orbit_type_choice == "Auto/infer" else orbit_type_choice

        decay_status_default = float(features_df["decay_status"].median())
        decay_status = st.toggle("Decayed?", value=bool(decay_status_default >= 0.5))

        if st.button("Predict risk for this object"):
            # Map altitude to 1D density-based crowding.
            idx = np.digitize([altitude], alt_edges, right=False) - 1
            idx = int(np.clip(idx[0], 0, len(alt_edges) - 2))
            max_count = float(np.max(alt_counts)) if np.max(alt_counts) > 0 else 1.0
            crowding = float(alt_counts[idx] / max_count)

            if orbit_type is None:
                if altitude < DEFAULT_ORBIT_THRESHOLDS.leo_max_km:
                    orbit_type_pred = "LEO"
                elif altitude >= DEFAULT_ORBIT_THRESHOLDS.geo_min_km:
                    orbit_type_pred = "GEO"
                else:
                    orbit_type_pred = "MEO"
            else:
                orbit_type_pred = orbit_type

            df_pred = pd.DataFrame(
                [
                    {
                        "altitude_km": float(altitude),
                        "velocity_km_s": float(velocity),
                        "inclination_deg": float(inclination),
                        "object_size": float(size),
                        "orbit_type": orbit_type_pred,
                        "decay_status": 1.0 if decay_status else 0.0,
                        "crowding": crowding,
                    }
                ]
            )

            pred_df = predict_risk(artifacts, df_pred)
            pred = pred_df.iloc[0]

            st.markdown(
                f"""
                <div class="neon-card">
                  <b>Predicted Risk Level:</b> {pred["pred_risk_label"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

            probs = pd.Series({lbl: float(pred[f"p_{lbl.lower()}"]) for lbl in artifacts.class_order}, name="Probability")
            st.bar_chart(probs)

        st.divider()
        st.header("Density Heatmap Section")
        st.plotly_chart(plot_density_heatmap(dgrid), use_container_width=True)

        st.divider()
        st.header("Time Analysis Section")
        st.plotly_chart(plot_yearly_growth(df_train, year_col="launch_year"), use_container_width=True)

        with st.expander("Advanced: Future Debris Growth Forecast", expanded=False):
            forecast = forecast_debris_growth(df_train, year_col="launch_year", horizon_years=5)
            if forecast is not None:
                st.plotly_chart(forecast_figure(forecast), use_container_width=True)
            else:
                st.info("Not enough year data for forecasting.")

        st.divider()
        st.header("Orbital Region Risk Mapping (WOW Feature)")
        # Prefer risk_score if present, else crowding.
        value_col = "risk_score" if "risk_score" in df_train.columns else "crowding"
        st.plotly_chart(
            plot_orbital_region_risk_map(df_train, value_col=value_col),
            use_container_width=True,
        )

        anim = plot_orbit_evolution_animation(df_train, year_col="launch_year", value_col=value_col)
        if anim is not None:
            st.plotly_chart(anim, use_container_width=True)
        else:
            st.info("No year field available for animation.")

