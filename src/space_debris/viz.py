from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from space_debris.density import DensityGrid


def plot_altitude_distribution(df: pd.DataFrame) -> go.Figure:
    return px.histogram(
        df,
        x="altitude_km",
        nbins=60,
        title="Altitude Distribution",
        labels={"altitude_km": "Altitude (km)"},
    )


def plot_density_heatmap(dgrid: DensityGrid) -> go.Figure:
    # Plotly expects z shape (y,x) depending. We'll use (altitude, inclination).
    fig = px.imshow(
        dgrid.counts,
        aspect="auto",
        origin="lower",
        labels={"x": "Inclination bin", "y": "Altitude bin", "color": "Relative density"},
        title="Debris Density Heatmap (Altitude vs Inclination)",
    )
    return fig


def plot_clustering_scatter(df: pd.DataFrame, *, kmeans, max_points: int = 5000) -> go.Figure:
    d = df
    if len(d) > max_points:
        d = d.sample(n=max_points, random_state=42)
    feats = d[["altitude_km", "inclination_deg", "velocity_km_s", "object_size", "crowding"]]
    labels = kmeans.predict(feats)
    fig = px.scatter(
        d,
        x="altitude_km",
        y="inclination_deg",
        color=labels.astype(str),
        title="Orbital Clustering Visualization",
        labels={"altitude_km": "Altitude (km)", "inclination_deg": "Inclination (deg)"},
        opacity=0.85,
    )
    fig.update_layout(legend_title_text="Cluster")
    return fig


def plot_yearly_growth(df: pd.DataFrame, *, year_col: str = "launch_year") -> go.Figure:
    if year_col not in df.columns:
        raise ValueError(f"Missing {year_col} in dataframe.")
    d = df[df[year_col] > 0].copy()
    if d.empty:
        # Fallback plot when no years exist.
        return px.line(title="Debris Growth Over Years (No year data found)")

    counts = d.groupby(year_col).size().reset_index(name="debris_count")
    counts = counts.sort_values(year_col)
    return px.line(counts, x=year_col, y="debris_count", markers=True, title="Debris Growth Over Years")


def _orbit_xyz(
    altitude_km: np.ndarray,
    inclination_deg: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Circular-orbit proxy positions.
    re_km = 6371.0
    r = re_km + altitude_km
    theta = rng.uniform(0, 2 * np.pi, size=len(r))

    incl = np.deg2rad(inclination_deg)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = r * np.sin(incl) * np.cos(theta) * 0.35
    return x, y, z


def plot_orbital_region_risk_map(
    df: pd.DataFrame,
    *,
    value_col: str = "crowding",
    max_points: int = 2500,
    seed: int = 42,
) -> go.Figure:
    d = df
    if len(d) > max_points:
        d = d.sample(n=max_points, random_state=seed)

    rng = np.random.default_rng(seed)
    x, y, z = _orbit_xyz(d["altitude_km"].to_numpy(), d["inclination_deg"].to_numpy(), rng=rng)
    c = d[value_col].to_numpy() if value_col in d.columns else d["crowding"].to_numpy()

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=3,
                    color=c,
                    colorscale="Turbo",
                    opacity=0.75,
                    colorbar=dict(title=value_col),
                ),
            )
        ]
    )
    fig.update_layout(
        title="Orbital Region Risk Mapping (3D Proxy)",
        scene=dict(
            xaxis_title="X (km proxy)",
            yaxis_title="Y (km proxy)",
            zaxis_title="Z (km proxy)",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_orbit_evolution_animation(
    df: pd.DataFrame,
    *,
    year_col: str = "launch_year",
    value_col: str = "crowding",
    max_frames: int = 8,
    max_points_per_frame: int = 250,
    seed: int = 42,
) -> Optional[go.Figure]:
    if year_col not in df.columns:
        return None

    d = df[df[year_col] > 0].copy()
    if d.empty:
        return None

    years = sorted(d[year_col].unique().tolist())
    if len(years) > max_frames:
        # Take evenly spaced year slices.
        idxs = np.linspace(0, len(years) - 1, num=max_frames, dtype=int)
        years = [years[i] for i in idxs]

    rng = np.random.default_rng(seed)

    frames = []
    base_points = None
    for y in years:
        frame_df = d[d[year_col] == y]
        if len(frame_df) > max_points_per_frame:
            frame_df = frame_df.sample(n=max_points_per_frame, random_state=seed)

        x, yy, zz = _orbit_xyz(
            frame_df["altitude_km"].to_numpy(),
            frame_df["inclination_deg"].to_numpy(),
            rng=rng,
        )
        c = frame_df[value_col].to_numpy() if value_col in frame_df.columns else frame_df["crowding"].to_numpy()

        scatter = go.Scatter3d(
            x=x,
            y=yy,
            z=zz,
            mode="markers",
            marker=dict(
                size=3,
                color=c,
                colorscale="Turbo",
                opacity=0.75,
            ),
        )
        frames.append(go.Frame(data=[scatter], name=str(y)))
        base_points = scatter

    if base_points is None:
        return None

    fig = go.Figure(data=[base_points], frames=frames)
    fig.update_layout(
        title="Animated Orbital Evolution (Risk-colored Proxy)",
        scene=dict(
            xaxis_title="X (km proxy)",
            yaxis_title="Y (km proxy)",
            zaxis_title="Z (km proxy)",
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=550, redraw=True), fromcurrent=True)],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

