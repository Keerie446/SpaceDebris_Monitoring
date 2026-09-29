from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression


@dataclass(frozen=True)
class GrowthForecast:
    years: np.ndarray
    counts: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


def risk_distribution(df: pd.DataFrame, *, label_col: str = "risk_label") -> pd.Series:
    if label_col not in df.columns:
        return pd.Series(dtype=int)
    return df[label_col].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)


def orbit_type_distribution(df: pd.DataFrame, *, col: str = "orbit_type") -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=int)
    return df[col].value_counts()


def debris_count(df: pd.DataFrame) -> int:
    return int(len(df))


def forecast_debris_growth(
    df: pd.DataFrame,
    *,
    year_col: str = "launch_year",
    horizon_years: int = 5,
) -> GrowthForecast | None:
    d = df[df[year_col] > 0].copy()
    if len(d) < 3:
        return None

    counts = d.groupby(year_col).size().reset_index(name="debris_count").sort_values(year_col)
    X = counts[year_col].to_numpy().reshape(-1, 1)
    y = counts["debris_count"].to_numpy()

    model = LinearRegression()
    model.fit(X, y)
    yhat = model.predict(X)

    residuals = y - yhat
    sigma = float(np.std(residuals)) if len(residuals) else 0.0

    last_year = int(counts[year_col].iloc[-1])
    future_years = np.arange(last_year + 1, last_year + 1 + horizon_years)
    X_future = future_years.reshape(-1, 1)
    y_future = model.predict(X_future)
    lower = y_future - 1.96 * sigma
    upper = y_future + 1.96 * sigma

    years = np.concatenate([counts[year_col].to_numpy(), future_years])
    counts_all = np.concatenate([y, y_future])
    lower_all = np.concatenate([y - 1.96 * sigma, lower])
    upper_all = np.concatenate([y + 1.96 * sigma, upper])

    return GrowthForecast(years=years, counts=counts_all, lower=lower_all, upper=upper_all)


def forecast_figure(forecast: GrowthForecast, *, title: str = "Future Debris Growth Forecast") -> px.line:
    plot_df = pd.DataFrame(
        {
            "year": forecast.years,
            "debris_count": forecast.counts,
            "lower": forecast.lower,
            "upper": forecast.upper,
        }
    )
    fig = px.line(plot_df, x="year", y="debris_count", markers=True, title=title)
    fig.add_traces(
        [
            # Lower bound
            px.line(plot_df, x="year", y="lower").data[0],
            # Upper bound
            px.line(plot_df, x="year", y="upper").data[0],
        ]
    )
    fig.update_traces(line=dict(dash="dot"), selector=dict(mode="lines"))
    return fig

