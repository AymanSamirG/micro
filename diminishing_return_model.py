#!/usr/bin/env python3
"""
diminishing_return_model.py

Self-contained utility to fit a diminishing-returns (Hill) model to a single
marketing channel.  The script expects a CSV file with at least two columns:
  1) spend  – channel spend for each period (float)
  2) response – observed outcome (sales, conversions, revenue)
Optionally a date column can be supplied for nicer plots.

Usage example:
  python diminishing_return_model.py \
      --input weekly_data.csv \
      --spend_col spend \
      --response_col conversions \
      --adstock 0.6 \
      --plot curve.png \
      --output params.json

Outputs
-------
• params.json  – fitted parameters + derived metrics in JSON.
• curve.png    – (optional) diagnostic plot of fitted curve vs. data.

Dependencies: numpy, pandas, scipy, matplotlib, seaborn (see requirements.txt)
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

# -------------  Model definitions -------------------------------------------------

def adstock(spend: np.ndarray, lam: float) -> np.ndarray:
    """Apply geometric ad-stock transformation.

    s_t = x_t + lam * s_(t-1)

    Parameters
    ----------
    spend : array-like
        Raw spend series.
    lam : float, 0 ≤ λ < 1
        Carry-over (decay) parameter.

    Returns
    -------
    np.ndarray
        Ad-stocked spend series.
    """
    if not 0 <= lam < 1:
        raise ValueError("λ (adstock) must be in [0, 1).")
    s = np.zeros_like(spend, dtype=float)
    for t, x in enumerate(spend):
        s[t] = x + lam * s[t - 1] if t else x
    return s


def hill(x: np.ndarray, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Hill / Michaelis-Menten saturating curve (monotone, concave)."""
    return alpha * (x ** beta) / (gamma + x ** beta)


def fit_hill(spend: np.ndarray, response: np.ndarray, p0: Tuple[float, float, float] = (1.0, 0.5, 100.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Fit Hill curve to data via non-linear least squares.

    Returns (params, covariance).
    """
    bounds = (0, np.inf)  # enforce positivity of α, β, γ
    params, pcov = curve_fit(hill, spend, response, p0=p0, bounds=bounds, maxfev=100000)
    return params, pcov


def derive_metrics(alpha: float, beta: float, gamma: float) -> Dict[str, float]:
    """Compute useful derived metrics from Hill parameters."""
    half_sat = gamma ** (1.0 / beta)  # spend at 50% saturation
    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "spend_half_saturation": float(half_sat),
    }

# -------------  CLI helpers -------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a diminishing-returns Hill model to marketing spend data.")
    parser.add_argument("--input", required=True, help="Path to input CSV containing spend + response columns.")
    parser.add_argument("--spend_col", default="spend", help="Column name for spend (default: spend)")
    parser.add_argument("--response_col", default="response", help="Column name for response variable (default: response)")
    parser.add_argument("--date_col", default=None, help="Optional date column for plotting.")
    parser.add_argument("--adstock", type=float, default=None, help="λ parameter for ad-stock (0 ≤ λ < 1). If omitted, no ad-stock applied.")
    parser.add_argument("--output", default="params.json", help="JSON file to write fitted parameters (default: params.json)")
    parser.add_argument("--group_cols", default=None, help="Comma-separated list of column names to group by (e.g., campaign,ad_group). If provided, the model is fitted separately for each group.")
    parser.add_argument("--plot", default=None, help="Optional path (file or directory) to save fitted curve plot(s). If grouping is enabled and a directory is supplied, one plot per group is saved inside the directory.")
    return parser.parse_args()


# -------------  Main --------------------------------------------------------------

def main():
    args = parse_args()

    # --- Load data
    df = pd.read_csv(args.input)

    # Determine grouping
    group_cols = [c.strip() for c in args.group_cols.split(',')] if args.group_cols else None

    if group_cols:
        for col in group_cols:
            if col not in df.columns:
                raise KeyError(f"Group column '{col}' not found in input CSV.")

        results = {}
        for group_key, gdf in df.groupby(group_cols):
            key_str = "|".join(str(k) for k in (group_key if isinstance(group_key, tuple) else (group_key,)))
            spend_arr = gdf[args.spend_col].astype(float).values
            resp_arr = gdf[args.response_col].astype(float).values

            spend_trans = adstock(spend_arr, args.adstock) if args.adstock is not None else spend_arr.copy()

            params, pcov = fit_hill(spend_trans, resp_arr)
            alpha, beta, gamma = params

            metrics = derive_metrics(alpha, beta, gamma)
            metrics["adstock_lambda"] = args.adstock
            metrics["param_covariance"] = pcov.tolist()

            results[key_str] = metrics

            # Plot per group
            if args.plot:
                # If plot is directory, save inside; else append group key to filename
                plot_path = args.plot
                if Path(args.plot).is_dir():
                    plot_path = Path(args.plot) / f"curve_{key_str}.png"
                else:
                    stem, ext = Path(args.plot).stem, Path(args.plot).suffix
                    plot_path = Path(f"{stem}_{key_str}{ext}")
                _make_plot(gdf, spend_trans, resp_arr, params, args, save_path=plot_path)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Parameters for {len(results)} groups saved to {args.output}")
        return

    if args.spend_col not in df.columns or args.response_col not in df.columns:
        raise KeyError("Specified spend or response column not found in input CSV.")

    spend = df[args.spend_col].astype(float).values
    response = df[args.response_col].astype(float).values

    # --- Ad-stock transformation
    if args.adstock is not None:
        spend_trans = adstock(spend, args.adstock)
    else:
        spend_trans = spend.copy()

    # --- Fit model
    params, pcov = fit_hill(spend_trans, response)
    alpha, beta, gamma = params

    # --- Save parameters + derived metrics
    metrics = derive_metrics(alpha, beta, gamma)
    metrics["adstock_lambda"] = args.adstock
    metrics["param_covariance"] = pcov.tolist()

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Parameters saved to {args.output}")

    # --- Plot (single, non-group case)
    if args.plot:
        _make_plot(df, spend_trans, response, params, args, save_path=args.plot)
        print(f"Plot saved to {args.plot}")


def _make_plot(df: pd.DataFrame, spend_trans: np.ndarray, response: np.ndarray, params: np.ndarray, args, save_path: Path):
    alpha, beta, gamma = params

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))

    # Scatter original points
    plt.scatter(spend_trans, response, label="Observed", alpha=0.7)

    # Fitted curve
    x_line = np.linspace(0, spend_trans.max() * 1.1, 200)
    y_hat = hill(x_line, alpha, beta, gamma)
    plt.plot(x_line, y_hat, color="red", label="Fitted Hill curve")

    plt.xlabel("Ad-stocked spend" if args.adstock is not None else "Spend")
    plt.ylabel(args.response_col)
    plt.title("Diminishing-Returns Hill Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()