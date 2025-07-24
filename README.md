# Diminishing Returns Model

This repository contains a simple, self-contained Python utility to estimate a diminishing-returns response curve (Hill model) for a single marketing channel.

## Quick start

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Prepare your data**  
   Create a CSV file (e.g. `weekly_data.csv`) with at least two columns:

   * `spend` — spend amount per period (float)
   * `response` — observed outcome (sales, conversions, revenue, etc.)

   Optionally include a `date` column for clearer plots.

3. **Run the model**

```bash
python diminishing_return_model.py \
    --input weekly_data.csv \
    --spend_col spend \
    --response_col response \
    --adstock 0.6 \
    --plot curve.png \
    --output params.json
```

Arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input CSV | **required** |
| `--spend_col` | Column with spend values | `spend` |
| `--response_col` | Column with outcome values | `response` |
| `--date_col` | Optional date column (only affects plot) | `None` |
| `--adstock` | λ carry-over parameter (0 ≤ λ < 1); if omitted, raw spend is used | `None` |
| `--output` | Output JSON for fitted parameters + metrics | `params.json` |
| `--group_cols` | Comma-separated column names to fit separate curves per group (e.g., `campaign,ad_group`) | `None` |
| `--plot` | File or directory path for plots. If grouping is used and a directory is given, one plot per group is saved there. | `None` |

## Outputs

* **`params.json`** — Fitted parameters (`alpha`, `beta`, `gamma`) plus derived metric `spend_half_saturation` (or a nested object keyed by group for `--group_cols`).
* **`curve.png`** or **`plots/curve_<group>.png`** — Diagnostic plot(s) of spend vs. response with fitted curve.

## Model details

The response curve is the Hill (Michaelis–Menten) function:

\[ \hat{y}(x) = \frac{\alpha x^{\beta}}{\gamma + x^{\beta}} \]

• **Monotonic**: more spend never reduces response.  
• **Concave**: marginal returns decline with spend.

Optional **ad-stock** transformation models carry-over effects:

\[ s_t = x_t + \lambda s_{t-1} \quad (0 \le \lambda < 1) \]

Parameters are estimated by non-linear least squares (`scipy.optimize.curve_fit`).

## Extending

* Replace the estimator with a Bayesian implementation (e.g. PyMC) for credible intervals and partial pooling across groups.
* Fit multiple channels simultaneously by extending the model function.
* Add seasonality or control variables with regression or state-space models.

Pull requests welcome!