import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import os
import sys
sys.path.append(os.path.abspath(".."))
from pdv_plots import *
from emp_pdv import empirical_study, TSPL, NormTSPL



def compare_vols(pred_vol, true_vol, interval_label=None, plot=True):
    """
    Creates a volatility comparison DataFrame.
    
    Parameters
    ----------
    pred_vol : pd.Series or np.ndarray
        Predicted volatility values, with datetime index if pd.Series
    true_vol : pd.Series or np.ndarray
        True volatility values, with matching datetime index if pd.Series
    interval_label : str, optional
        Label for the interval (e.g., 'test set', 'training set') for reference
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'predicted_vol': model predictions
        - 'true_vol': actual values
        - 'error': prediction error (predicted - true)
        - 'abs_error': absolute prediction error
        - 'pct_error': percent error
        Indexed by date if inputs are pd.Series
    """
    # Convert to pd.Series if needed
    if isinstance(pred_vol, np.ndarray):
        pred_vol = pd.Series(pred_vol)
    if isinstance(true_vol, np.ndarray):
        true_vol = pd.Series(true_vol)
    
    # Align indices
    common_idx = pred_vol.index.intersection(true_vol.index)
    pred_vol = pred_vol.loc[common_idx]
    true_vol = true_vol.loc[common_idx]
    
    # Compute errors
    error = pred_vol.values - true_vol.values
    abs_error = np.abs(error)
    pct_error = (error / np.where(true_vol.values != 0, true_vol.values, np.nan)) * 100
    
    # Create DataFrame
    comparison_df = pd.DataFrame({
        "predicted_vol": pred_vol.values,
        "true_vol": true_vol.values,
        "error": error,
        "abs_error": abs_error,
        "pct_error": pct_error
    }, index=common_idx)
    
    if interval_label:
        comparison_df.attrs["interval"] = interval_label

    if plot:
        set_style()
        vol_ts_plot = plot_vol_timeseries(comparison_df["predicted_vol"], comparison_df["true_vol"])
        return vol_ts_plot, comparison_df

    return comparison_df

"""
Grid search over max_delta for PDV model.

Searches over candidate max_delta values, clipping the span grid 
appropriately at each step, and collects train/test R² to identify 
the effective memory horizon for each asset class.
"""

def build_span_grid(max_delta, use_lasso=True):
    """
    Build a log-spaced span grid clipped to max_delta.
    
    For large max_delta the full ~60-span dense grid is used.
    For small max_delta the grid is truncated so no span exceeds
    max_delta (an EWMA with span > max_delta would reference 
    lags outside the available return history).
    """
    if use_lasso:
        # dense log-spaced grid (matches your usual pipeline)
        full_grid = np.unique(
            np.round(np.logspace(np.log10(2), np.log10(max(max_delta, 3)), 60))
        ).astype(int)
        full_grid = full_grid[full_grid >= 2]          # span >= 2
        full_grid = full_grid[full_grid <= max_delta]   # clip
    else:
        # sparse Ridge fallback
        full_grid = np.array([10, 20, 120, 250])
        full_grid = full_grid[full_grid <= max_delta]
        if len(full_grid) < 2:
            full_grid = np.array([2, max_delta])

    return full_grid.tolist()


def search_max_delta(
    # data identifiers
    index, vol, index_suffix, vol_suffix,
    load_from,
    # dates
    train_start_date, test_start_date, test_end_date,
    # model config
    KernelClass=TSPL,
    model_spec=(1, 2),
    # search grid
    max_delta_candidates=None,
    # pipeline options
    use_lasso=True,
    cv_splits=0,
    plot=False,
    neg_ret_feat=False,
    pos_ret_feat=False,
    forecast=0,
    api_key=None,
    filepath=None,
):
    """
    Run the full PDV pipeline for each candidate max_delta and collect results.
    
    Parameters
    ----------
    max_delta_candidates : list[int], optional
        Values of max_delta to try. Defaults to a coarse grid
        [50, 100, 150, 200, 300, 500, 750, 1000].
    
    Returns
    -------
    summary_df : pd.DataFrame
        Columns: max_delta, train_r2, test_r2, train_rmse, test_rmse,
                 alphas, deltas, betas
    all_results : dict
        Full result dicts keyed by max_delta.
    """
    if max_delta_candidates is None:
        max_delta_candidates = [50, 100, 150, 200, 300, 500, 750, 1000]

    rows = []
    all_results = {}

    for md in max_delta_candidates:
        print(f"\n{'='*60}")
        print(f"  max_delta = {md}")
        print(f"{'='*60}")

        spans = build_span_grid(md, use_lasso=use_lasso)
        print(f"  span grid: {spans[:5]}...{spans[-3:]} ({len(spans)} spans)")

        try:
            res = empirical_study(
                load_from=load_from,
                train_start_date=train_start_date,
                test_start_date=test_start_date,
                test_end_date=test_end_date,
                index=index, vol=vol,
                index_suffix=index_suffix, vol_suffix=vol_suffix,
                max_delta=md,
                KernelClass=KernelClass,
                model_spec=model_spec,
                spans=spans,
                cv_splits=cv_splits,
                use_lasso=use_lasso,
                plot=False,  # suppress per-run plots
                neg_ret_feat=neg_ret_feat,
                pos_ret_feat=pos_ret_feat,
                forecast=forecast,
                api_key=api_key,
                filepath=filepath,
            )

            rows.append({
                "max_delta": md,
                "train_r2": res["train_r2"],
                "test_r2": res["test_r2"],
                "train_rmse": res["train_rmse"],
                "test_rmse": res["test_rmse"],
                "alphas": res["alphas"].tolist(),
                "deltas": res["deltas"].tolist(),
                "betas": res["betas"].tolist(),
                "intercept": res["intercept"],
            })
            all_results[md] = res

        except Exception as e:
            print(f"  FAILED: {e}")
            rows.append({
                "max_delta": md,
                "train_r2": np.nan, "test_r2": np.nan,
                "train_rmse": np.nan, "test_rmse": np.nan,
                "alphas": None, "deltas": None,
                "betas": None, "intercept": None,
            })

    summary_df = pd.DataFrame(rows)
    return summary_df, all_results


def plot_search_results(summary_df, asset_label=""):
    """
    Plot train and test R^2 as a function of max_delta.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # R^2 plot
    ax = axes[0]
    ax.plot(summary_df["max_delta"], summary_df["train_r2"],
            "o-", label="Train R^2", color="steelblue")
    ax.plot(summary_df["max_delta"], summary_df["test_r2"],
            "s--", label="Test R^2", color="coral")
    ax.set_xlabel("max_delta (trading days)")
    ax.set_ylabel("R^2")
    ax.set_title(f"{asset_label}: R^2 vs max_delta")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMSE plot
    ax = axes[1]
    ax.plot(summary_df["max_delta"], summary_df["train_rmse"],
            "o-", label="Train RMSE", color="steelblue")
    ax.plot(summary_df["max_delta"], summary_df["test_rmse"],
            "s--", label="Test RMSE", color="coral")
    ax.set_xlabel("max_delta (trading days)")
    ax.set_ylabel("RMSE")
    ax.set_title(f"{asset_label}: RMSE vs max_delta")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

"""
Generate alternative-specification tables for the leverage-effect ablation study.

Produces four .tex snippet files:
  - alt_mets_tspl.tex   : metrics for TSPL kernel return specifications
  - alt_mets_oneday.tex : metrics for one-day return specifications
  - alt_params_tspl.tex : parameters for TSPL kernel return specifications
  - alt_params_oneday.tex : parameters for one-day return specifications
"""

TABLES_DIR = "../tables"
os.makedirs(TABLES_DIR, exist_ok=True) # output directory

ALPHA_UPPER_BOUND = 10.0          # optimiser bound-hit
DELTA_LOWER_BOUND = 1 / (100*252)
BETA2_STATIONARITY_BOUND = 1.0    # beta_2 >= 1 violates stationarity

def extract_metrics_row(result, index_name, spec_label):
    """Return a dict with train/test RMSE and R² for one (spec, index) pair."""
    return {
        "Index": index_name,
        "Specification": spec_label,
        "train_rmse": result["train_rmse"],
        "train_r2": result["train_r2"],
        "test_rmse": result["test_rmse"],
        "test_r2": result["test_r2"],
    }


def extract_oneday_params_row(result, index_name, spec_label, extras):
    """
    Row for the one-day-return parameter table.

    Parameters
    ----------
    extras : list of str
        Which extra_betas are present, in order.  e.g. ['neg'], ['pos'],
        ['neg', 'pos'], or [].
    """
    row = {
        "Index": index_name,
        "Specification": spec_label,
        "beta_0": result["intercept"],
        "beta_2": result["betas"][0],
        "alpha_2": result["alphas"][0],
        "delta_2": result["deltas"][0],
        "beta_neg": np.nan,
        "beta_pos": np.nan,
    }
    for i, tag in enumerate(extras):
        row[f"beta_{tag}"] = result["extra_betas"][i]
    return row


def extract_tspl_params_row(result, index_name, spec_label, feature_tags):
    """
    Row for the TSPL kernel parameter table.

    Parameters
    ----------
    feature_tags : list of str
        Ordered feature keys matching result['betas'], e.g.
        ['neg', '2'] for R_{-} + \\sqrt{R_2},  ['pos', 'neg', '2'] for R_{+} + R_{-} + \\sqrt{R_2}.
    """
    row = {
        "Index": index_name,
        "Specification": spec_label,
        "beta_0": result["intercept"],
        # initialise all optional columns to NaN
        "beta_neg": np.nan, "alpha_neg": np.nan, "delta_neg": np.nan,
        "beta_pos": np.nan, "alpha_pos": np.nan, "delta_pos": np.nan,
        "beta_2":   np.nan, "alpha_2":   np.nan, "delta_2":   np.nan,
    }
    for i, tag in enumerate(feature_tags):
        row[f"beta_{tag}"]  = result["betas"][i]
        row[f"alpha_{tag}"] = result["alphas"][i]
        row[f"delta_{tag}"] = result["deltas"][i]
    return row


def collect_all_rows(
        sp2, o2, g2, sp2neg, o2neg, g2neg, sp2pos, o2pos, g2pos, sp2posneg,
        o2posneg, g2posneg, sp2n, o2n, g2n, sp2p, o2p, g2p, sp2pn, o2pn, g2pn
        ):
    """
    Build four lists of dicts: metrics_tspl, metrics_oneday,
    params_tspl, params_oneday.

    Edit the (result_var, index_name) tuples to match your namespace.
    """

    base_specs = [(sp2, "VIX"), (o2, "OVX"), (g2, "GVZ")]

    # One-day return augmentations
    neg_specs     = [(sp2neg, "VIX"),    (o2neg, "OVX"),    (g2neg, "GVZ")]
    pos_specs     = [(sp2pos, "VIX"),    (o2pos, "OVX"),    (g2pos, "GVZ")]
    posneg_specs  = [(sp2posneg, "VIX"), (o2posneg, "OVX"), (g2posneg, "GVZ")]

    # TSPL kernel return augmentations
    Rneg_specs    = [(sp2n, "VIX"),  (o2n, "OVX"),  (g2n, "GVZ")]
    Rpos_specs    = [(sp2p, "VIX"),  (o2p, "OVX"),  (g2p, "GVZ")]
    Rposneg_specs = [(sp2pn, "VIX"), (o2pn, "OVX"), (g2pn, "GVZ")]

    metrics_oneday, metrics_tspl = [], []

    spec_groups_oneday = [
        (base_specs,   r"$\sqrt{R_2}$"),
        (pos_specs,    r"$\sqrt{R_2} + r^{\mathbf{+}}$"),
        (neg_specs,    r"$\sqrt{R_2} + r^{\mathbf{-}}$"),
        (posneg_specs, r"$\sqrt{R_2} + r^{\mathbf{+}} + r^{\mathbf{-}}")
    ]
    spec_groups_tspl = [
        (base_specs,    r"$\sqrt{R_2}$"),
        (Rpos_specs,    r"$\sqrt{R_2} + R_{\mathbf{+}}$"),
        (Rneg_specs,    r"$\sqrt{R_2} + R_{\mathbf{-}}$"),
        (Rposneg_specs, r"$\sqrt{R_2} + R_{\mathbf{+}} + R_{\mathbf{-}}$")
    ]

    for specs, label in spec_groups_oneday:
        for result, idx in specs:
            metrics_oneday.append(extract_metrics_row(result, idx, label))

    for specs, label in spec_groups_tspl:
        for result, idx in specs:
            metrics_tspl.append(extract_metrics_row(result, idx, label))

    params_oneday = []
    for result, idx in base_specs:
        params_oneday.append(
            extract_oneday_params_row(result, idx, r"$\sqrt{R_2}$", []))
    for result, idx in pos_specs:
        params_oneday.append(
            extract_oneday_params_row(result, idx, r"$\sqrt{R_2} + r^{\mathbf{+}}$", ["pos"]))
    for result, idx in neg_specs:
        params_oneday.append(
            extract_oneday_params_row(result, idx, r"$\sqrt{R_2} + r^{\mathbf{-}}$", ["neg"]))
    for result, idx in posneg_specs:
        params_oneday.append(
            extract_oneday_params_row(result, idx, r"$\sqrt{R_2} + r^{\mathbf{+}} + r^{\mathbf{-}}$",
                                     ["pos", "neg"]))

    params_tspl = []
    for result, idx in base_specs:
        params_tspl.append(
            extract_tspl_params_row(result, idx, r"$\sqrt{R_2}$", ["2"]))
    for result, idx in Rpos_specs:
        params_tspl.append(
            extract_tspl_params_row(result, idx, r"$\sqrt{R_2} + R_{\mathbf{+}}$",
                                   ["pos", "2"]))
    for result, idx in Rneg_specs:
        params_tspl.append(
            extract_tspl_params_row(result, idx, r"$\sqrt{R_2} + R_{\mathbf{-}}$",
                                   ["neg", "2"]))
    for result, idx in Rposneg_specs:
        params_tspl.append(
            extract_tspl_params_row(result, idx, r"$\sqrt{R_2} + R_{\mathbf{+}} + R_{\mathbf{-}}$",
                                   ["pos", "neg", "2"]))

    return metrics_tspl, metrics_oneday, params_tspl, params_oneday


def _fmt_num(val, decimals=3):
    """Format a number, returning '---' (em-dash) for NaN."""
    if pd.isna(val):
        return "{---}"
    return f"{val:.{decimals}f}"


def _fmt_param(val, col_name, decimals=3):
    """
    Format a parameter value with dagger annotation for pathologies.
    Wraps annotated cells in {} for siunitx S-column compatibility.
    """
    if pd.isna(val):
        return "{---}"

    s = f"{val:.{decimals}f}"

    # Check pathologies
    is_pathological = False
    if col_name.startswith("alpha") and abs(val - ALPHA_UPPER_BOUND) < 0.01:
        is_pathological = True
    if col_name.startswith("delta") and val < DELTA_LOWER_BOUND * 1.01:
        is_pathological = True
    if col_name.startswith("beta_2") and val >= BETA2_STATIONARITY_BOUND:
        is_pathological = True

    if is_pathological:
        return r"{" + s + r"\textsuperscript{\dag}}"
    return s


def format_metrics_df(rows):
    """
    Build a formatted metrics DataFrame with Index → Specification hierarchy.
    Returns (df, latex_str).
    """
    df = pd.DataFrame(rows)

    # Sort: index order VIX -> OVX -> GVZ, then spec order (base -> pos -> neg -> combined)
    idx_order = {"VIX": 0, "OVX": 1, "GVZ": 2}
    df["_idx_sort"] = df["Index"].map(idx_order)
    
    # Determine spec order: base (no extras), positive, negative, combined (both)
    def spec_sort_key(spec):
        if r"$\sqrt{R_2}$" == spec:  # base only
            return 0
        elif "+" in spec and "-" in spec:  # combined
            return 3
        elif "+" in spec:  # positive only
            return 1
        elif "-" in spec:  # negative only
            return 2
        return 4
    
    df["_spec_sort"] = df["Specification"].apply(spec_sort_key)
    df = df.sort_values(["_idx_sort", "_spec_sort"]).drop(columns=["_idx_sort", "_spec_sort"])
    df = df.reset_index(drop=True)

    # Find best test R^2 per index (for bolding)
    best_r2 = df.groupby("Index")["test_r2"].max()

    # Format numbers
    fmt_rows = []
    for _, row in df.iterrows():
        is_best = abs(row["test_r2"] - best_r2[row["Index"]]) < 1e-6
        test_r2_str = f"{row['test_r2']:.3f}"
        if is_best:
            test_r2_str = r"{\textbf{" + test_r2_str + r"}}"

        fmt_rows.append({
            "Index": row["Index"],
            "Specification": row["Specification"],
            "train_rmse": _fmt_num(row["train_rmse"]),
            "train_r2":   _fmt_num(row["train_r2"]),
            "test_rmse":  _fmt_num(row["test_rmse"]),
            "test_r2":    test_r2_str,
        })

    fmt_df = pd.DataFrame(fmt_rows)

    # Suppress repeated Index labels for row hierarchy
    prev_idx = None
    for i in range(len(fmt_df)):
        if fmt_df.loc[i, "Index"] == prev_idx:
            fmt_df.loc[i, "Index"] = ""
        else:
            prev_idx = fmt_df.loc[i, "Index"]

    return fmt_df


def format_params_df(rows, col_order, col_labels):
    """
    Build a formatted parameter DataFrame.

    Parameters
    ----------
    col_order : list of str   – internal column keys in display order
    col_labels : list of str  – LaTeX column headers
    """
    df = pd.DataFrame(rows)

    idx_order = {"VIX": 0, "OVX": 1, "GVZ": 2}
    df["_idx_sort"] = df["Index"].map(idx_order)
    
    # Determine spec order: base (no extras), positive, negative, combined (both)
    def spec_sort_key(spec):
        if r"$\sqrt{R_2}$" == spec:
            return 0
        elif "+" in spec and "-" in spec:
            return 3
        elif "+" in spec:
            return 1
        elif "-" in spec:
            return 2
        return 4
    
    df["_spec_sort"] = df["Specification"].apply(spec_sort_key)
    df = df.sort_values(["_idx_sort", "_spec_sort"]).drop(columns=["_idx_sort", "_spec_sort"])
    df = df.reset_index(drop=True)

    fmt_rows = []
    for _, row in df.iterrows():
        fmt = {"Index": row["Index"], "Specification": row["Specification"]}
        for col in col_order:
            fmt[col] = _fmt_param(row.get(col, np.nan), col)
        fmt_rows.append(fmt)

    fmt_df = pd.DataFrame(fmt_rows)

    # Suppress repeated Index labels
    prev_idx = None
    for i in range(len(fmt_df)):
        if fmt_df.loc[i, "Index"] == prev_idx:
            fmt_df.loc[i, "Index"] = ""
        else:
            prev_idx = fmt_df.loc[i, "Index"]

    return fmt_df

def render_metrics_latex(fmt_df, caption, label):
    """Render a metrics table to a LaTeX tabular string."""

    n_specs_per_idx = fmt_df["Specification"].ne("").cumsum()
    # Find rows where a new index group starts (non-empty Index column)
    new_group_rows = [i for i in range(len(fmt_df)) if fmt_df.iloc[i]["Index"] != ""]

    lines = []
    lines.append(r"\begin{tabular}{ll SSSS}")
    lines.append(r"  \toprule")
    lines.append(
        r"  Index & Specification "
        r"& {RMSE} & {$R^2$} & {RMSE} & {$R^2$} \\"
    )
    lines.append(
        r"  & "
        r"& \multicolumn{2}{c}{Train} & \multicolumn{2}{c}{Test} \\"
    )
    lines.append(r"  \cmidrule(lr){3-4} \cmidrule(lr){5-6}")

    for i, (_, row) in enumerate(fmt_df.iterrows()):
        if i in new_group_rows and i > 0:
            lines.append(r"  \midrule")
        vals = (
            f"  {row['Index']} & {row['Specification']} "
            f"& {row['train_rmse']} & {row['train_r2']} "
            f"& {row['test_rmse']} & {row['test_r2']} \\\\"
        )
        lines.append(vals)

    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def render_params_latex(fmt_df, col_order, col_labels, footnote=True):
    """Render a parameter table to a LaTeX tabular string."""

    n_data_cols = len(col_order)
    col_fmt = "ll" + "S" * n_data_cols

    new_group_rows = [i for i in range(len(fmt_df)) if fmt_df.iloc[i]["Index"] != ""]

    lines = []
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"  \toprule")

    header = "  Index & Specification"
    for lbl in col_labels:
        header += f" & {{{lbl}}}"
    header += r" \\"
    lines.append(header)
    lines.append(r"  \midrule")

    for i, (_, row) in enumerate(fmt_df.iterrows()):
        if i in new_group_rows and i > 0:
            lines.append(r"  \midrule")
        vals = f"  {row['Index']} & {row['Specification']}"
        for col in col_order:
            vals += f" & {row[col]}"
        vals += r" \\"
        lines.append(vals)

    lines.append(r"  \bottomrule")
    lines.append(r"\end{tabular}")

    if footnote:
        lines.append("")
        lines.append(
            r"\vspace{2pt}"
        )
        lines.append(
            r"\noindent{\footnotesize "
            r"${}^{\dag}$\,Pathological: "
            r"$\alpha$ at optimiser upper bound (10), "
            r"$\delta$ at optimiser lower bound ($\frac{1}{25200}$), "
            r"or $\beta_2 \geq 1$ (stationarity violation).}"
        )

    return "\n".join(lines)

def write_all_tables(
        sp2, o2, g2, sp2neg, o2neg, g2neg, sp2pos, o2pos, g2pos, sp2posneg,
        o2posneg, g2posneg, sp2n, o2n, g2n, sp2p, o2p, g2p, sp2pn, o2pn, g2pn
):
    metrics_tspl, metrics_oneday, params_tspl, params_oneday = collect_all_rows(
        sp2, o2, g2, sp2neg, o2neg, g2neg, sp2pos, o2pos, g2pos,
        sp2posneg, o2posneg, g2posneg,
        sp2n, o2n, g2n, sp2p, o2p, g2p, sp2pn, o2pn, g2pn
        )

    fmt_met_tspl = format_metrics_df(metrics_tspl)
    tex = render_metrics_latex(
        fmt_met_tspl,
        caption="Fit metrics for TSPL kernel return specifications.",
        label="tab:alt_mets_tspl",
    )
    with open(os.path.join(TABLES_DIR, "alt_mets_tspl.tex"), "w") as f:
        f.write(tex)

    fmt_met_oneday = format_metrics_df(metrics_oneday)
    tex = render_metrics_latex(
        fmt_met_oneday,
        caption="Fit metrics for one-day return specifications.",
        label="tab:alt_mets_oneday",
    )
    with open(os.path.join(TABLES_DIR, "alt_mets_oneday.tex"), "w") as f:
        f.write(tex)

    oneday_cols   = ["beta_0", "beta_2", "alpha_2", "delta_2",
                     "beta_pos", "beta_neg"]
    oneday_labels = [r"$\beta_0$", r"$\beta_2$", r"$\alpha_2$", r"$\delta_2$",
                     r"$\beta_{\mathbf{+}}$", r"$\beta_{\mathbf{-}}$"]

    fmt_par_oneday = format_params_df(params_oneday, oneday_cols, oneday_labels)
    tex = render_params_latex(fmt_par_oneday, oneday_cols, oneday_labels,
                              footnote=False)
    with open(os.path.join(TABLES_DIR, "alt_params_oneday.tex"), "w") as f:
        f.write(tex)

    # intercept, base params, positive feature params, negative feature params
    tspl_cols   = ["beta_0",
                   "beta_2",   "alpha_2",   "delta_2",
                   "beta_pos", "alpha_pos", "delta_pos",
                   "beta_neg", "alpha_neg", "delta_neg"]
    tspl_labels = [r"$\beta_0$",
                   r"$\beta_2$",   r"$\alpha_2$",   r"$\delta_2$",
                   r"$\beta_{\mathbf{+}}$", r"$\alpha_{\mathbf{+}}$", r"$\delta_{\mathbf{+}}$",
                   r"$\beta_{\mathbf{-}}$", r"$\alpha_{\mathbf{-}}$", r"$\delta_{\mathbf{-}}$"]

    fmt_par_tspl = format_params_df(params_tspl, tspl_cols, tspl_labels)
    tex = render_params_latex(fmt_par_tspl, tspl_cols, tspl_labels,
                              footnote=True)
    with open(os.path.join(TABLES_DIR, "alt_params_tspl.tex"), "w") as f:
        f.write(tex)

    print("Tables written to", TABLES_DIR)


# Diebold-Mariano (DM) Test

def diebold_mariano(actual, pred1, pred2, H1="two-sided"):
    """
    Test equal predictive accuracy of two models

    Returns DM statistic and p-value
    Positive DM ==> model 2 has smaller errors
    """

    e1 = actual - pred1
    e2 = actual -pred2
    d = e1**2 - e2**2
    d_bar = d.mean()

    # Newey-West with 1-lag for mild autocor
    n = len(d)
    gamma0 = np.cov(d, ddof=1)[0][0] if d.ndim > 1 else np.var(d, ddof=1)
    gamma1 = np.cov(d[1:], d[:-1], ddof=1)[0, 1]
    var_d = (gamma0 + 2*gamma1) / n

    dm_stat = d_bar /np.sqrt(var_d)

    if H1 == "two-sided":
        p_val = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    elif H1 == "greater":
        p_val = 1 - stats.norm.cdf(dm_stat)
    else:
        p_val = stats.norm.cdf(dm_stat)

    return dm_stat, p_val