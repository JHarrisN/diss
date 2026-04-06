import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import matplotlib.dates as mdates
import matplotlib.ticker as mticker


def set_style():
    """
    Set matplotlib rcParams for consistent dissertation-quality figures.
    Call once at the start of any notebook or script.
    
    Key choices:
      - PDF-safe font (Computer Modern via TeX, or serif fallback)
      - Thin lines, muted colours, no chartjunk
      - Font sizes calibrated for a figure width of ~0.9\textwidth in a
        standard LaTeX article/report class
    """
    plt.rcParams.update({
        # font
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        
        # lines
        "lines.linewidth": 0.9,
        "lines.markersize": 3,
        
        # figs
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": True,         # kept for twin-axis plots
        
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",
        "legend.fancybox": False,
        
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


C_TRUE = "#1f77b4"   # steel blue
C_PRED = "#d62728"   # muted red
C_INDEX = "#39880C"  # green

COLOURS = [C_TRUE, C_PRED, C_INDEX]

SPEC_COLOURS = {
    "arch":      "#B85450",   # muted red
    "base":      "#08792E",   # dark grey
    "R_neg":     "#2D6CA2",   # navy
    "R_pos":     "#3E8ABF",   # mid blue
    "R_posneg":  "#69A7D3",   # sky blue
    "r_neg":     "#D4783BFF",   # burnt orange
    "r_pos":     "#E8913AFF",   # orange
    "r_posneg":  "#F2C45AFF",   # amber
}
 
BAR_SPECS = [
    ("arch",      r"ARCH(1000)"),
    ("base",      r"$\sqrt{R_2}$"),
    ("R_neg",     r"$\sqrt{R_2}\mathbf{+}\,R_{\mathbf{-}}$"),
    ("R_pos",     r"$\sqrt{R_2}\mathbf{+}\,R_{\mathbf{+}}$"),
    ("R_posneg",  r"$\sqrt{R_2}\mathbf{+}\,R_{\mathbf{+}}\!\mathbf{+}\,R_{\mathbf{-}}$"),
    ("r_neg",     r"$\sqrt{R_2}\mathbf{+}\,r^{\mathbf{-}}$"),
    ("r_pos",     r"$\sqrt{R_2}\mathbf{+}\,r^{\mathbf{+}}$"),
    ("r_posneg",  r"$\sqrt{R_2}\mathbf{+}\,r^{\mathbf{+}}\!\mathbf{+}\,r^{\mathbf{-}}$"),
]
 
INDICES = ["VIX", "OVX", "GVZ"]
PANEL_TITLES = {
    "VIX": "VIX",
    "OVX": "OVX",
    "GVZ": "GVZ"
}

def plot_vol_timeseries(pred_vol, true_vol, index=None,
                        label_true="VIX", label_pred=r"$\widehat{\mathrm{VIX}}$",
                        label_index="$S_t$",
                        r2=None, figsize=(6.5, 3.2)):
    """
    Overlay predicted and true implied-volatility time series.

    Parameters
    ----------
    pred_vol, true_vol : pd.Series  (values in %, i.e. VIX scale)
    index : pd.Series or None       underlying price series (plotted on right axis)
    label_true, label_pred : str    legend labels
    label_index : str               right-axis label (only if index is not None)
    r2 : float or None              if given, annotated in the legend area
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(true_vol.index, true_vol, color=C_TRUE,
            linewidth=0.8, label=label_true, zorder=3)
    ax.plot(pred_vol.index, pred_vol, color=C_PRED,
            linewidth=0.7, alpha=0.85, label=label_pred, zorder=4)

    ax.set_ylabel("Volatility (%)")

    if index is not None:
        ax2 = ax.twinx()
        ax2.plot(index.index, index, color=C_INDEX,
                 linewidth=0.5, alpha=0.45, linestyle="-.")
        ax2.set_ylabel(label_index, color=C_INDEX)
        ax2.tick_params(axis="y", colors=C_INDEX)
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")

    leg_title = None
    if r2 is not None:
        leg_title = f"$R^2 = {r2:.3f}$"
    ax.legend(loc="upper left", framealpha=0.85, title=leg_title,
              title_fontsize=9)

    fig.tight_layout()
    return fig

def clip_yaxis(fig, ylim_upper, ax_index=0, annotation_fmt='{label} peak: {val:.0f}%', show_arrow=True):
    """
    Truncate y-axis and annotate any clipped peaks.
    
    Parameters
    ----------
    fig : matplotlib.Figure
    ylim_upper : float
        Cap for the y-axis (in the units already plotted, e.g. % vol).
    ax_index : int
        Which axes to clip (0 = main).
    annotation_fmt : str
        Format string with {label} and {val} placeholders.
    show_arrow : bool
        If True, draw arrow from annotation to clipped peak.
    """
    ax = fig.axes[ax_index]
    ax.set_ylim(0, ylim_upper)

    # annotate any series whose max exceeds the cap
    for line in ax.get_lines():
        ydata = line.get_ydata()
        xdata = line.get_xdata()
        peak_val = max(ydata)
        if peak_val > ylim_upper:
            peak_idx = np.argmax(ydata)
            label = line.get_label() or ''
            arrowprops = dict(arrowstyle='->', color=line.get_color()) if show_arrow else None
            ax.annotate(
                annotation_fmt.format(label=label, val=peak_val),
                xy=(xdata[peak_idx], ylim_upper),
                xytext=(15, 10), textcoords='offset points',
                fontsize=10,
                arrowprops=arrowprops,
                color=line.get_color()
            )
    return fig


def plot_pred_vs_true(pred_vol, true_vol, r2=None,
                      label="", figsize=(3.8, 3.8)):
    """
    Scatter plot of predicted vs true volatility with 45-degree line.

    Parameters
    ----------
    pred_vol, true_vol : array-like  (same units)
    r2 : float or None
    label : str   e.g. "Train" or "Test"
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(true_vol, pred_vol, s=8, alpha=0.35, color=C_PRED,
               edgecolors="none", rasterized=True)   # rasterized keeps pdf small

    # 45-degree reference line
    lo = min(np.min(true_vol), np.min(pred_vol)) * 0.9
    hi = max(np.max(true_vol), np.max(pred_vol)) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.6, alpha=0.6)

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    if r2 is not None:
        ax.text(0.05, 0.92, f"$R^2 = {r2:.3f}$",
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))

    if label:
        ax.set_title(label)

    fig.tight_layout()
    return fig


def plot_train_test_scatter(pred_train, true_train, r2_train,
                            pred_test, true_test, r2_test,
                            figsize=(7, 3.5)):
    """
    Two-panel scatter plot (train left, test right) following Guyon & Lekeufack Fig. 4.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, pred, true, r2, title in [
        (ax1, pred_train, true_train, r2_train, "Train"),
        (ax2, pred_test,  true_test,  r2_test,  "Test"),
    ]:
        ax.scatter(true, pred, s=8, alpha=0.35, color=C_PRED,
                   edgecolors="none", rasterized=True)
        lo = min(np.min(true), np.min(pred)) * 0.9
        hi = max(np.max(true), np.max(pred)) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.6, alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        ax.text(0.05, 0.92, f"$R^2 = {r2:.3f}$",
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))

    fig.tight_layout()
    return fig


def plot_ker(lags_days, kernels_dict, figsize=(10, 4), ylim=None, annotation_fmt='{label} peak: {val:.0f}%'):
    """
    Plot K1 and K2 kernels for each volatility index on separate subplots.
    
    Parameters
    ----------
    lags_days : array of lag values in trading days
    kernels_dict : dict
        Nested dictionary with structure:
        {'VIX': {'K1': array, 'K2': array}, 'OVX': {...}, 'GVZ': {...}}
    figsize : tuple
        Figure size (default fits A4 with standard margins).
    ylim : float or None
        If set, truncates y-axis and annotates clipped peaks.
    annotation_fmt : str
        Format string for clip_yaxis annotations.
    """
    fig, axs = plt.subplots(ncols=3, figsize=figsize)
    
    colors = {r"$K_1$": C_TRUE, r"$K_2$": C_PRED}
    linestyles = {r"$K_1$": '-', r"$K_2$": '--'}
    
    for ax, (index_name, kernels) in zip(axs, kernels_dict.items()):
        for k_label, kernel in kernels.items():
            label = index_name + " " + k_label
            ax.plot(lags_days, kernel, 
                   color=colors[k_label], 
                   linewidth=1.0, 
                   label=label,
                   linestyle=linestyles[k_label])
        
        ax.set_xlabel("Lag (trading days)")
        ax.set_ylabel("Weight")
        # ax.set_title(index_name)
        ax.legend()
    
    if ylim is not None:
        for ax_idx in range(3):
            clip_yaxis(fig, ylim, ax_index=ax_idx, annotation_fmt=annotation_fmt, show_arrow=False)
    
    fig.tight_layout()
    return fig


def plot_leverage_ablation2(
    results_dict,
    arch_dict,
    savepath=None,
    bar_label_fontsize=8,
    tick_fontsize=9,
    title_fontsize=11,
    legend_fontsize=9,
    ylabel_fontsize=10,
):
    """
    Parameters
    ----------
    results_dict : dict
        results_dict[index][spec_key] = result_dict with keys
        'train_r2' and 'test_r2'.
 
    arch_dict : dict
        arch_dict[index] = result_dict with 'train_r2' and 'test_r2'.
    """
 
    n_bars = len(BAR_SPECS)
    x = np.arange(n_bars)
 
    fig, axes = plt.subplots(
        3, 2, figsize=(12, 10),
        gridspec_kw={"hspace": 0.35, "wspace": 0.15},
    )
 
    for row, idx in enumerate(INDICES):

        all_vals = []
        for metric_key in ["train_r2", "test_r2"]:
            for key, _ in BAR_SPECS:
                src = arch_dict[idx] if key == "arch" else results_dict[idx][key]
                all_vals.append(float(src[metric_key] if isinstance(src, dict) else src))
        row_ymin = max(0, min(all_vals) - 0.05)
        row_ymax = min(1, max(all_vals) + 0.035)
        for col, (metric_key, col_title) in enumerate([
            ("train_r2", "Train"),
            ("test_r2", "Test"),
        ]):
            ax = axes[row, col]
 
            vals = []
            colours = []
            for key, _ in BAR_SPECS:
                if key == "arch":
                    src = arch_dict[idx]
                else:
                    src = results_dict[idx][key]
                v = src[metric_key] if isinstance(src, dict) else float(src)
                vals.append(float(v))
                colours.append(SPEC_COLOURS[key])
 
            vals = np.array(vals)
 
            bars = ax.bar(
                x, vals, color=colours, width=0.72,
                edgecolor="white", linewidth=0.6, zorder=3,
            )
 
            base_val = float(results_dict[idx]["base"][metric_key])
            ax.axhline(
                base_val, color=SPEC_COLOURS["base"],
                ls="--", lw=0.8, alpha=0.5, zorder=2,
            )
 
            ax.set_ylim(row_ymin, row_ymax)
 
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.004,
                    f"{v:.3f}",
                    ha="center", va="bottom",
                    fontsize=bar_label_fontsize, fontweight="medium",
                )
 
            ax.set_xticks(x)
            if row == len(INDICES) - 1:
                ax.set_xticklabels(
                    [lbl for _, lbl in BAR_SPECS],
                    rotation=45, ha="right", fontsize=tick_fontsize,
                )
            else:
                ax.set_xticklabels([])
 
            if col == 0:
                ax.set_ylabel(r"$R^2$", fontsize=ylabel_fontsize)
 
            if row == 0:
                ax.set_title(col_title, fontsize=title_fontsize, fontweight="bold")
 
            if col == 1:
                ax.annotate(
                    PANEL_TITLES[idx],
                    xy=(1.02, 0.5), xycoords="axes fraction",
                    fontsize=title_fontsize, fontweight="bold",
                    ha="left", va="center", rotation=-90,
                )
 
            ax.grid(axis="y", alpha=0.2, zorder=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=tick_fontsize)
 
    legend_handles = [
        Patch(facecolor=SPEC_COLOURS[key], label=label)
        for key, label in BAR_SPECS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=legend_fontsize,
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.5,
        handletextpad=0.5,
    )
 
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
        print(f"Saved to {savepath}")
 
    return fig