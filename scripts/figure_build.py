import numpy as np
import plotly.graph_objects as go
from scipy import stats


# -------------------------------------------------------
# Kendall Tau-b overlay in figure subplots
# -------------------------------------------------------

def add_kendall_overlay(
    fig,
    df,
    value_col,
    period_col,
    row,
    col,
    period_order,
    name="Kendall Tau-b",
    annotate_mean=True,
    overlay_idx=10,
    show_overlay_x=True,
    show_overlay_y=True,
    n_boot=1000,
    rng=None,
):
    """
    Overlay Kendall Tau-b between consecutive periods (T1 vs T2, T2 vs T3, ...).

    - Observed τ for each pair is computed on the *intersection of the two periods*.
    - Bootstrap 95% CI per pair is also based only on that pairwise intersection.
    - Global ⟨τ⟩ and its 95% CI are obtained by averaging the per-pair bootstrap
      τ's across pairs, *per bootstrap draw*.
    """

    if rng is None:
        rng = np.random.default_rng(42)

    # ---- Helper: bootstrap for a single pair ----
    def bootstrap_kendall_pair(r1, r2, n_boot=1000, rng=None):
        r1 = np.asarray(r1)
        r2 = np.asarray(r2)
        n = len(r1)
        boot = np.empty(n_boot, dtype=float)

        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            tau_b = stats.kendalltau(r1[idx], r2[idx]).correlation
            boot[b] = tau_b

        return {
            "mean": np.nanmean(boot),
            "sd": np.nanstd(boot, ddof=1),
            "ci_low": np.nanpercentile(boot, 2.5),
            "ci_high": np.nanpercentile(boot, 97.5),
            "boot_samples": boot,
        }

    # ---- Identify entity key ----
    if "node" in df.columns:
        key = "node"
    elif "names" in df.columns:
        key = "names"
    elif "short_name" in df.columns:
        key = "short_name"
    else:
        key = df.columns[0]

    # ---- Pre-split dataframe by period ----
    period_dfs = {
        p: df[df[period_col] == p].copy()
        for p in period_order
    }

    # ---- Compute observed τ and pairwise bootstraps ----
    tau_results = {}       # (p1, p2) -> observed tau
    tau_vals = []          # observed tau in period-order sequence
    ci_vals = []           # (ci_low, ci_high) in period-order sequence
    pair_boot_samples = [] # list of 1D arrays, one per valid pair

    for i in range(len(period_order) - 1):
        p1, p2 = period_order[i], period_order[i + 1]
        df1 = period_dfs[p1]
        df2 = period_dfs[p2]

        # Pairwise intersection for this period pair
        common_pair = set(df1[key]).intersection(df2[key])
        if len(common_pair) < 2:
            tau_results[(p1, p2)] = np.nan
            tau_vals.append(np.nan)
            ci_vals.append((np.nan, np.nan))
            pair_boot_samples.append(None)
            continue

        common_pair = sorted(common_pair)

        df1_aligned = df1.set_index(key).loc[common_pair]
        df2_aligned = df2.set_index(key).loc[common_pair]

        r1 = df1_aligned[value_col].rank(method="average").values
        r2 = df2_aligned[value_col].rank(method="average").values

        # Observed tau on this pairwise intersection
        tau_obs = stats.kendalltau(r1, r2).correlation
        tau_results[(p1, p2)] = tau_obs
        tau_vals.append(tau_obs)

        # Bootstrap on this pairwise intersection
        stats_pair = bootstrap_kendall_pair(r1, r2, n_boot=n_boot, rng=rng)
        ci_vals.append((stats_pair["ci_low"], stats_pair["ci_high"]))
        pair_boot_samples.append(stats_pair["boot_samples"])

    # ---- Global bootstrap mean and CI (across pairs, per draw) ----
    valid_boot = [b for b in pair_boot_samples if b is not None]

    global_stats = None
    if len(valid_boot) > 0:
        # Stack shape: (n_pairs_valid, n_boot)
        boot_mat = np.vstack(valid_boot)
        # For each bootstrap draw b, average τ over pairs
        global_samples = np.nanmean(boot_mat, axis=0)

        global_stats = {
            "mean": np.nanmean(global_samples),
            "sd": np.nanstd(global_samples, ddof=1),
            "ci_low": np.nanpercentile(global_samples, 2.5),
            "ci_high": np.nanpercentile(global_samples, 97.5),
            "boot_samples": global_samples,
        }

    # ---- Prepare labels and positions ----
    tau_labels = [f"T{i} vs. T{i+1}" for i in range(1, len(period_order))]
    x_positions = [i + 0.5 for i in range(len(tau_labels))]

    ci_low = [ci[0] for ci in ci_vals]
    ci_high = [ci[1] for ci in ci_vals]

    upper_error = [
        hi - mu if (not np.isnan(hi) and not np.isnan(mu)) else np.nan
        for hi, mu in zip(ci_high, tau_vals)
    ]
    lower_error = [
        mu - lo if (not np.isnan(lo) and not np.isnan(mu)) else np.nan
        for lo, mu in zip(ci_low, tau_vals)
    ]

    # ---- Add Kendall Tau line trace (observed τ's) ----
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=tau_vals,
            mode="lines+markers",
            line=dict(color="#404040", dash="dashdot", width=1),
            marker=dict(color="#404040", size=3),
            name=name,
            hovertemplate=(
                "Comparison: %{text}"
                "<br>τ(b) (obs) = %{y:.3f}"
                "<br>95% CI (boot) = [%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
                "<extra></extra>"
            ),
            customdata=np.array(ci_vals),
            text=tau_labels,
            showlegend=(row == 1 and col == 1),
            error_y=dict(
                type="data",
                array=upper_error,
                arrayminus=lower_error,
                symmetric=False,
                visible=True,
                color="#404040",
                thickness=1,
                width=2,
            ),
        ),
        row=row,
        col=col,
    )

    # ---- Identify subplot domain ----
    subplot_index = (row - 1) * 3 + col
    xaxis_name = "xaxis" if subplot_index == 1 else f"xaxis{subplot_index}"
    yaxis_name = "yaxis" if subplot_index == 1 else f"yaxis{subplot_index}"

    x_domain = fig.layout[xaxis_name].domain
    y_domain = fig.layout[yaxis_name].domain

    # ---- Overlay axes ----
    overlay_x = f"xaxis{overlay_idx}"
    overlay_y = f"yaxis{overlay_idx}"

    fig.update_layout({
        overlay_x: dict(
            domain=x_domain,
            anchor=f"y{overlay_idx}",
            overlaying=f"x{subplot_index if subplot_index > 1 else ''}",
            side="top",
            tickmode="array",
            tickvals=x_positions,
            ticktext=tau_labels,
            showgrid=True,
            zeroline=False,
            categoryorder="array",
            range=[-0.5, len(period_order) - 0.5],
            showticklabels=show_overlay_x,
        ),
        overlay_y: dict(
            domain=y_domain,
            anchor=f"x{overlay_idx}",
            overlaying=f"y{subplot_index if subplot_index > 1 else ''}",
            side="right",
            title="Kendall τ(b)" if show_overlay_y else "",
            range=[-1, 1],
            tickformat=".2f",
            showticklabels=show_overlay_y,
            dtick=0.5,
            title_standoff=2,
            ticklabelstandoff=2,
        ),
    })

    # Assign last trace to overlay axes
    fig.data[-1].update(xaxis=f"x{overlay_idx}", yaxis=f"y{overlay_idx}")

    # ---- Global ⟨τ⟩ annotation (from pairwise-based global bootstrap) ----
    if annotate_mean and global_stats is not None:
        # tau_mean = global_stats["mean"]
        tau_mean = np.nanmean(tau_vals)
        ci_g_lo = global_stats["ci_low"]
        ci_g_hi = global_stats["ci_high"]
        fig.add_annotation(
            text=f"⟨τ⟩ = {tau_mean:.2f} [{ci_g_lo:.2f}, {ci_g_hi:.2f}]",
            xref=f"x{subplot_index if subplot_index > 1 else ''} domain",
            yref=f"y{subplot_index if subplot_index > 1 else ''} domain",
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=7, color="black"),
            align="right",
        )

    return tau_results


## Used for Random Network Analysis figures !!

# -------------------------------------------------------
# Kendall Tau-b overlay without recomputing (use existing df)
# -------------------------------------------------------

def add_kendall_overlay_no_compute(df, metric,
                           fig, row, col, period_order,
                           overlay_idx, show_overlay_x=True,
                           show_overlay_y=True,
                           annotate_mean=False,
                           global_stats=None,
                           name="Kendall τ(b)"):
# ---- Add Kendall Tau line trace (observed τ's) ----
    mask = df["metric"] == metric
    df = df.loc[mask].copy()
    tau_vals = df["tau_mean"].values
    tau_labels = [f"T{i} vs. T{i+1}" for i in range(1, len(period_order))]
    x_positions = [i + 0.5 for i in range(len(tau_labels))]
    lower_ci = df["tau_ci_lower"].values
    upper_ci = df["tau_ci_upper"].values
    upper_error = upper_ci - tau_vals
    lower_error = tau_vals - lower_ci
    
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=tau_vals,
            mode="lines+markers",
            line=dict(color="#404040", dash="dashdot", width=1),
            marker=dict(color="#404040", size=3),
            name=name,
            hovertemplate=(
                "Comparison: %{text}"
                "<br>τ(b) (obs) = %{y:.3f}"
                "<br>95% CI (boot) = [%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
                "<extra></extra>"
            ),
            customdata=np.stack((lower_ci, upper_ci), axis=-1),
            text=tau_labels,
            showlegend=(row == 1 and col == 1),
            error_y=dict(
                type="data",
                array=upper_error,
                arrayminus=lower_error,
                symmetric=False,
                visible=True,
                color="#404040",
                thickness=1,
                width=2,
            ),
        ),
        row=row,
        col=col,
    )

    # ---- Identify subplot domain ----
    subplot_index = (row - 1) * 3 + col
    xaxis_name = "xaxis" if subplot_index == 1 else f"xaxis{subplot_index}"
    yaxis_name = "yaxis" if subplot_index == 1 else f"yaxis{subplot_index}"

    x_domain = fig.layout[xaxis_name].domain
    y_domain = fig.layout[yaxis_name].domain

    # ---- Overlay axes ----
    overlay_x = f"xaxis{overlay_idx}"
    overlay_y = f"yaxis{overlay_idx}"

    fig.update_layout({
        overlay_x: dict(
            domain=x_domain,
            anchor=f"y{overlay_idx}",
            overlaying=f"x{subplot_index if subplot_index > 1 else ''}",
            side="top",
            tickmode="array",
            tickvals=x_positions,
            ticktext=tau_labels,
            showgrid=True,
            zeroline=False,
            categoryorder="array",
            range=[-0.5, len(period_order) - 0.5],
            showticklabels=show_overlay_x,
        ),
        overlay_y: dict(
            domain=y_domain,
            anchor=f"x{overlay_idx}",
            overlaying=f"y{subplot_index if subplot_index > 1 else ''}",
            side="right",
            title="Kendall τ(b)" if show_overlay_y else "",
            range=[-1, 1],
            tickformat=".2f",
            showticklabels=show_overlay_y,
            dtick=0.5,
            title_standoff=2,
            ticklabelstandoff=2,
        ),
    })

    # Assign last trace to overlay axes
    fig.data[-1].update(xaxis=f"x{overlay_idx}", yaxis=f"y{overlay_idx}")

    # ---- Global ⟨τ⟩ annotation (from pairwise-based global bootstrap) ----
    if annotate_mean and global_stats is not None:
        tau_mean = np.nanmean(tau_vals)
        ci_g_lo = global_stats["ci_low"]
        ci_g_hi = global_stats["ci_high"]
        fig.add_annotation(
            text=f"⟨τ⟩ = {tau_mean:.2f} [{ci_g_lo:.2f}, {ci_g_hi:.2f}]",
            xref=f"x{subplot_index if subplot_index > 1 else ''} domain",
            yref=f"y{subplot_index if subplot_index > 1 else ''} domain",
            x=0.02,
            y=0.98,
            showarrow=False,
            font=dict(size=7, color="black"),
            align="right",
        )


# -------------------------------------------------------
# Bootstrap global mean CI for tau values over period pairs
# -------------------------------------------------------


def bootstrap_global_mean_ci(tau_vals, n_boot=1000, conf=0.95, rng=None):
    """
    Bootstrap CI for the global mean of tau over adjacent period pairs.

    tau_vals: 1D array-like of tau statistics per period pair (e.g. tau_mean from rn_summary)
    """
    tau_vals = np.asarray(tau_vals, float)
    tau_vals = tau_vals[~np.isnan(tau_vals)]
    if tau_vals.size == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    if rng is None:
        rng = np.random.default_rng(123)

    n = tau_vals.size
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = tau_vals[idx].mean()

    alpha = 1.0 - conf
    ci_low, ci_high = np.quantile(boot_means, [alpha/2, 1 - alpha/2])

    return {
        "mean": float(tau_vals.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }