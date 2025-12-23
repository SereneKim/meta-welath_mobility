#!/usr/bin/env python3
"""
Edge capacitance inquiry (cleaned)

This script:
1) Builds a period-specific tripartite projection graph from paper triplets (M, D, R).
2) Computes baseline edge "capacitance" metrics (weighted edge betweenness vs local frequency/strength).
3) For every possible triplet (M, D, R), applies a +1 counterfactual update to its three edges,
   recomputes capacitances, and stores edge-level deltas vs baseline.
4) Extracts, for a target edge (u, v), how much each counterfactual triplet changes its capacitance.

Assumes local project modules are available:
- data_loads (must expose .main and .tri OR at least .main with m_short_name/dt_short_name/rq_short_name/period)
- taxonomy (must expose period_order / period_labels; optional here)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import networkx as nx


# ---------------------------------------------------------------------
# Local project imports
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.getcwd())
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

import scripts.data_loads_old as data_loads_old  # noqa: E402


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    period_name: str = "2001-2005"
    target_u: str = "M4"
    target_v: str = "D11"
    top_k_changes: int = 20  # 0 disables "top changes" view
    keep_other_metrics: bool = False  # kept for compatibility; not used by default


# ---------------------------------------------------------------------
# Helpers: triplets, edges, data selection
# ---------------------------------------------------------------------
TripletKey = Tuple[str, str, str]
EdgeKey = Tuple[str, str]


def choose_period(df: pd.DataFrame, period_col: str, period_name: str) -> pd.DataFrame:
    """Return a copy of rows matching a period."""
    return df.loc[df[period_col] == period_name].copy()


def all_possible_triplets_df(m: Sequence[str], dt: Sequence[str], rq: Sequence[str]) -> pd.DataFrame:
    """Cartesian product of (m, dt, rq) as a DataFrame."""
    return pd.DataFrame(product(m, dt, rq), columns=["m_short_name", "dt_short_name", "rq_short_name"])


def normalize_edge(u: str, v: str) -> EdgeKey:
    """Stable undirected edge key (lexicographic)."""
    return (u, v) if u <= v else (v, u)


def edges_for_triplet(triplet: TripletKey) -> List[EdgeKey]:
    m, d, r = triplet
    return [normalize_edge(m, d), normalize_edge(d, r), normalize_edge(r, m)]


def inc_edge_weight(G: nx.Graph, u: str, v: str, delta: float = 1.0) -> None:
    """Increment (or decrement) an undirected edge weight by delta, removing if <= 0."""
    if G.has_edge(u, v):
        G[u][v]["w"] = float(G[u][v].get("w", 1.0)) + float(delta)
        if G[u][v]["w"] <= 0:
            G.remove_edge(u, v)
    else:
        if delta > 0:
            G.add_edge(u, v, w=float(delta))


def apply_triplet_delta(G: nx.Graph, triplet: TripletKey, delta: float = 1.0) -> None:
    """Apply a +/- delta to the 3 edges implied by a (M, D, R) triplet."""
    m, d, r = triplet
    inc_edge_weight(G, m, d, delta)
    inc_edge_weight(G, d, r, delta)
    inc_edge_weight(G, r, m, delta)


# ---------------------------------------------------------------------
# Graph construction and metrics
# ---------------------------------------------------------------------
def build_tripartite_graph_per_period(p_data: pd.DataFrame) -> nx.Graph:
    """
    Build an undirected projection graph for a single period.
    Each paper triplet adds +1 to edges (M-D), (D-R), (R-M).
    Edge weight 'w' equals the number of papers generating that pair in the period.
    """
    G = nx.Graph()
    for _, row in p_data.iterrows():
        m = row["m_short_name"]
        d = row["dt_short_name"]
        r = row["rq_short_name"]
        inc_edge_weight(G, m, d, 1.0)
        inc_edge_weight(G, d, r, 1.0)
        inc_edge_weight(G, r, m, 1.0)
    return G


def ensure_inverse_weight(G: nx.Graph, weight_attr: str = "w", inv_attr: str = "inv_w") -> None:
    """Add inverse-weight attribute for shortest path computations (1/w)."""
    for _, _, d in G.edges(data=True):
        w = float(d.get(weight_attr, 1.0))
        d[inv_attr] = 1.0 / w if w > 0 else 1e12


def compute_edge_capacitances(G: nx.Graph) -> pd.DataFrame:
    """
    Edge-level quantities:
      - pair_count (w)
      - weighted edge betweenness (B_e^w using inv_w)
      - cap_pair_norm  = B_e^w * max(pair_count) / pair_count
      - edge_strength  = strength(u) + strength(v) - 2*pair_count(u,v)
      - cap_edge_strength = B_e^w / edge_strength
    """
    if G.number_of_edges() == 0:
        return pd.DataFrame(
            columns=[
                "u",
                "v",
                "pair_count",
                "edge_betweenness_weighted",
                "max_pair_count",
                "cap_pair_norm",
                "edge_strength",
                "cap_edge_strength",
            ]
        )

    ensure_inverse_weight(G)

    ebc_w: Mapping[Tuple[str, str], float] = nx.edge_betweenness_centrality(G, weight="inv_w", normalized=True)
    ebc_uw: Mapping[Tuple[str, str], float] = nx.edge_betweenness_centrality(G, weight=None, normalized=True)
    
    deg = dict(G.degree())

    edge_degree = {normalize_edge(u, v): max(deg[u] + deg[v] - 2, 1) for u, v in G.edges()}

    strength = {
        n: float(sum(ed.get("w", 1.0) for _, _, ed in G.edges(n, data=True)))
        for n in G.nodes()
    }

    pair_count: Dict[EdgeKey, float] = {normalize_edge(u, v): float(d.get("w", 1.0)) for u, v, d in G.edges(data=True)}
    max_pair = max(pair_count.values()) if pair_count else 1.0

    rows = []
    for (u, v), bw in ebc_w.items():
        uu, vv = normalize_edge(u, v)
        w = pair_count.get((uu, vv), 0.0)
        ed = edge_degree.get((uu, vv), 0.0)

        cap_pair_norm = (bw * max_pair / w) if w > 0 else np.nan
        es = strength.get(u, 0.0) + strength.get(v, 0.0) - 2.0 * w
        cap_edge_strength = (bw / es) if es > 0 else np.nan
        cap_unweighted_edge = (ebc_uw.get((uu, vv), 0.0) / ed) if ed > 0 else np.nan

        rows.append(
            {
                "u": uu,
                "v": vv,
                "pair_count": w,
                "edge_betweenness_weighted": float(bw),
                "edge_betweenness_unweighted": float(ebc_uw.get((uu, vv), 0.0)),
                "edge_degree": float(ed),
                "max_pair_count": float(max_pair),
                "cap_pair_norm": float(cap_pair_norm) if np.isfinite(cap_pair_norm) else np.nan,
                "edge_strength": float(es),
                "cap_edge_strength": float(cap_edge_strength) if np.isfinite(cap_edge_strength) else np.nan,
                "cap_unweighted_edge": float(cap_unweighted_edge) if np.isfinite(cap_unweighted_edge) else np.nan,
            }
        )

    caps = pd.DataFrame(rows).sort_values(["u", "v"]).reset_index(drop=True)
    return caps


def capacitance_delta(base_caps: pd.DataFrame, new_caps: pd.DataFrame) -> pd.DataFrame:
    """Compute Δcap = new - base for each undirected edge."""
    keep = ["u", "v", "pair_count", "cap_pair_norm", "edge_strength", "cap_edge_strength", "cap_unweighted_edge", "edge_degree"]

    b = base_caps[keep].copy()
    n = new_caps[keep].copy()

    b["edge"] = list(zip(b["u"], b["v"]))
    n["edge"] = list(zip(n["u"], n["v"]))

    merged = b.merge(n, on="edge", how="outer", suffixes=("_base", "_new"))

    merged["u"] = merged["u_base"].combine_first(merged["u_new"])
    merged["v"] = merged["v_base"].combine_first(merged["v_new"])

    merged["d_cap_pair_norm"] = merged["cap_pair_norm_new"] - merged["cap_pair_norm_base"]
    merged["d_cap_edge_strength"] = merged["cap_edge_strength_new"] - merged["cap_edge_strength_base"]
    merged["d_cap_unweighted_edge"] = merged["cap_unweighted_edge_new"] - merged["cap_unweighted_edge_base"]

    out_cols = [
        "u",
        "v",
        "pair_count_base",
        "pair_count_new",
        "cap_pair_norm_base",
        "cap_pair_norm_new",
        "d_cap_pair_norm",
        "edge_strength_base",
        "edge_strength_new",
        "cap_edge_strength_base",
        "cap_edge_strength_new",
        "d_cap_edge_strength",
        "cap_unweighted_edge_base",
        "cap_unweighted_edge_new",
        "d_cap_unweighted_edge",
        "edge_degree_base",
        "edge_degree_new",
    ]
    return merged[out_cols].sort_values(["u", "v"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Counterfactual analysis
# ---------------------------------------------------------------------
DeltaObj = Union[pd.DataFrame, Dict[str, pd.DataFrame]]


def analyze_period_counterfactuals_capacitance(
    df_main: pd.DataFrame,
    period_name: str,
    all_possible_triplets: pd.DataFrame,
    top_k_changes: int = 0,
) -> Dict[str, object]:
    """
    For each possible (M, D, R), apply +1 to its three edges and compute edge-level capacitance deltas.

    Returns dict with:
      - baseline_caps: DataFrame
      - caps_deltas: dict[triplet_key -> DataFrame] OR dict[triplet_key -> {"delta_all": df, "delta_top": df}]
    """
    p_data = choose_period(df_main, "period", period_name)
    G0 = build_tripartite_graph_per_period(p_data)

    base_caps = compute_edge_capacitances(G0)
    caps_deltas: Dict[TripletKey, DeltaObj] = {}

    for trip in all_possible_triplets.itertuples(index=False):
        key: TripletKey = (trip.m_short_name, trip.dt_short_name, trip.rq_short_name)

        apply_triplet_delta(G0, key, delta=1.0)
        new_caps = compute_edge_capacitances(G0)
        d_caps = capacitance_delta(base_caps, new_caps)

        if top_k_changes and top_k_changes > 0 and not d_caps.empty:
            absmax = np.nanmax(
                np.vstack(
                    [
                        np.abs(d_caps["d_cap_pair_norm"].to_numpy(dtype=float)),
                        np.abs(d_caps["d_cap_edge_strength"].to_numpy(dtype=float)),
                        np.abs(d_caps["d_cap_unweighted_edge"].to_numpy(dtype=float)),
                    ]
                ),
                axis=0,
            )
            d_caps2 = d_caps.copy()
            d_caps2["_absmax"] = absmax
            top = d_caps2.sort_values("_absmax", ascending=False).head(top_k_changes).drop(columns="_absmax")
            caps_deltas[key] = {"delta_all": d_caps, "delta_top": top}
        else:
            caps_deltas[key] = d_caps

        apply_triplet_delta(G0, key, delta=-1.0)

    return {"baseline_caps": base_caps, "caps_deltas": caps_deltas}


def extract_delta_df(obj: DeltaObj) -> pd.DataFrame:
    """Normalize a caps_deltas value to a DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        if "delta_all" in obj and isinstance(obj["delta_all"], pd.DataFrame):
            return obj["delta_all"]
        for v in obj.values():
            if isinstance(v, pd.DataFrame):
                return v
        raise TypeError("caps_deltas dict entry contains no DataFrame.")
    raise TypeError(f"caps_deltas entry is {type(obj)}; expected DataFrame or dict.")


def get_target_edge_delta_row(delta_obj: DeltaObj, u: str, v: str) -> Optional[pd.Series]:
    """Return the delta row for a specific undirected edge (u, v), or None if not present."""
    df = extract_delta_df(delta_obj)
    uu, vv = normalize_edge(u, v)
    hit = df[(df["u"] == uu) & (df["v"] == vv)]
    return None if hit.empty else hit.iloc[0]


def connectivity_features(triplet_key: TripletKey, G_baseline: nx.Graph, u: str, v: str) -> Dict[str, float]:
    """Simple proximity features of a triplet to a target edge endpoints in the baseline graph."""
    # m, d, r = triplet_key
    # touch = int(m == u) + int(d == v)
    # direct_edge_added = int((m == u) and (d == v))
    
    nodes = set(triplet_key)
    
    touch_u = int(u in nodes)
    touch_v = int(v in nodes)
    touch = touch_u + touch_v
    direct_edge_added = int((u in nodes) and (v in nodes))

    # nodes = [m, d, r]

    def dist(a: str, b: str) -> float:
        if a not in G_baseline or b not in G_baseline:
            return float("inf")
        try:
            return float(nx.shortest_path_length(G_baseline, a, b))
        except nx.NetworkXNoPath:
            return float("inf")

    nodes_list = list(nodes)
    min_dist_to_u = min(dist(x, u) for x in nodes_list)
    min_dist_to_v = min(dist(x, v) for x in nodes_list)

    return {
        "touch": float(touch),
        "touch_u": float(touch_u),
        "touch_v": float(touch_v),
        "direct_edge_added": float(direct_edge_added),
        "min_dist_either": float(min(min_dist_to_u, min_dist_to_v)),
        f"min_dist_to_{u}": float(min_dist_to_u),
        f"min_dist_to_{v}": float(min_dist_to_v),
        "sum_min_dist": float(min_dist_to_u + min_dist_to_v),
    }


def build_edge_impact_table(
    df_main: pd.DataFrame,
    period_name: str,
    caps_deltas: Mapping[TripletKey, DeltaObj],
    target_u: str,
    target_v: str,
) -> Tuple[pd.DataFrame, nx.Graph]:
    """Build a row per counterfactual triplet describing how it changes the target edge capacitances."""
    p_data = choose_period(df_main, "period", period_name)
    G_base = build_tripartite_graph_per_period(p_data)

    rows = []
    for triplet_key, delta_obj in caps_deltas.items():
        row = get_target_edge_delta_row(delta_obj, target_u, target_v)

        if row is None:
            d1 = np.nan
            d2 = np.nan
            d3 = np.nan
            pc_base = np.nan
            pc_new = np.nan
        else:
            d1 = row.get("d_cap_pair_norm", np.nan)
            d2 = row.get("d_cap_edge_strength", np.nan)
            d3 = row.get("d_cap_unweighted_edge", np.nan)
            pc_base = row.get("pair_count_base", np.nan)
            pc_new = row.get("pair_count_new", np.nan)

        feats = connectivity_features(triplet_key, G_base, target_u, target_v)

        m, dt, rq = triplet_key
        rows.append(
            {
                "m": m,
                "dt": dt,
                "rq": rq,
                "triplet": f"{m}-{dt}-{rq}",
                "d_cap_pair_norm": d1,
                "d_cap_edge_strength": d2,
                "d_cap_unweighted_edge": d3,
                "pair_count_base": pc_base,
                "pair_count_new": pc_new,
                **feats,
            }
        )

    return pd.DataFrame(rows), G_base


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(cfg: Config) -> pd.DataFrame:
    df_main = data_loads_old.main

    all_possible_triplets = all_possible_triplets_df(
        df_main["m_short_name"].unique().tolist(),
        df_main["dt_short_name"].unique().tolist(),
        df_main["rq_short_name"].unique().tolist(),
    )

    out = analyze_period_counterfactuals_capacitance(
        df_main=df_main,
        period_name=cfg.period_name,
        all_possible_triplets=all_possible_triplets,
        top_k_changes=cfg.top_k_changes,
    )

    impact_df, _ = build_edge_impact_table(
        df_main=df_main,
        period_name=cfg.period_name,
        caps_deltas=out["caps_deltas"],  # type: ignore[arg-type]
        target_u=cfg.target_u,
        target_v=cfg.target_v,
    )

    return impact_df


if __name__ == "__main__":
    cfg = Config()
    impact_df = main(cfg)
    # Print a quick sanity view
    print(impact_df.head(10))




# ---------------------------------------------------------------------
# Counterfactual removals (realized triplets only)
# ---------------------------------------------------------------------
def realized_triplets_in_period(
    df_main: pd.DataFrame,
    period_name: str,
    period_col: str = "period",
) -> pd.DataFrame:
    """
    Return unique realized triplets (m, dt, rq) in the given period.
    Also returns an 'n' column = how many times the triplet occurs (paper count).
    """
    p_data = choose_period(df_main, period_col, period_name)
    g = (
        p_data.groupby(["m_short_name", "dt_short_name", "rq_short_name"])
        .size()
        .reset_index(name="n")
        .sort_values(["n", "m_short_name", "dt_short_name", "rq_short_name"], ascending=[False, True, True, True])
        .reset_index(drop=True)
    )
    return g


def analyze_period_realized_removals_capacitance(
    df_main: pd.DataFrame,
    period_name: str,
    realized_triplets: Optional[pd.DataFrame] = None,
    top_k_changes: int = 0,
) -> Dict[str, object]:
    """
    For each REALIZED (M, D, R) in this period, apply -1 to its three edges
    and compute edge-level capacitance deltas vs baseline.

    Returns dict with:
      - baseline_caps: DataFrame
      - realized_triplets: DataFrame with columns [m_short_name, dt_short_name, rq_short_name, n]
      - caps_deltas_removed: dict[triplet_key -> DataFrame or {"delta_all": df, "delta_top": df}]
    """
    p_data = choose_period(df_main, "period", period_name)
    G0 = build_tripartite_graph_per_period(p_data)

    base_caps = compute_edge_capacitances(G0)

    if realized_triplets is None:
        realized_triplets = realized_triplets_in_period(df_main, period_name)

    caps_deltas_removed: Dict[TripletKey, DeltaObj] = {}

    for trip in realized_triplets.itertuples(index=False):
        key: TripletKey = (trip.m_short_name, trip.dt_short_name, trip.rq_short_name)

        # Remove ONE occurrence (one paper) of this realized triplet
        apply_triplet_delta(G0, key, delta=-1.0)
        new_caps = compute_edge_capacitances(G0)
        d_caps = capacitance_delta(base_caps, new_caps)

        if top_k_changes and top_k_changes > 0 and not d_caps.empty:
            absmax = np.nanmax(
                np.vstack(
                    [
                        np.abs(d_caps["d_cap_pair_norm"].to_numpy(dtype=float)),
                        np.abs(d_caps["d_cap_edge_strength"].to_numpy(dtype=float)),
                        np.abs(d_caps["d_cap_unweighted_edge"].to_numpy(dtype=float)),
                    ]
                ),
                axis=0,
            )
            d_caps2 = d_caps.copy()
            d_caps2["_absmax"] = absmax
            top = d_caps2.sort_values("_absmax", ascending=False).head(top_k_changes).drop(columns="_absmax")
            caps_deltas_removed[key] = {"delta_all": d_caps, "delta_top": top}
        else:
            caps_deltas_removed[key] = d_caps

        # Revert removal
        apply_triplet_delta(G0, key, delta=+1.0)

    return {
        "baseline_caps": base_caps,
        "realized_triplets": realized_triplets,
        "caps_deltas_removed": caps_deltas_removed,
    }


# ---------------------------------------------------------------------
# Impact of one triplet on the entire network
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd
import networkx as nx

# --- assumes these are already in scope from your script ---
# choose_period, build_tripartite_graph_per_period
# compute_edge_capacitances, capacitance_delta
# apply_triplet_delta, edges_for_triplet, normalize_edge


def network_impact_of_one_triplet(
    df_main: pd.DataFrame,
    period_name: str,
    triplet_key: tuple[str, str, str],
    *,
    delta: float = 1.0,
    top_k: int = 30,
    restrict_to_two_edges: bool = False,
) -> dict[str, object]:
    """
    Add ONE occurrence of `triplet_key` (delta=+1) to the period graph and measure
    how *all edges'* capacitances change.

    If restrict_to_two_edges=True, only apply the update to (m,dt) and (dt,rq)
    (i.e., treat the triplet as a 2-edge event rather than the full 3-pair projection).
    """

    # 1) baseline
    p_data = choose_period(df_main, "period", period_name)
    G0 = build_tripartite_graph_per_period(p_data)
    base_caps = compute_edge_capacitances(G0)

    # 2) counterfactual update
    m, dt, rq = triplet_key
    if restrict_to_two_edges:
        # Only two edges: (m,dt) and (dt,rq)
        inc_edge_weight(G0, m, dt, delta)
        inc_edge_weight(G0, dt, rq, delta)
        touched = [normalize_edge(m, dt), normalize_edge(dt, rq)]
    else:
        apply_triplet_delta(G0, triplet_key, delta=delta)
        touched = edges_for_triplet(triplet_key)  # 3 edges in your projection

    # 3) recompute + deltas for ALL edges
    new_caps = compute_edge_capacitances(G0)
    d_caps = capacitance_delta(base_caps, new_caps)

    # 4) convenience subsets + summary
    d_caps["edge"] = list(zip(d_caps["u"], d_caps["v"]))
    touched_set = set(touched)
    delta_touched = d_caps[d_caps["edge"].isin(touched_set)].copy()

    # Global “impact” metrics (choose what you like; these are robust defaults)
    for col in ["d_cap_pair_norm", "d_cap_edge_strength", "d_cap_unweighted_edge"]:
        d_caps[col] = pd.to_numeric(d_caps[col], errors="coerce")

    abs_pair = d_caps["d_cap_pair_norm"].abs()
    abs_str  = d_caps["d_cap_edge_strength"].abs()
    abs_uw   = d_caps["d_cap_unweighted_edge"].abs()
    summary = {
        "period": period_name,
        "triplet": triplet_key,
        "restrict_to_two_edges": restrict_to_two_edges,
        "n_edges_baseline": int(base_caps.shape[0]),
        "n_edges_new": int(new_caps.shape[0]),
        "n_edges_in_union": int(d_caps.shape[0]),
        "abs_sum_d_cap_pair_norm": float(np.nansum(abs_pair)),
        "abs_sum_d_cap_edge_strength": float(np.nansum(abs_str)),
        "abs_max_d_cap_pair_norm": float(np.nanmax(abs_pair)) if len(abs_pair) else np.nan,
        "abs_max_d_cap_edge_strength": float(np.nanmax(abs_str)) if len(abs_str) else np.nan,
        "n_edges_changed_pair": int(np.nansum(abs_pair > 0)),
        "n_edges_changed_strength": int(np.nansum(abs_str > 0)),
        "abs_sum_d_cap_unweighted_edge": float(np.nansum(abs_uw)),
        "abs_max_d_cap_unweighted_edge": float(np.nanmax(abs_uw)) if len(abs_uw) else np.nan,
        "n_edges_changed_unweighted": int(np.nansum(abs_uw > 0)),
    }

    # rank edges by largest effect (combined)
    d_caps["_absmax"] = np.nanmax(
        np.vstack([abs_pair.to_numpy(), abs_str.to_numpy()]),
        axis=0
    )
    top_changed = (
        d_caps.sort_values("_absmax", ascending=False)
        .drop(columns=["_absmax"])
        .head(top_k)
        .reset_index(drop=True)
    )
    d_caps = d_caps.drop(columns=["_absmax"])

    return {
        "baseline_caps": base_caps,
        "new_caps": new_caps,
        "delta_all": d_caps.drop(columns=["edge"]),
        "delta_touched": delta_touched.drop(columns=["edge"]),
        "top_changed": top_changed,
        "summary": summary,
    }
