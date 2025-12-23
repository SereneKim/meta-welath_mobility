"""
Project data loader

This module centralizes all CSV I/O and the derived, *analysis-ready* DataFrames
used across notebooks/scripts.

Design goals
- Keep the public API stable: variables like `main`, `pairs`, `tri`,
  `all_top_degree_df`, ... are still created at import time (as you requested).
- Remove duplicated boilerplate via small helpers.
- Make paths explicit and overridable (env var PROJECT_ROOT).
- Keep transformations deterministic and in one place.

Notes
- This file assumes local project modules `graph` and `taxonomy` exist.
- All normalization uses `graph.norm` and short-name mappings use `taxonomy.short_names`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

import graph
import taxonomy


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    project_root: str

    @property
    def data_abstracts(self) -> str:
        return os.path.join(self.project_root, "data_abstracts")

    @property
    def results(self) -> str:
        return os.path.join(self.project_root, "results")

    def periods_dir(self, subdir: str) -> str:
        return os.path.join(self.results, "feature-only-KG", "periods", subdir)

    def kg_dir(self, *parts: str) -> str:
        return os.path.join(self.results, "feature-only-KG", *parts)


# Default: notebook root is one level below project root in your setup.
_PROJECT_ROOT = os.environ.get("PROJECT_ROOT") or os.path.dirname(os.getcwd())
PATHS = Paths(project_root=_PROJECT_ROOT)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _read_csv(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def _load_period_csv_folder(folder: str, *, period_from_filename: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Read all .csv files in a folder and return {period -> df}.

    Convention used in your exports: "<PERIOD>_....csv" so we take `file.split('_')[0]`.
    """
    out: Dict[str, pd.DataFrame] = {}
    for file in os.listdir(folder):
        if not file.endswith(".csv"):
            continue
        period = file.split("_")[0] if period_from_filename else file
        df = _read_csv(os.path.join(folder, file))
        df["period"] = period
        out[period] = df
    return out


def _concat_period_dfs(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(d.values(), ignore_index=True) if d else pd.DataFrame()


def _add_short_name_and_group(df: pd.DataFrame, node_col: str = "node") -> pd.DataFrame:
    """Add `short_name` and `color_group` for node-level tables."""
    df = df.copy()
    df["short_name"] = df[node_col].apply(graph.norm).map(taxonomy.short_names)
    df["color_group"] = df[node_col].apply(graph.norm).map(taxonomy.NODE_HIGHLIGHTS).fillna("Other")
    return df


def _add_pair_short_names_and_group(df: pd.DataFrame, u_col: str, v_col: str, *, name_col: str = "name") -> pd.DataFrame:
    """Add from/to short names and highlight group for edge-level tables."""
    df = df.copy()
    df["from_short_name"] = df[u_col].apply(graph.norm).map(taxonomy.short_names)
    df["to_short_name"] = df[v_col].apply(graph.norm).map(taxonomy.short_names)
    df["short_name"] = df["from_short_name"] + "-" + df["to_short_name"]
    df[name_col] = df[u_col] + " - " + df[v_col]
    df["color_group"] = df[name_col].apply(graph.norm).map(taxonomy.PAIR_HIGHLIGHTS).fillna("Other")
    return df


def _normalize_betweenness(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """
    Add `score_norm` = betweenness normalized by (n-1)(n-2) for an un-directed graph.

    Your exported per-period node betweenness CSVs appear to contain one row per node,
    so `n_nodes = len(df)` is appropriate.
    """
    df = df.copy()
    n_nodes = int(df.shape[0])
    if n_nodes > 2:
        factor = 1.0 / ((n_nodes - 1) * (n_nodes - 2))
        df["score_norm"] = df[score_col] * factor
    else:
        df["score_norm"] = df[score_col]
    return df


# -----------------------------------------------------------------------------
# Core datasets
# -----------------------------------------------------------------------------
def load_main() -> pd.DataFrame:
    df = _read_csv(os.path.join(PATHS.data_abstracts, "true_mobility_studies_617_forKGs_cleaned.csv"))

    # Period binning
    df["period"] = pd.cut(
        df["year"],
        bins=[1900, 2000, 2005, 2010, 2015, 2020, 2025],
        right=True,
        labels=["-2000", "2001-2005", "2006-2010", "2011-2015", "2016-2020", "2021-2025"],
    )

    # Normalize categories
    df["category_1"] = df["category_1"].replace({"Others": "Others_Measure"}).apply(graph.norm)
    df["data_cat"] = df["data_cat"].replace({"Others": "Others_DataType"}).apply(graph.norm)
    df["rq_cat"] = df["rq_cat"].replace({"Others": "Others_RqType"}).apply(graph.norm)

    # Short names
    df["m_short_name"] = df["category_1"].map(taxonomy.short_names)
    df["dt_short_name"] = df["data_cat"].map(taxonomy.short_names)
    df["rq_short_name"] = df["rq_cat"].map(taxonomy.short_names)

    return df


def load_top_degree() -> pd.DataFrame:
    d = _load_period_csv_folder(PATHS.periods_dir("top_degree"))
    df = _concat_period_dfs(d)
    return _add_short_name_and_group(df, node_col="node")


def load_top_strength() -> pd.DataFrame:
    d = _load_period_csv_folder(PATHS.periods_dir("top_strength"))
    df = _concat_period_dfs(d)
    return _add_short_name_and_group(df, node_col="node")


def load_norm_degree() -> pd.DataFrame:
    d = _load_period_csv_folder(PATHS.periods_dir("degree_normalized"))
    df = _concat_period_dfs(d)
    return _add_short_name_and_group(df, node_col="node")


def load_top_betweenness(*, weighted: bool = True) -> pd.DataFrame:
    folder = "top_betweenness" if weighted else "top_betweenness_noweight"
    d = _load_period_csv_folder(PATHS.periods_dir(folder))
    df = _concat_period_dfs(d)
    df = _normalize_betweenness(df, score_col="score")
    return _add_short_name_and_group(df, node_col="node")


def load_edge_betweenness() -> pd.DataFrame:
    d = _load_period_csv_folder(PATHS.periods_dir("edge_betweenness"))
    df = _concat_period_dfs(d)
    df = _add_pair_short_names_and_group(df, u_col="u", v_col="v", name_col="name")
    return df


def load_pairs_with_metrics(
    *,
    edge_betweenness_df: pd.DataFrame,
    top_strength_df: pd.DataFrame,
    top_degree_df: pd.DataFrame,
) -> pd.DataFrame:
    pairs = _read_csv(PATHS.kg_dir("pair_counts_perYear.csv"))

    # Pair names
    pairs["from_short_name"] = pairs["from_name"].apply(graph.norm).map(taxonomy.short_names)
    pairs["to_short_name"] = pairs["to_name"].apply(graph.norm).map(taxonomy.short_names)
    pairs["short_name"] = pairs["from_short_name"] + "-" + pairs["to_short_name"]
    pairs["name"] = pairs["from_name"] + " - " + pairs["to_name"]
    pairs["color_group"] = pairs["name"].apply(graph.norm).map(taxonomy.PAIR_HIGHLIGHTS).fillna("Other")

    # Lookups via merges (faster and less error-prone than row-wise apply)
    str_df = top_strength_df[["short_name", "period", "strength"]].rename(columns={"strength": "strength_val"})
    deg_df = top_degree_df[["short_name", "period", "score"]].rename(columns={"score": "degree_val"})
    eb_df = edge_betweenness_df[
        ["short_name", "period", "edge_betweenness", "edge_betweenness_weighted"]
    ]

    # Strength for endpoints (to / from)
    pairs = pairs.merge(
        str_df.rename(columns={"short_name": "to_short_name", "strength_val": "to_strength"}),
        on=["to_short_name", "period"],
        how="left",
    )
    pairs = pairs.merge(
        str_df.rename(columns={"short_name": "from_short_name", "strength_val": "from_strength"}),
        on=["from_short_name", "period"],
        how="left",
    )

    # Degree for endpoints
    pairs = pairs.merge(
        deg_df.rename(columns={"short_name": "to_short_name", "degree_val": "to_degree"}),
        on=["to_short_name", "period"],
        how="left",
    )
    pairs = pairs.merge(
        deg_df.rename(columns={"short_name": "from_short_name", "degree_val": "from_degree"}),
        on=["from_short_name", "period"],
        how="left",
    )

    # Edge betweenness (weighted + unweighted)
    pairs = pairs.merge(eb_df, on=["short_name", "period"], how="left")

    return pairs


def load_triangles() -> pd.DataFrame:
    tri = _read_csv(PATHS.kg_dir("triangle_counts_papers.csv"))
    tri["m.name"] = tri["m.name"].apply(graph.norm)
    tri["dt.name"] = tri["dt.name"].apply(graph.norm)
    tri["rq.name"] = tri["rq.name"].apply(graph.norm)
    tri["m_short_name"] = tri["m.name"].map(taxonomy.short_names)
    tri["dt_short_name"] = tri["dt.name"].map(taxonomy.short_names)
    tri["rq_short_name"] = tri["rq.name"].map(taxonomy.short_names)
    tri["short_name"] = tri["dt_short_name"] + "-" + tri["m_short_name"] + "-" + tri["rq_short_name"]
    tri["name"] = "[" + tri["dt.name"] + ", " + tri["m.name"] + ", " + tri["rq.name"] + "]"
    tri["color_group"] = tri["name"].map(taxonomy.TRIANGLE_HIGHLIGHTS).fillna("Other")
    return tri


# -----------------------------------------------------------------------------
# Public module variables (kept for backwards compatibility)
# -----------------------------------------------------------------------------
main = load_main()

all_top_degree_df = load_top_degree()
all_top_strength_df = load_top_strength()
all_norm_degree_df = load_norm_degree()

all_top_betweenness_df = load_top_betweenness(weighted=True)
all_top_betweenness_noweight_df = load_top_betweenness(weighted=False)

all_edge_betweenness_df = load_edge_betweenness()

pairs = load_pairs_with_metrics(
    edge_betweenness_df=all_edge_betweenness_df,
    top_strength_df=all_top_strength_df,
    top_degree_df=all_top_degree_df,
)

tri = load_triangles()
