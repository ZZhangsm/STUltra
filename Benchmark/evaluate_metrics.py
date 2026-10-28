#!/usr/bin/env python
"""Evaluate integration/clustering metrics on the cells shared by every method."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

warnings.filterwarnings("ignore")

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
# Avoid OpenBLAS thread overflow on many-core hosts (built with NUM_THREADS=128).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_ST_ANALYSIS = Path(__file__).resolve().parents[1]
if str(_ST_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ST_ANALYSIS))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csgraph
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from metrics import compute_lisi

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / "results"
METHOD_COLUMNS = {
    "graphst": ("emb", "domain"),
    "sedr": ("SEDR.Harmony", "mclust"),
    "banksy": ("reduced_pc", "banksy_cluster"),
    "cellcharter": ("X_cellcharter", "spatial_cluster"),
    "staligner": ("STAligner", "mclust"),
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    path: Path
    embedding_key: str
    cluster_key: str


def result_filename(dataset: str, method: str) -> str:
    if dataset:
        return f"{dataset}_{method}.h5ad"
    return f"{method}.h5ad"


def default_methods(results_dir: Path, dataset: str) -> Dict[str, MethodSpec]:
    return {
        name: MethodSpec(name, results_dir / result_filename(dataset, name), embedding_key, cluster_key)
        for name, (embedding_key, cluster_key) in METHOD_COLUMNS.items()
    }


def read_h5ad_compat(path: Path):
    """Read h5ad, skipping incompatible uns when newer writers store null as empty datasets."""
    try:
        return sc.read_h5ad(path)
    except Exception as exc:
        if type(exc).__name__ != "AnnDataReadError":
            raise
    import h5py
    from anndata import AnnData
    from anndata._io.h5ad import read_dataframe
    from anndata._io.specs import read_elem

    with h5py.File(path, "r") as f:
        obs = read_dataframe(f["obs"])
        obsm = {}
        if "obsm" in f:
            obsm = {k: read_elem(f["obsm"][k]) for k in f["obsm"].keys()}
    return AnnData(X=None, obs=obs, obsm=obsm)


def read_h5ad_compat_backed(path: Path):
    try:
        return sc.read_h5ad(path, backed="r")
    except Exception as exc:
        if type(exc).__name__ != "AnnDataReadError":
            raise
    return read_h5ad_compat(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate spatial benchmark metrics")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--dataset", default=os.environ.get("DATASET", ""))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--ours-h5ad", type=Path, default=None)
    p.add_argument("--ours-embedding-key", default="STIntg")
    p.add_argument("--ours-cluster-key", default="louvain_0.7")
    p.add_argument(
        "--reference-method",
        default=None,
        help="Method whose cell order is used for the shared evaluation set",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["graphst", "sedr", "banksy", "cellcharter", "staligner"],
    )
    p.add_argument("--batch-key", default="batch_name")
    p.add_argument("--knn", type=int, default=15)
    p.add_argument("--lisi-perplexity", type=int, default=30)
    return p.parse_args()


def obs_names_for_method(spec: MethodSpec) -> set[str]:
    if not spec.path.is_file():
        raise FileNotFoundError(f"{spec.name}: missing {spec.path}")
    adata = read_h5ad_compat_backed(spec.path)
    try:
        return set(adata.obs_names.astype(str))
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def common_obs_names(
    methods: Dict[str, MethodSpec],
    method_names: Sequence[str],
) -> set[str]:
    common: set[str] | None = None
    for name in method_names:
        names = obs_names_for_method(methods[name])
        common = names if common is None else common & names
    return common or set()


def common_cell_index(
    methods: Dict[str, MethodSpec],
    method_names: Sequence[str],
    reference_spec: MethodSpec,
) -> List[str]:
    common = common_obs_names(methods, method_names)
    if not common:
        raise ValueError("No common cells found across selected method outputs.")

    adata = read_h5ad_compat_backed(reference_spec.path)
    try:
        cell_index = [str(name) for name in adata.obs_names if str(name) in common]
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    missing = common - set(cell_index)
    if missing:
        raise ValueError(f"{len(missing)} common cells are missing from {reference_spec.name}.")
    return cell_index


def _candidate_embedding_keys(adata, preferred: str) -> List[str]:
    keys = list(adata.obsm.keys())
    if preferred in keys:
        return [preferred]
    if preferred == "reduced_pc":
        return [key for key in keys if key.startswith("reduced_pc")] + keys
    return keys


def _require_columns(adata, spec: MethodSpec, batch_key: str) -> None:
    if spec.cluster_key not in adata.obs.columns:
        raise KeyError(f"{spec.name}: obs['{spec.cluster_key}'] missing in {spec.path}")
    if batch_key not in adata.obs.columns:
        for alt in ("sample", "slice_name", "batch"):
            if alt in adata.obs.columns:
                adata.obs[batch_key] = adata.obs[alt].astype(str)
                return
        raise KeyError(f"{spec.name}: no batch column found in {spec.path}")


def _align_or_slice(adata, cell_index: Sequence[str], batch_key: str):
    present = adata.obs_names.isin(cell_index)
    if int(present.sum()) == len(cell_index):
        return adata[list(cell_index)].copy()
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Cannot align {len(cell_index)} common cells without {batch_key}")
    missing = len(cell_index) - int(present.sum())
    raise ValueError(
        f"{missing} common cells are missing from result. "
        "Ensure model scripts preserve obs_names as {cell}_{section}."
    )


def load_method_cells(spec: MethodSpec, cell_index: Sequence[str], batch_key: str):
    if not spec.path.is_file():
        raise FileNotFoundError(f"{spec.name}: missing {spec.path}")
    adata = read_h5ad_compat(spec.path)
    _require_columns(adata, spec, batch_key)
    adata = _align_or_slice(adata, cell_index, batch_key)

    embedding_keys = _candidate_embedding_keys(adata, spec.embedding_key)
    if not embedding_keys:
        raise KeyError(f"{spec.name}: no embeddings in obsm of {spec.path}")
    embedding_key = embedding_keys[0]
    if embedding_key != spec.embedding_key:
        print(f"{spec.name}: using embedding {embedding_key!r} instead of {spec.embedding_key!r}")
    return adata, embedding_key


def _valid_labels(labels: Iterable) -> pd.Series:
    series = pd.Series(labels).astype(str)
    return series.replace({"nan": np.nan, "None": np.nan})


def safe_score(func, x: np.ndarray, labels: Sequence[str]) -> float:
    labels_s = _valid_labels(labels)
    mask = labels_s.notna().to_numpy()
    unique = labels_s[mask].nunique()
    if mask.sum() < 3 or unique < 2 or unique >= mask.sum():
        return np.nan
    return float(func(x[mask], labels_s[mask]))


def graph_connectivity(x: np.ndarray, labels: Sequence[str], n_neighbors: int) -> float:
    labels_s = _valid_labels(labels)
    mask = labels_s.notna().to_numpy()
    x = x[mask]
    labels_s = labels_s[mask].reset_index(drop=True)
    if x.shape[0] < 3 or labels_s.nunique() < 2:
        return np.nan

    n_neighbors = min(n_neighbors + 1, x.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(x)
    graph = nn.kneighbors_graph(x, mode="connectivity")
    graph = graph.maximum(graph.T)

    scores = []
    weights = []
    for cluster in labels_s.unique():
        idx = np.flatnonzero(labels_s.to_numpy() == cluster)
        if len(idx) <= 1:
            scores.append(1.0)
            weights.append(len(idx))
            continue
        subgraph = graph[idx][:, idx]
        _, comp = csgraph.connected_components(subgraph, directed=False)
        largest = np.bincount(comp).max()
        scores.append(float(largest / len(idx)))
        weights.append(len(idx))
    return float(np.average(scores, weights=weights))


def ilisi_batch(x: np.ndarray, batches: Sequence[str], perplexity: int) -> float:
    if x.shape[0] < 4 or pd.Series(batches).astype(str).nunique() < 2:
        return np.nan
    max_perplexity = max(1, (x.shape[0] - 1) // 3)
    perplexity = min(perplexity, max_perplexity)
    values = compute_lisi(
        x,
        pd.DataFrame({"batch": pd.Series(batches).astype(str).to_numpy()}),
        ["batch"],
        perplexity=perplexity,
    )
    return float(np.nanmean(values[:, 0]))


def evaluate_method(
    spec: MethodSpec,
    cell_index: Sequence[str],
    batch_key: str,
    knn: int,
    lisi_perplexity: int,
) -> dict:
    adata, embedding_key = load_method_cells(spec, cell_index, batch_key)
    x = np.asarray(adata.obsm[embedding_key], dtype=np.float64)
    labels = adata.obs[spec.cluster_key].astype(str).to_numpy()
    batches = adata.obs[batch_key].astype(str).to_numpy()

    row = {
        "method": spec.name,
        "n_cells": int(adata.n_obs),
        "embedding_key": embedding_key,
        "cluster_key": spec.cluster_key,
        "silhouette_cluster": safe_score(silhouette_score, x, labels),
        "calinski_harabasz": safe_score(calinski_harabasz_score, x, labels),
        "davies_bouldin": safe_score(davies_bouldin_score, x, labels),
        "silhouette_batch": safe_score(silhouette_score, x, batches),
        "iLISI_batch": ilisi_batch(x, batches, lisi_perplexity),
        "graph_connectivity": graph_connectivity(x, labels, knn),
    }
    print(
        f"{spec.name}: n={row['n_cells']:,}, "
        f"sil_cluster={row['silhouette_cluster']:.4f}, "
        f"iLISI={row['iLISI_batch']:.4f}"
    )
    return row


def main() -> None:
    args = parse_args()
    methods = default_methods(args.results_dir, args.dataset)
    if args.ours_h5ad is not None:
        methods["ours"] = MethodSpec(
            "ours", args.ours_h5ad, args.ours_embedding_key, args.ours_cluster_key
        )
    elif "ours" in args.methods:
        raise SystemExit("ERROR: method 'ours' requires --ours-h5ad")

    for method in args.methods:
        if method not in methods:
            raise ValueError(f"Unknown method {method!r}; choose from {sorted(methods)}")

    if args.reference_method:
        reference_name = args.reference_method
    elif "ours" in args.methods:
        reference_name = "ours"
    else:
        reference_name = args.methods[0]
    if reference_name not in args.methods:
        raise ValueError(f"reference method {reference_name!r} is not in --methods")

    cell_index = common_cell_index(
        methods,
        args.methods,
        methods[reference_name],
    )
    print(f"Evaluation cells: {len(cell_index):,} (common across methods)")

    rows = []
    for method in args.methods:
        rows.append(
            evaluate_method(
                methods[method],
                cell_index,
                args.batch_key,
                args.knn,
                args.lisi_perplexity,
            )
        )

    df = pd.DataFrame(rows)
    col_order = [
        "method",
        "n_cells",
        "embedding_key",
        "cluster_key",
        "silhouette_cluster",
        "calinski_harabasz",
        "davies_bouldin",
        "silhouette_batch",
        "iLISI_batch",
        "graph_connectivity",
    ]
    df = df[col_order]

    summary_name = f"{args.dataset}_metrics_summary.csv" if args.dataset else "metrics_summary.csv"
    out = args.output or (args.results_dir / summary_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
