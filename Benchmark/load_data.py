#!/usr/bin/env python
"""Shared data loading helpers for multi-slice spatial benchmark scripts.

Each section is one h5ad in ``--data-dir``:

* ``{section_id}.h5ad``
* ``{section_id}_data.h5ad``

``obsm['spatial']`` is used when present. Otherwise coordinates are taken from
common obs columns (Xenium centroids, Visium array/pixel columns, or x/y).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix, issparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

_SPATIAL_COLUMN_PAIRS = (
    ("x_centroid", "y_centroid"),
    ("x_coord", "y_coord"),
    ("x", "y"),
    ("X", "Y"),
    ("array_col", "array_row"),
    ("pxl_col_in_fullres", "pxl_row_in_fullres"),
)
_SKIP_H5AD_STEMS = {"concat", "merged", "all"}
_SECTION_PATHS: Dict[str, str] = {}


def normalize_total(
    adata: ad.AnnData,
    *,
    target_sum: float = 1e4,
    layer: str | None = None,
) -> None:
    """Library-size normalize (in-place), equivalent to sc.pp.normalize_total.

    With NUMBA_DISABLE_JIT=1, scanpy 1.11.x raises UnboundLocalError in
    _normalize_csr because counts_per_cols is only set when
    exclude_highly_expressed=True.
    """
    rep = adata.layers[layer] if layer is not None else adata.X
    X = rep.tocsr() if issparse(rep) and hasattr(rep, "tocsr") else rep
    if np.issubdtype(X.dtype, np.integer):
        X = X.astype(np.float32)
    if issparse(X):
        counts = np.asarray(X.sum(axis=1)).ravel()
        scale = np.zeros(counts.shape, dtype=np.float64)
        np.divide(target_sum, counts, out=scale, where=counts > 0)
        X = csr_matrix(X.multiply(scale[:, np.newaxis]))
    else:
        X = np.asarray(X, dtype=np.float32)
        counts = X.sum(axis=1)
        scale = np.divide(
            target_sum,
            counts,
            out=np.zeros_like(counts, dtype=np.float64),
            where=counts > 0,
        )
        X = X * scale[:, np.newaxis]
    if layer is not None:
        adata.layers[layer] = X
    else:
        adata.X = X


def add_n_clusters_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--n-clusters",
        type=int,
        required=True,
        help="Number of spatial domains",
    )


def add_data_args(parser: argparse.ArgumentParser, *, with_output: bool = True) -> None:
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("DATA_DIR", ""),
        help="Directory of per-section h5ad files",
    )
    parser.add_argument(
        "--section-ids",
        nargs="+",
        default=None,
        help="Section ids or h5ad paths. Default: every section h5ad in --data-dir",
    )
    parser.add_argument(
        "--dataset",
        default=os.environ.get("DATASET", ""),
        help="Prefix for result files. Default: basename of --data-dir",
    )
    if with_output:
        parser.add_argument("--output", default=None, help="Output h5ad path")


def _stem_to_section_id(filename: str) -> str:
    stem = filename[:-5] if filename.endswith(".h5ad") else filename
    if stem.endswith("_data"):
        stem = stem[: -len("_data")]
    return stem


def _conventional_path(data_dir: str, section_id: str) -> str:
    direct = os.path.join(data_dir, f"{section_id}.h5ad")
    alt = os.path.join(data_dir, f"{section_id}_data.h5ad")
    if os.path.isfile(direct):
        return direct
    if os.path.isfile(alt):
        return alt
    return direct


def resolve_section(spec: str, data_dir: str) -> Tuple[str, str]:
    """Return ``(section_id, h5ad path)`` for an id or a path."""
    if not spec.endswith(".h5ad") and os.path.sep not in spec:
        cached = _SECTION_PATHS.get(spec)
        if cached and os.path.isfile(cached):
            return spec, cached
    path_like = spec.endswith(".h5ad") or os.path.sep in spec
    if path_like:
        candidates = []
        if os.path.isabs(spec):
            candidates.append(spec)
        candidates.append(os.path.join(data_dir, spec))
        candidates.append(os.path.join(data_dir, os.path.basename(spec)))
        path = next((item for item in candidates if os.path.isfile(item)), candidates[0])
        return _stem_to_section_id(os.path.basename(path)), path
    return spec, _conventional_path(data_dir, spec)


def section_path(data_dir: str, section_id: str) -> str:
    cached = _SECTION_PATHS.get(section_id)
    if cached:
        return cached
    return _conventional_path(data_dir, section_id)


def discover_section_ids(data_dir: str) -> List[str]:
    if not data_dir or not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir!r}. Set --data-dir or DATA_DIR."
        )
    found: List[str] = []
    seen = set()
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".h5ad") or name.startswith("."):
            continue
        section_id = _stem_to_section_id(name)
        if section_id in _SKIP_H5AD_STEMS:
            continue
        path = os.path.join(data_dir, name)
        if section_id in seen:
            if name == f"{section_id}.h5ad":
                _SECTION_PATHS[section_id] = path
            continue
        seen.add(section_id)
        _SECTION_PATHS[section_id] = path
        found.append(section_id)
    if not found:
        raise FileNotFoundError(f"No section .h5ad files in {data_dir}")
    return found


def default_output_path(method: str, dataset: str) -> str:
    safe = dataset.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return os.path.join(SCRIPT_DIR, "results", f"{safe}_{method}.h5ad")


def torch_device(spec: Optional[str] = None):
    import torch

    if not spec:
        spec = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(spec)


def resolve_run_config(args: argparse.Namespace, method: str) -> argparse.Namespace:
    if not getattr(args, "data_dir", ""):
        raise SystemExit("ERROR: set --data-dir or DATA_DIR to the section h5ad directory.")
    specs = list(args.section_ids) if args.section_ids else discover_section_ids(args.data_dir)
    section_ids: List[str] = []
    for spec in specs:
        section_id, path = resolve_section(spec, args.data_dir)
        if not os.path.isfile(path):
            raise SystemExit(f"ERROR: missing section file for {spec}: {path}")
        _SECTION_PATHS[section_id] = path
        section_ids.append(section_id)
    if not section_ids:
        raise SystemExit(f"ERROR: no sections found in {args.data_dir}")
    args.section_ids = section_ids
    if not getattr(args, "dataset", ""):
        args.dataset = os.path.basename(os.path.abspath(args.data_dir.rstrip(os.sep))) or "dataset"
    if hasattr(args, "output") and not args.output:
        args.output = default_output_path(method, args.dataset)
    print(f"Dataset: {args.dataset}")
    print(f"Sections: {', '.join(section_ids)}")
    return args


def ensure_spatial(adata: ad.AnnData, section_id: str) -> ad.AnnData:
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        if coords.ndim == 2 and coords.shape[1] >= 2:
            adata.obsm["spatial"] = np.asarray(coords[:, :2], dtype=np.float64)
            return adata
    for x_key, y_key in _SPATIAL_COLUMN_PAIRS:
        if {x_key, y_key}.issubset(adata.obs.columns):
            adata.obsm["spatial"] = adata.obs[[x_key, y_key]].to_numpy(dtype=np.float64)
            return adata
    pairs = ", ".join(f"{x}/{y}" for x, y in _SPATIAL_COLUMN_PAIRS)
    raise KeyError(f"{section_id}: need obsm['spatial'] or coordinate columns ({pairs})")


def _prepare_section(adata: ad.AnnData, section_id: str) -> ad.AnnData:
    adata.var_names_make_unique(join="++")
    adata.obs_names = [f"{name}_{section_id}" for name in adata.obs_names.astype(str)]
    adata.obs["batch_name"] = section_id
    adata.obs["slice_name"] = section_id
    return ensure_spatial(adata, section_id)


def load_sections(
    section_ids: Sequence[str],
    data_dir: str,
) -> List[ad.AnnData]:
    if not section_ids:
        section_ids = discover_section_ids(data_dir)
    batch_list: List[ad.AnnData] = []
    for spec in section_ids:
        section_id, path = resolve_section(spec, data_dir)
        print(f"Loading {section_id}: {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{section_id}: expected {section_id}.h5ad or {section_id}_data.h5ad in {data_dir}"
            )
        adata = sc.read_h5ad(path)
        adata = _prepare_section(adata, section_id)
        print(f"  shape: {adata.shape}")
        batch_list.append(adata)
    return batch_list


def concat_sections(
    batch_list: List[ad.AnnData],
    section_ids: Optional[Sequence[str]] = None,
) -> ad.AnnData:
    if section_ids is None:
        section_ids = [str(adata.obs["slice_name"].iloc[0]) for adata in batch_list]
    adata = ad.concat(batch_list, label="slice_name", keys=list(section_ids), join="inner")
    adata.obs["batch_name"] = adata.obs["slice_name"].astype("category")
    print(f"Concatenated shape: {adata.shape}")
    return adata


def load_concat(
    section_ids: Sequence[str],
    data_dir: str,
) -> Tuple[ad.AnnData, List[ad.AnnData]]:
    batch_list = load_sections(section_ids, data_dir)
    return concat_sections(batch_list), batch_list
