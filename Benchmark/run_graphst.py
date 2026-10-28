#!/usr/bin/env python
"""GraphST clustering for a generic multi-slice spatial benchmark."""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("R_HOME", "/usr/lib/R")

_ST_ANALYSIS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_GRAPHST_ROOT = os.path.join(_ST_ANALYSIS, "GraphST")
for _path in reversed((_GRAPHST_ROOT, _ST_ANALYSIS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import scanpy as sc
from GraphST import GraphST
from GraphST.utils import clustering

from load_data import (
    add_data_args,
    add_n_clusters_arg,
    load_concat,
    normalize_total,
    resolve_run_config,
    torch_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GraphST multi-slice spatial benchmark")
    add_data_args(p)
    add_n_clusters_arg(p)
    p.add_argument("--radius", type=int, default=50, help="Spatial refinement radius in coordinate units")
    p.add_argument("--n-hvgs", type=int, default=300)
    p.add_argument("--device", default=None, help="e.g. cuda:0; default cuda:0 if available")
    return p.parse_args()


def preprocess_graphst(adata, n_hvgs: int):
    sc.pp.filter_genes(adata, min_cells=10)
    n_top = min(n_hvgs, max(1, adata.n_vars - 1))
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top)
    adata = adata[:, adata.var["highly_variable"]].copy()
    normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata


def main() -> None:
    args = resolve_run_config(parse_args(), "graphst")
    device = torch_device(args.device)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    adata, _ = load_concat(args.section_ids, args.data_dir)
    adata = preprocess_graphst(adata, args.n_hvgs)
    print(f"After preprocessing: {adata.shape}, device={device}")

    model = GraphST.GraphST(adata, device=device)
    adata = model.train()
    clustering(
        adata,
        n_clusters=args.n_clusters,
        radius=args.radius,
        key="emb",
        method="mclust",
        refinement=True,
    )
    print(f"domain clusters: {adata.obs['domain'].nunique()}")

    for key in list(adata.uns.keys()):
        if key in ("adj", "edgeList", "spatial_net"):
            del adata.uns[key]

    adata.write_h5ad(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

