#!/usr/bin/env python
"""CellCharter clustering for a generic multi-slice spatial benchmark."""

from __future__ import annotations

import argparse
import os
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import scanpy as sc
import squidpy as sq

try:
    import cellcharter as cc
except ImportError as exc:
    raise ImportError(
        "cellcharter is required. Run this script in an environment with "
        "cellcharter, torchgmm, squidpy, and scanpy installed."
    ) from exc

from load_data import (
    add_data_args,
    add_n_clusters_arg,
    load_concat,
    normalize_total,
    resolve_run_config,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CellCharter multi-slice spatial benchmark")
    add_data_args(p)
    add_n_clusters_arg(p)
    p.add_argument("--min-cells", type=int, default=10)
    p.add_argument("--n-hvgs", type=int, default=300)
    p.add_argument("--n-pcs", type=int, default=10)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=666)
    return p.parse_args()


def preprocess(adata, min_cells: int, n_hvgs: int, n_pcs: int, seed: int):
    sc.pp.filter_genes(adata, min_cells=min_cells)
    normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=min(n_hvgs, adata.n_vars - 1),
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=seed)
    return adata


def main() -> None:
    args = resolve_run_config(parse_args(), "cellcharter")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    adata, _ = load_concat(args.section_ids, args.data_dir)
    adata = preprocess(adata, args.min_cells, args.n_hvgs, args.n_pcs, args.seed)
    print(f"After preprocessing: {adata.shape}")

    sq.gr.spatial_neighbors(
        adata,
        library_key="batch_name",
        coord_type="generic",
        delaunay=True,
    )
    cc.gr.remove_long_links(adata)
    cc.gr.aggregate_neighbors(adata, n_layers=args.n_layers, use_rep="X_pca")

    gmm = cc.tl.Cluster(n_clusters=args.n_clusters, random_state=args.seed)
    gmm.fit(adata, use_rep="X_cellcharter")
    adata.obs["spatial_cluster"] = gmm.predict(adata, use_rep="X_cellcharter")
    adata.obs[f"cellcharter_{args.n_clusters}"] = adata.obs["spatial_cluster"]
    print(f"spatial_cluster: {adata.obs['spatial_cluster'].nunique()}")

    if "_cellcharter" in adata.uns:
        adata.uns["_cellcharter"] = {
            k: v for k, v in adata.uns["_cellcharter"].items() if v is not None
        }

    adata.write_h5ad(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
