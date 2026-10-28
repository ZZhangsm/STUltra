#!/usr/bin/env python
"""STAligner + mclust clustering for a generic multi-slice spatial benchmark."""

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
if _ST_ANALYSIS not in sys.path:
    sys.path.insert(0, _ST_ANALYSIS)

import anndata as ad
import scanpy as sc
import scipy.sparse as sp

import STAligner
from load_data import (
    add_data_args,
    add_n_clusters_arg,
    load_sections,
    normalize_total,
    resolve_run_config,
    torch_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STAligner multi-slice spatial benchmark")
    add_data_args(p)
    add_n_clusters_arg(p)
    p.add_argument("--n-hvgs", type=int, default=300)
    p.add_argument("--rad-cutoff", type=int, default=50, help="Spatial network radius in coordinate units")
    p.add_argument("--knn-neigh", type=int, default=50)
    p.add_argument("--n-epochs", type=int, default=1000)
    p.add_argument("--device", default=None, help="e.g. cuda:0; default cuda:0 if available")
    p.add_argument("--seed", type=int, default=666)
    return p.parse_args()


def main() -> None:
    args = resolve_run_config(parse_args(), "staligner")
    device = torch_device(args.device)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    batch_list = []
    adj_list = []
    for section_id in args.section_ids:
        print(f"Loading slice: {section_id}")
        adata_tmp = load_sections([section_id], args.data_dir)[0]
        STAligner.Cal_Spatial_Net(adata_tmp, rad_cutoff=args.rad_cutoff)

        sc.pp.highly_variable_genes(
            adata_tmp,
            flavor="seurat_v3",
            n_top_genes=min(args.n_hvgs, adata_tmp.n_vars - 1),
        )
        normalize_total(adata_tmp, target_sum=1e4)
        sc.pp.log1p(adata_tmp)
        adata_tmp = adata_tmp[:, adata_tmp.var["highly_variable"]].copy()

        adj_list.append(adata_tmp.uns["adj"])
        batch_list.append(adata_tmp)

    section_keys = [str(item.obs["slice_name"].iloc[0]) for item in batch_list]
    adata_concat = ad.concat(
        batch_list,
        label="slice_name",
        keys=section_keys,
        join="inner",
        index_unique=None,
    )
    adata_concat.obs["batch_name"] = adata_concat.obs["slice_name"].astype("category")
    print(f"Concatenated: {adata_concat.shape}, device={device}")

    adj_concat = sp.block_diag(adj_list, format="coo")
    adata_concat.uns["edgeList"] = (adj_concat.row, adj_concat.col)

    print("Training STAligner ...")
    adata_concat = STAligner.train_STAligner(
        adata_concat,
        verbose=True,
        knn_neigh=args.knn_neigh,
        n_epochs=args.n_epochs,
        device=device,
        random_seed=args.seed,
    )

    STAligner.ST_utils.mclust_R(
        adata_concat,
        num_cluster=args.n_clusters,
        used_obsm="STAligner",
        random_seed=args.seed,
    )
    print(f"mclust clusters: {adata_concat.obs['mclust'].nunique()}")

    for key in list(adata_concat.uns.keys()):
        if key in ("adj", "edgeList", "Spatial_Net"):
            del adata_concat.uns[key]

    adata_concat.write_h5ad(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

