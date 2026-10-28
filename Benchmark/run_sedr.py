#!/usr/bin/env python
"""SEDR + Harmony + mclust clustering for a generic multi-slice spatial benchmark."""

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

import harmonypy as hm
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA

import SEDR
from load_data import (
    add_data_args,
    add_n_clusters_arg,
    load_sections,
    normalize_total,
    resolve_run_config,
    section_path,
    torch_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEDR multi-slice spatial benchmark")
    add_data_args(p)
    add_n_clusters_arg(p)
    p.add_argument("--n-neighbors", type=int, default=12)
    p.add_argument("--n-hvgs", type=int, default=300)
    p.add_argument("--pca-components", type=int, default=200)
    p.add_argument("--device", default=None, help="e.g. cuda:0; default cuda:0 if available")
    return p.parse_args()


def build_multislice_graph(
    section_ids,
    data_dir: str,
    n_neighbors: int,
):
    adata = None
    graph_dict = None
    for section_id in section_ids:
        print(f"Processing {section_id}: {section_path(data_dir, section_id)}")
        adata_tmp = load_sections([section_id], data_dir)[0]
        adata_tmp.obs["proj_name"] = section_id
        graph_dict_tmp = SEDR.graph_construction(adata_tmp, n=n_neighbors)
        if adata is None:
            adata = adata_tmp
            graph_dict = graph_dict_tmp
        else:
            var_names = adata.var_names.intersection(adata_tmp.var_names)
            adata = adata[:, var_names]
            adata_tmp = adata_tmp[:, var_names]
            adata = adata.concatenate(adata_tmp, index_unique=None)
            graph_dict = SEDR.combine_graph_dict(graph_dict, graph_dict_tmp)
    adata.obs["batch"] = adata.obs["proj_name"].astype("category")
    return adata, graph_dict


def main() -> None:
    args = resolve_run_config(parse_args(), "sedr")
    device = torch_device(args.device)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    adata, graph_dict = build_multislice_graph(
        args.section_ids,
        args.data_dir,
        args.n_neighbors,
    )
    print(f"Concatenated: {adata.shape}")

    adata.layers["count"] = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    normalize_total(adata, target_sum=1e6)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        layer="count",
        n_top_genes=min(args.n_hvgs, adata.n_vars - 1),
    )
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata)

    n_components = min(args.pca_components, adata.n_vars - 1, adata.n_obs - 1)
    adata.obsm["X_pca"] = PCA(n_components=n_components, random_state=42).fit_transform(adata.X)

    sedr_net = SEDR.Sedr(
        adata.obsm["X_pca"], graph_dict, mode="clustering", device=device
    )
    sedr_net.train_without_dec()
    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm["SEDR"] = sedr_feat

    ho = hm.run_harmony(adata.obsm["SEDR"], adata.obs[["batch"]], vars_use=["batch"])
    res = pd.DataFrame(ho.Z_corr).T
    adata.obsm["SEDR.Harmony"] = pd.DataFrame(
        data=res.values,
        columns=[f"X{i + 1}" for i in range(res.shape[1])],
        index=adata.obs.index,
    )

    SEDR.mclust_R(
        adata,
        n_clusters=args.n_clusters,
        use_rep="SEDR.Harmony",
        key_added="mclust",
        random_seed=666,
    )
    print(f"mclust clusters: {adata.obs['mclust'].nunique()}")

    adata.write_h5ad(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
