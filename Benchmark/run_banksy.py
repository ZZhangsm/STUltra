#!/usr/bin/env python
"""BANKSY clustering for a generic multi-slice spatial benchmark."""

from __future__ import annotations

import argparse
import os
import random
import warnings
from typing import Dict, List, Sequence

warnings.filterwarnings("ignore")

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from banksy.cluster_methods import run_Leiden_partition
from banksy.embed_banksy import generate_banksy_matrix
from banksy.initialize_banksy import initialize_banksy
from banksy.main import concatenate_all
from banksy_utils import filter_utils
from banksy_utils.refine_clusters import refine_once
from banksy_utils.umap_pca import pca_umap

from load_data import (
    add_data_args,
    load_sections,
    resolve_run_config,
)

COORD_KEYS = ("x_coord", "y_coord", "coord_xy")
NBR_WEIGHT_DECAY = "scaled_gaussian"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BANKSY multi-slice spatial benchmark")
    add_data_args(p)
    p.add_argument("--min-cells", type=int, default=10)
    p.add_argument("--n-hvgs", type=int, default=300)
    p.add_argument("--num-neighbours", type=int, default=15)
    p.add_argument("--lambda-param", type=float, default=0.2, dest="lambda_param")
    p.add_argument("--max-m", type=int, default=0)
    p.add_argument("--resolution", type=float, default=0.6)
    p.add_argument("--pca-dims", type=int, default=20)
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--no-refine", action="store_true")
    return p.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def param_index_name(pca_dim: int, lambda_param: float, resolution: float) -> str:
    return f"{NBR_WEIGHT_DECAY}_pc{pca_dim}_nc{lambda_param:0.2f}_r{resolution:0.2f}"


def load_slice(
    section_id: str,
    data_dir: str,
    min_cells: int,
) -> ad.AnnData:
    adata = load_sections([section_id], data_dir)[0]
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata = filter_utils.normalize_total(adata)
    adata.obs["x_coord"] = adata.obsm["spatial"][:, 0]
    adata.obs["y_coord"] = adata.obsm["spatial"][:, 1]
    adata.obsm["coord_xy"] = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    adata.obs["sample"] = str(adata.obs["slice_name"].iloc[0])
    return adata


def union_hvg_genes(adatas: Dict[str, ad.AnnData], n_hvgs: int) -> List[str]:
    union: set = set()
    for section_id, adata in adatas.items():
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=min(n_hvgs, adata.n_vars - 1),
        )
        hvgs = adata.var_names[adata.var["highly_variable"]].tolist()
        union.update(hvgs)
        print(f"  {section_id}: {len(hvgs)} HVGs")
    genes = sorted(union)
    print(f"Union HVG count: {len(genes)}")
    return genes


def run_banksy_per_sample(
    adata: ad.AnnData,
    section_id: str,
    lambda_list: List[float],
    num_neighbours: int,
    max_m: int,
) -> dict:
    print(f"\n========== BANKSY: {section_id} ==========")
    banksy_dict = initialize_banksy(
        adata,
        COORD_KEYS,
        num_neighbours=num_neighbours,
        nbr_weight_decay=NBR_WEIGHT_DECAY,
        max_m=max_m,
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False,
    )
    banksy_dict, _ = generate_banksy_matrix(
        adata,
        banksy_dict,
        lambda_list,
        max_m=max_m,
        verbose=True,
    )
    banksy_dict["nonspatial"] = {0.0: {"adata": concatenate_all([adata.X], 0, adata=adata)}}
    return banksy_dict


def concatenate_banksy_dicts(banksy_dict: Dict[str, dict], section_ids: Sequence[str]) -> dict:
    banksy_dict_concat: dict = {}
    for sample in section_ids:
        for nbr_wd in banksy_dict[sample]:
            if sample == section_ids[0]:
                banksy_dict_concat[nbr_wd] = {}
            for lambda_param in banksy_dict[sample][nbr_wd]:
                if isinstance(lambda_param, str):
                    continue
                if sample == section_ids[0]:
                    banksy_dict_concat[nbr_wd][lambda_param] = {}
                adata_temp = banksy_dict[sample][nbr_wd][lambda_param]["adata"]
                adata_temp.obs["sample"] = sample
                if sample == section_ids[0]:
                    banksy_dict_concat[nbr_wd][lambda_param]["adata"] = adata_temp
                else:
                    ref = banksy_dict_concat[nbr_wd][lambda_param]["adata"]
                    df_obs = pd.concat([ref.obs, adata_temp.obs])
                    banksy_dict_concat[nbr_wd][lambda_param]["adata"] = ad.concat(
                        [ref, adata_temp],
                        label="sample",
                        join="outer",
                    )
                    banksy_dict_concat[nbr_wd][lambda_param]["adata"].obs = df_obs
    return banksy_dict_concat


def main() -> None:
    args = resolve_run_config(parse_args(), "banksy")
    set_seeds(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    target_params = param_index_name(args.pca_dims, args.lambda_param, args.resolution)
    adata_list: Dict[str, ad.AnnData] = {}
    for section_id in args.section_ids:
        adata_list[section_id] = load_slice(
            section_id,
            args.data_dir,
            args.min_cells,
        )

    hvg_union = union_hvg_genes(adata_list, args.n_hvgs)
    for section_id in args.section_ids:
        present = set(adata_list[section_id].var_names)
        genes = [gene for gene in hvg_union if gene in present]
        adata_list[section_id] = adata_list[section_id][:, genes].copy()
        print(f"  {section_id} after HVG subset: {adata_list[section_id].shape}")

    banksy_dict: Dict[str, dict] = {}
    for section_id in args.section_ids:
        banksy_dict[section_id] = run_banksy_per_sample(
            adata_list[section_id],
            section_id,
            [args.lambda_param],
            args.num_neighbours,
            args.max_m,
        )

    banksy_dict_concat = concatenate_banksy_dicts(banksy_dict, args.section_ids)
    pca_umap(
        banksy_dict_concat,
        pca_dims=[args.pca_dims],
        plt_remaining_var=False,
        add_umap=False,
    )
    results_df, _ = run_Leiden_partition(
        banksy_dict_concat,
        [args.resolution],
        num_nn=50,
        num_iterations=-1,
        partition_seed=1234,
        match_labels=True,
    )
    print(results_df[["decay", "lambda_param", "num_pcs", "resolution", "num_labels"]])

    if target_params not in results_df.index:
        print(f"Warning: {target_params} not in results; using {results_df.index[0]}")
        target_params = results_df.index[0]

    tmp_label = results_df.loc[target_params, "labels"]
    tmp_adata = results_df.loc[target_params, "adata"].copy()
    label_col = f"labels_{target_params}"
    tmp_adata.obs[label_col] = pd.Categorical(tmp_label.dense)
    tmp_adata.obs["banksy_cluster"] = tmp_adata.obs[label_col]

    if not args.no_refine:
        refined_by_obs: Dict[str, int] = {}
        for section_id in args.section_ids:
            adata_sec = tmp_adata[tmp_adata.obs["sample"] == section_id].copy()
            adata_sec.obsm["coord_xy"] = adata_list[section_id].obsm["coord_xy"]
            raw_labels = list(adata_sec.obs[label_col].astype(int))
            refine_label, refine_ari, total_entropy = refine_once(
                adata_sec,
                raw_labels=raw_labels,
                truth_labels=None,
                coord_keys=COORD_KEYS,
                num_neigh=args.num_neighbours,
            )
            print(f"  {section_id} refine (ARI vs truth={refine_ari}), entropy={total_entropy}")
            for name, lab in zip(adata_sec.obs_names, refine_label):
                refined_by_obs[name] = lab
        tmp_adata.obs["banksy_refined"] = pd.Categorical(
            [refined_by_obs.get(name, -1) for name in tmp_adata.obs_names]
        )

    tmp_adata.write_h5ad(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

