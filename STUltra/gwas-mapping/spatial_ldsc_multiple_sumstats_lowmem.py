import gc
import gzip
import logging
import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.ipc as arrow_ipc
import yaml
import zarr
from scipy.stats import norm
from tqdm.contrib.concurrent import thread_map

from gsMap.config import SpatialLDSCConfig
from gsMap.spatial_ldsc_multiple_sumstats import (
    S_LDSC_Boost_with_pre_calculate_SNP_Gene_weight_matrix,
    append_intercept,
    determine_chunk_range,
    determine_total_chunks,
    jackknife_for_processmap,
    load_ldscore_chunk,
)
from gsMap.utils.regression_read import _read_chr_files, _read_ref_ld_v2, _read_w_ld

logger = logging.getLogger("gsMap.spatial_ldsc")
SUMSTATS_CHUNKSIZE = 500_000
LDSCORE_SPOTS_PER_BATCH = int(os.environ.get("GSMAP_LDSC_SPOTS_PER_BATCH", "250"))


def _iter_sumstats_paths(config: SpatialLDSCConfig):
    """Yield trait names and sumstats paths, with a YAML fallback for older configs."""
    if config.sumstats_config_dict:
        yield from config.sumstats_config_dict.items()
        return

    if config.sumstats_config_file is not None:
        with open(config.sumstats_config_file) as f:
            sumstats_config = yaml.load(f, Loader=yaml.FullLoader)
        if not sumstats_config:
            raise ValueError(f"No sumstats found in {config.sumstats_config_file}")
        yield from sumstats_config.items()
        return

    if config.sumstats_file is not None and config.trait_name is not None:
        yield config.trait_name, config.sumstats_file
        return

    raise ValueError("One of sumstats_file and sumstats_config_file must be provided.")


def _validate_sumstats_files(sumstats_items):
    for _trait_name, sumstat_file_path in sumstats_items:
        if not os.path.exists(sumstat_file_path):
            raise FileNotFoundError(f"{sumstat_file_path} not found")


def _sumstats_compression(sumstat_file_path):
    if sumstat_file_path.endswith("gz"):
        return "gzip"
    if sumstat_file_path.endswith("bz2"):
        return "bz2"
    return None


def _iter_sumstats_chunks(sumstat_file_path):
    dtype_dict = {"SNP": str, "Z": float, "N": float}
    try:
        reader = pd.read_csv(
            sumstat_file_path,
            sep=r"\s+",
            na_values=".",
            usecols=["SNP", "Z", "N"],
            dtype=dtype_dict,
            compression=_sumstats_compression(sumstat_file_path),
            chunksize=SUMSTATS_CHUNKSIZE,
        )
        yield from reader
    except (AttributeError, ValueError) as e:
        logger.error(f"Failed to parse sumstats file: {str(e.args)}")
        raise ValueError("Improperly formatted sumstats file: " + str(e.args)) from e


def _resolve_chisq_max(sumstat_file_path, chisq_max):
    if chisq_max is not None:
        return chisq_max

    max_n = None
    for chunk in _iter_sumstats_chunks(sumstat_file_path):
        chunk_max_n = chunk.N.max()
        if pd.isna(chunk_max_n):
            continue
        max_n = chunk_max_n if max_n is None else max(max_n, chunk_max_n)

    if max_n is None:
        raise ValueError(f"No valid N values found in {sumstat_file_path}")

    resolved_chisq_max = max(0.001 * max_n, 80)
    logger.info(f"No chi^2 threshold provided, using {resolved_chisq_max} as default")
    return resolved_chisq_max


def _read_filtered_sumstats(trait_name, sumstat_file_path, chisq_max=None, snp_filter=None):
    """Read sumstats in chunks and keep only SNPs needed downstream."""
    logger.info(f"Reading summary statistics from {sumstat_file_path} ...")
    chisq_max = _resolve_chisq_max(sumstat_file_path, chisq_max)
    snp_filter = None if snp_filter is None else pd.Index(snp_filter)

    total_rows = 0
    duplicate_rows = 0
    selected_before_chisq = 0
    seen_snps = set()
    filtered_chunks = []

    for chunk in _iter_sumstats_chunks(sumstat_file_path):
        total_rows += len(chunk)
        # Drop SNPs already seen in earlier chunks AND duplicates within this chunk.
        # Without the intra-chunk check, repeated rs numbers inside a single read
        # chunk slip through and later inflate sumstats via .loc reindexing.
        duplicate_mask = chunk.SNP.isin(seen_snps) | chunk.SNP.duplicated()
        duplicate_rows += int(duplicate_mask.sum())
        seen_snps.update(chunk.SNP)
        chunk = chunk.loc[~duplicate_mask]

        if snp_filter is not None:
            chunk = chunk[chunk.SNP.isin(snp_filter)]
        if chunk.empty:
            continue

        chunk.set_index("SNP", inplace=True)
        chunk = chunk.astype(np.float32)
        chunk["chisq"] = chunk.Z**2
        selected_before_chisq += len(chunk)
        chunk = chunk[chunk.chisq < chisq_max]
        if not chunk.empty:
            filtered_chunks.append(chunk)

    logger.info(f"Read summary statistics for {total_rows} SNPs.")
    if duplicate_rows > 0:
        logger.info(f"Dropped {duplicate_rows} SNPs with duplicated rs numbers.")

    if filtered_chunks:
        sumstats = pd.concat(filtered_chunks, axis=0)
    else:
        sumstats = pd.DataFrame(columns=["Z", "N", "chisq"], dtype=np.float32)
        sumstats.index.name = "SNP"

    removed_rows = selected_before_chisq - len(sumstats)
    if removed_rows > 0:
        logger.info(
            f"Removed {removed_rows} selected SNPs with chi^2 > {chisq_max} ({len(sumstats)} selected SNPs remain)"
        )
    else:
        logger.info(
            f"No selected SNPs removed with chi^2 > {chisq_max} ({len(sumstats)} selected SNPs remain)"
        )

    if len(sumstats) < 200000:
        logger.warning(
            f"WARNING: number of SNPs less than 200k; for {trait_name} this is almost always bad."
        )
    return sumstats


def _collect_common_snp_lowmem(
    sumstats_items, baseline_and_w_ld_common_snp: pd.Index, chisq_max=None
):
    """Collect the common SNP set without keeping all sumstats in memory."""
    logger.info("Validating sumstats files...")
    _validate_sumstats_files(sumstats_items)

    common_snp_among_all_sumstats = None
    for trait_name, sumstat_file_path in sumstats_items:
        sumstats = _read_filtered_sumstats(
            trait_name,
            sumstat_file_path,
            chisq_max,
            snp_filter=baseline_and_w_ld_common_snp,
        )
        common_snp = baseline_and_w_ld_common_snp.intersection(sumstats.index)
        if len(common_snp) < 200000:
            logger.warning(
                f"WARNING: number of SNPs less than 200k; for {trait_name} this is almost always bad."
            )

        if common_snp_among_all_sumstats is None:
            common_snp_among_all_sumstats = common_snp
        else:
            common_snp_among_all_sumstats = common_snp_among_all_sumstats.intersection(
                common_snp
            )

        del sumstats
        gc.collect()

    if common_snp_among_all_sumstats is None:
        raise ValueError("No summary statistics files were provided.")

    logger.info(f"Common SNPs among all sumstats: {len(common_snp_among_all_sumstats)}")
    return common_snp_among_all_sumstats


def _load_one_sumstats_for_common_snp(
    trait_name,
    sumstat_file_path,
    common_snp_among_all_sumstats: pd.Index,
    baseline_and_w_ld_common_snp: pd.Index,
    chisq_max=None,
):
    """Load and align one trait's sumstats to the precomputed common SNP set."""
    sumstats = _read_filtered_sumstats(
        trait_name,
        sumstat_file_path,
        chisq_max,
        snp_filter=common_snp_among_all_sumstats,
    )
    sumstats = sumstats.loc[common_snp_among_all_sumstats]
    sumstats["common_index_pos"] = pd.Index(baseline_and_w_ld_common_snp).get_indexer(
        sumstats.index
    )
    return sumstats


def _trait_output_file(config, trait_name, running_chunk_number, total_chunk_number, start_chunk, end_chunk):
    out_dir = config.ldsc_save_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_name = config.sample_name
    if running_chunk_number == total_chunk_number:
        return out_dir / f"{sample_name}_{trait_name}.csv.gz"
    return out_dir / f"{sample_name}_{trait_name}_chunk{start_chunk}-{end_chunk}.csv.gz"


def _format_out_chunk(out_chunk, spatial_annotation_cnames, chunk_index, trait_name):
    out_chunk = pd.DataFrame.from_records(
        out_chunk, columns=["beta", "se"], index=spatial_annotation_cnames
    )
    nan_spots = out_chunk[out_chunk.isna().any(axis=1)].index
    if len(nan_spots) > 0:
        logger.info(
            f"Nan spots: {nan_spots} in chunk-{chunk_index} for {trait_name}. They are removed."
        )
    out_chunk = out_chunk.dropna()
    out_chunk["z"] = out_chunk.beta / out_chunk.se
    out_chunk["p"] = norm.sf(out_chunk["z"])
    out_chunk["spot"] = out_chunk.index
    out_chunk = out_chunk[["spot", "beta", "se", "z", "p"]]
    out_chunk["p"] = out_chunk["p"].clip(1e-300, 1)
    return out_chunk


def _feather_ldscore_prefix(config, chunk_index):
    sample_name = config.sample_name
    return f"{config.ldscore_save_dir}/{sample_name}_chunk{chunk_index}/{sample_name}."


def _get_feather_spot_names(config, chunk_index):
    ld_file_spatial = _feather_ldscore_prefix(config, chunk_index)
    first_chr_file = _read_chr_files(ld_file_spatial, ".l2.ldscore")[0]
    with arrow_ipc.open_file(first_chr_file) as feather_file:
        return feather_file.schema.names[1:]


def _iter_spot_batches(spots):
    for batch_start in range(0, len(spots), LDSCORE_SPOTS_PER_BATCH):
        yield spots[batch_start : batch_start + LDSCORE_SPOTS_PER_BATCH]


def _load_feather_ldscore_spot_batch(chunk_index, spot_batch, common_snp_index, config):
    ld_file_spatial = _feather_ldscore_prefix(config, chunk_index)
    chr_files = _read_chr_files(ld_file_spatial, ".l2.ldscore")
    columns = ["SNP", *spot_batch]
    ref_ld_chunks = []

    for file_path in chr_files:
        ref_ld_chr = pd.read_feather(file_path, columns=columns)
        ref_ld_chr.set_index("SNP", inplace=True)
        ref_ld_chunks.append(ref_ld_chr)

    if not ref_ld_chunks:
        raise FileNotFoundError(f"No LD score files found matching pattern: {ld_file_spatial}*")

    ref_ld_spatial = pd.concat(ref_ld_chunks, axis=0)
    ref_ld_spatial = ref_ld_spatial.loc[common_snp_index]
    ref_ld_spatial = ref_ld_spatial.astype(np.float32, copy=False)
    return ref_ld_spatial.values, ref_ld_spatial.columns


def _iter_ldscore_spatial_batches(
    chunk_index,
    common_snp_among_all_sumstats_pos,
    common_snp_index,
    config,
    zarr_file=None,
    spots_name=None,
    s_ldsc=None,
):
    """Yield spatial LD score matrices, splitting large feather chunks by spot columns."""
    if config.ldscore_save_format == "feather":
        spot_names = _get_feather_spot_names(config, chunk_index)
        for spot_batch in _iter_spot_batches(spot_names):
            yield _load_feather_ldscore_spot_batch(
                chunk_index, spot_batch, common_snp_index, config
            )
        return

    yield load_ldscore_chunk(
        chunk_index,
        common_snp_among_all_sumstats_pos,
        config,
        zarr_file,
        spots_name,
        s_ldsc,
    )


def _write_trait_results(
    trait_name,
    sumstats,
    ref_ld_baseline,
    ref_ld_baseline_column_sum,
    w_ld,
    common_snp_among_all_sumstats_pos,
    common_snp_index,
    config,
    zarr_file,
    spots_name,
    s_ldsc,
    start_chunk,
    end_chunk,
    running_chunk_number,
    total_chunk_number,
):
    """Run all requested chunks for one trait and stream results to disk."""
    out_file_name = _trait_output_file(
        config, trait_name, running_chunk_number, total_chunk_number, start_chunk, end_chunk
    )
    tmp_file_name = out_file_name.with_name(f"{out_file_name.name}.tmp")
    if tmp_file_name.exists():
        tmp_file_name.unlink()

    baseline_annotation = ref_ld_baseline.copy().astype(np.float32, copy=False)
    w_ld_common_snp = w_ld.astype(np.float32, copy=False)
    baseline_annotation = (
        baseline_annotation * sumstats.N.values.reshape((-1, 1)) / sumstats.N.mean()
    )
    baseline_annotation = append_intercept(baseline_annotation)
    nbar = sumstats.N.mean()
    wrote_header = False

    with gzip.open(tmp_file_name, "wt") as out_fh:
        for chunk_index in range(start_chunk, end_chunk + 1):
            spatial_batch_iter = _iter_ldscore_spatial_batches(
                chunk_index,
                common_snp_among_all_sumstats_pos,
                common_snp_index,
                config,
                zarr_file,
                spots_name,
                s_ldsc,
            )
            for ref_ld_spatial, spatial_annotation_cnames in spatial_batch_iter:
                spatial_annotation = ref_ld_spatial.astype(np.float32, copy=False)
                chunk_size = spatial_annotation.shape[1]

                jackknife_func = partial(
                    jackknife_for_processmap,
                    spatial_annotation=spatial_annotation,
                    ref_ld_baseline_column_sum=ref_ld_baseline_column_sum,
                    sumstats=sumstats,
                    baseline_annotation=baseline_annotation,
                    w_ld_common_snp=w_ld_common_snp,
                    Nbar=nbar,
                    n_blocks=config.n_blocks,
                )

                out_chunk = thread_map(
                    jackknife_func,
                    range(chunk_size),
                    max_workers=config.num_processes,
                    chunksize=10,
                    desc=f"Chunk-{chunk_index}/Total-chunk-{running_chunk_number} for {trait_name}",
                )
                out_chunk = _format_out_chunk(
                    out_chunk, spatial_annotation_cnames, chunk_index, trait_name
                )
                out_chunk.to_csv(out_fh, index=False, header=not wrote_header)
                wrote_header = True

                del ref_ld_spatial, spatial_annotation, out_chunk
                gc.collect()

        if not wrote_header:
            out_fh.write("spot,beta,se,z,p\n")

    os.replace(tmp_file_name, out_file_name)
    logger.info(f"Output saved to {out_file_name} for {trait_name}")

    del baseline_annotation, w_ld_common_snp
    gc.collect()


def run_spatial_ldsc_lowmem(config: SpatialLDSCConfig):
    """Run spatial LDSC with one trait/chunk in memory at a time."""
    logger.info(f"------Running low-memory Spatial LDSC for {config.sample_name}...")
    sample_name = config.sample_name
    sumstats_items = list(_iter_sumstats_paths(config))

    w_ld = _read_w_ld(config.w_file)
    w_ld.set_index("SNP", inplace=True)

    ld_file_baseline = f"{config.ldscore_save_dir}/baseline/baseline."
    ref_ld_baseline = _read_ref_ld_v2(ld_file_baseline)
    baseline_and_w_ld_common_snp = ref_ld_baseline.index.intersection(w_ld.index)

    common_snp_among_all_sumstats = _collect_common_snp_lowmem(
        sumstats_items,
        baseline_and_w_ld_common_snp,
        chisq_max=config.chisq_max,
    )
    common_snp_among_all_sumstats_pos = ref_ld_baseline.index.get_indexer(
        common_snp_among_all_sumstats
    )

    if not pd.Series(common_snp_among_all_sumstats_pos).is_monotonic_increasing:
        raise ValueError("common_snp_among_all_sumstats_pos is not monotonic increasing")

    if len(common_snp_among_all_sumstats) < 200000:
        logger.warning(
            f"!!!!! WARNING: number of SNPs less than 200k; for {sample_name} this is almost always bad. Please check the sumstats files."
        )

    ref_ld_baseline = ref_ld_baseline.loc[common_snp_among_all_sumstats]
    w_ld = w_ld.loc[common_snp_among_all_sumstats]

    if config.use_additional_baseline_annotation:
        logger.info("Using additional baseline annotations")
        ld_file_baseline_additional = f"{config.ldscore_save_dir}/additional_baseline/baseline."
        ref_ld_baseline_additional = _read_ref_ld_v2(ld_file_baseline_additional)
        ref_ld_baseline_additional = ref_ld_baseline_additional.loc[
            common_snp_among_all_sumstats
        ]
        ref_ld_baseline = pd.concat([ref_ld_baseline, ref_ld_baseline_additional], axis=1)
        del ref_ld_baseline_additional
        gc.collect()

    s_ldsc = None
    if config.ldscore_save_format == "quick_mode":
        s_ldsc = S_LDSC_Boost_with_pre_calculate_SNP_Gene_weight_matrix(
            config, common_snp_among_all_sumstats_pos
        )
        total_chunk_number_found = len(s_ldsc.chunk_starts)
        logger.info(f"Split data into {total_chunk_number_found} chunks")
    else:
        total_chunk_number_found = determine_total_chunks(config)

    start_chunk, end_chunk = determine_chunk_range(config, total_chunk_number_found)
    running_chunk_number = end_chunk - start_chunk + 1

    zarr_file, spots_name = None, None
    if config.ldscore_save_format == "zarr":
        zarr_path = Path(config.ldscore_save_dir) / f"{config.sample_name}.ldscore.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(f"{zarr_path} not found, which is required for zarr format")
        zarr_file = zarr.open(str(zarr_path))
        spots_name = zarr_file.attrs["spot_names"]

    ref_ld_baseline_column_sum = ref_ld_baseline.sum(axis=1).values

    for trait_name, sumstat_file_path in sumstats_items:
        logger.info(f"Running low-memory Spatial LDSC for trait {trait_name}...")
        sumstats = _load_one_sumstats_for_common_snp(
            trait_name,
            sumstat_file_path,
            common_snp_among_all_sumstats,
            baseline_and_w_ld_common_snp,
            chisq_max=config.chisq_max,
        )
        _write_trait_results(
            trait_name=trait_name,
            sumstats=sumstats,
            ref_ld_baseline=ref_ld_baseline,
            ref_ld_baseline_column_sum=ref_ld_baseline_column_sum,
            w_ld=w_ld,
            common_snp_among_all_sumstats_pos=common_snp_among_all_sumstats_pos,
            common_snp_index=common_snp_among_all_sumstats,
            config=config,
            zarr_file=zarr_file,
            spots_name=spots_name,
            s_ldsc=s_ldsc,
            start_chunk=start_chunk,
            end_chunk=end_chunk,
            running_chunk_number=running_chunk_number,
            total_chunk_number=total_chunk_number_found,
        )
        del sumstats
        gc.collect()

    logger.info(f"------Low-memory Spatial LDSC for {sample_name} finished!")
