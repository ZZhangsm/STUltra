#!/usr/bin/env python
"""Generate a batch-wise gsMap report from existing gsMap outputs.

This is a report-only utility. It reuses gsMap's original plotting/stat helper
functions where possible and does not rerun generate_ldscore, spatial_ldsc, or
latent_to_gene.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

# The project environment installs scanpy in a way that can make numba's import
# cache fail before the script starts. Disable JIT for this report-only script;
# gsMap's original plotting helpers are still used below.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import scanpy as sc
from jinja2 import Template
from scipy.sparse import issparse
from scipy.stats import norm


REPO_ROOT = Path(__file__).resolve().parents[1]
GSMAP_SRC = REPO_ROOT / "gsMap" / "src"
if str(GSMAP_SRC) not in sys.path:
    sys.path.insert(0, str(GSMAP_SRC))

from gsMap.cauchy_combination_test import acat_test  # noqa: E402
from gsMap.utils.manhattan_plot import ManhattanPlot  # noqa: E402
from gsMap.utils.regression_read import _read_chr_files  # noqa: E402
from gsMap.visualize import draw_scatter, estimate_point_size_for_plot, load_ldsc, load_st_coord  # noqa: E402


logger = logging.getLogger("run_multisample_report")


REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
        .plot-container { margin-bottom: 50px; }
        .section-description { color: #6c757d; font-size: 0.95rem; margin-bottom: 20px; }
        .scrollable-table { max-height: 460px; overflow-y: auto; }
        .table thead th { position: sticky; top: 0; background-color: #f8f9fa; }
        .batch-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 18px; }
        .batch-panel { border: 1px solid #dee2e6; border-radius: 6px; padding: 12px; overflow: auto; }
        .batch-title { font-weight: bold; margin-bottom: 10px; }
        .gene-card { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .plot-frame { width: 100%; height: 540px; border: 1px solid #dee2e6; border-radius: 5px; }
        .collapse-toggle { cursor: pointer; color: #0d6efd; text-decoration: underline; }
    </style>
</head>
<body>
<div class="container-fluid">
    <h1 class="mb-4">{{ title }}</h1>

    <div class="plot-container">
        <h2>Genetic Spatial Mapping Plot</h2>
        <p class="section-description">Batch-wise spatial genetic mapping results.</p>
        <div class="batch-grid">
            {% for batch in batches %}
            <div class="batch-panel">
                <div class="batch-title">{{ batch.label }}</div>
                {{ batch.gsmap_html|safe }}
            </div>
            {% endfor %}
        </div>
    </div>

    <div class="plot-container">
        <h2>Cauchy Combination Result</h2>
        <p class="section-description">Cauchy combination results recomputed within each batch.</p>
        <div class="scrollable-table">
            <table class="table table-hover table-bordered table-sm">
                <thead class="table-light">
                    <tr><th>{{ split_key }}</th><th>Annotation</th><th>P Cauchy</th><th>P Median</th></tr>
                </thead>
                <tbody>
                    {% for row in cauchy_table %}
                    <tr>
                        <td>{{ row.batch_name }}</td>
                        <td>{{ row.annotation }}</td>
                        <td>{{ "%.4e"|format(row.p_cauchy) }}</td>
                        <td>{{ "%.4e"|format(row.p_median) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <div class="plot-container">
        <h2>Diagnosis Manhattan Plot</h2>
        <p class="section-description">Batch-wise diagnostic Manhattan plots using each batch's Gene Diagnostic Info.</p>
        <div class="batch-grid">
            {% for batch in batches %}
            <div class="batch-panel">
                <div class="batch-title">{{ batch.label }}</div>
                {{ batch.manhattan_html|safe }}
            </div>
            {% endfor %}
        </div>
    </div>

    <div class="plot-container">
        <h2>Gene Expression and GSS Distribution</h2>
        <p class="section-description">Global top genes shown across batches for direct comparison.</p>
        <label for="geneSelect" class="fw-bold mb-2">Select a gene:</label>
        <select id="geneSelect" class="form-select mb-4">
            {% for gene in gene_plots %}
            <option value="{{ gene.name }}">{{ gene.name }}</option>
            {% endfor %}
        </select>
        <div class="batch-grid">
            {% for batch in batches %}
            <div class="batch-panel">
                <div class="batch-title">{{ batch.label }}</div>
                <div class="gene-card">
                    <div>
                        <h6>Expression Distribution</h6>
                        <iframe class="plot-frame" id="expression-{{ batch.safe_label }}" title="Expression Distribution"></iframe>
                    </div>
                    <div>
                        <h6>Gene Specificity Score (GSS)</h6>
                        <iframe class="plot-frame" id="gss-{{ batch.safe_label }}" title="GSS Distribution"></iframe>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>

    <div class="plot-container">
        <h2>Gene Diagnostic Info</h2>
        <p class="section-description">Gene diagnostic information recomputed within each batch.</p>
        <div class="scrollable-table">
            <table class="table table-hover table-bordered table-sm">
                <thead class="table-light">
                    <tr><th>{{ split_key }}</th><th>Gene</th><th>Annotation</th><th>Median GSS</th><th>PCC</th></tr>
                </thead>
                <tbody>
                    {% for row in gene_diagnostic_info %}
                    <tr>
                        <td>{{ row.batch_name }}</td>
                        <td>{{ row.Gene }}</td>
                        <td>{{ row.Annotation }}</td>
                        <td>{{ "%.4f"|format(row.Median_GSS) }}</td>
                        <td>{{ "%.4f"|format(row.PCC) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <div class="plot-container">
        <h2>Running Info</h2>
        <p class="collapse-toggle" data-bs-toggle="collapse" href="#runningInfo" role="button">Show/Hide Running Info</p>
        <div class="collapse" id="runningInfo">
            <div class="card card-body">
                <ul class="mb-0">
                    {% for key, value in parameters.items() %}
                    <li><strong>{{ key }}:</strong> {{ value }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
const genePlots = {{ gene_plot_json|safe }};
const batches = {{ batch_json|safe }};
const geneSelect = document.getElementById("geneSelect");

function updateGenePlots() {
    const geneName = geneSelect.value;
    for (const batch of batches) {
        const paths = genePlots[geneName][batch.safe_label];
        document.getElementById(`expression-${batch.safe_label}`).src = paths.expression_plot;
        document.getElementById(`gss-${batch.safe_label}`).src = paths.gss_plot;
    }
}

geneSelect.addEventListener("change", updateGenePlots);
updateGenePlots();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a batch-wise gsMap HTML report.")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--sample_name", required=True)
    parser.add_argument("--trait_name", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--sumstats_file", required=True)
    parser.add_argument("--split_key", default="batch_name")
    parser.add_argument("--top_corr_genes", type=int, default=50)
    parser.add_argument("--split_values", nargs="*", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--fig_width", type=int, default=None)
    parser.add_argument("--fig_height", type=int, default=None)
    parser.add_argument("--point_size", type=float, default=None)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "group"


def make_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.workdir) / args.sample_name
    report_dir = root / "report" / args.trait_name
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else report_dir / f"multi_sample_by_{safe_name(args.split_key)}"
    )
    return {
        "root": root,
        "h5ad": root / "find_latent_representations" / f"{args.sample_name}_add_latent.h5ad",
        "mk_score": root / "latent_to_gene" / f"{args.sample_name}_gene_marker_score.feather",
        "ldsc": root / "spatial_ldsc" / f"{args.sample_name}_{args.trait_name}.csv.gz",
        "snp_gene_prefix": root / "generate_ldscore" / "SNP_gene_pair" / "SNP_gene_pair_chr",
        "report_dir": report_dir,
        "overall_gene_info": report_dir / f"{args.sample_name}_{args.trait_name}_Gene_Diagnostic_Info.csv",
        "output_dir": output_dir,
        "gsmap_dir": output_dir / "gsMap_plot",
        "manhattan_dir": output_dir / "manhattan_plot",
        "gss_dir": output_dir / "GSS_plot",
    }


def ensure_inputs(paths: dict[str, Path], sumstats_file: str) -> None:
    for path in [paths["h5ad"], paths["mk_score"], paths["ldsc"], Path(sumstats_file)]:
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")


def load_marker_score(path: Path) -> pd.DataFrame:
    mk_score = pd.read_feather(path).set_index("HUMAN_GENE_SYM").T
    mk_score.index = mk_score.index.astype(str)
    return mk_score


def load_gwas_data(sumstats_file: str) -> pd.DataFrame:
    gwas_data = pd.read_csv(sumstats_file, compression="gzip", sep="\t")
    if "P" not in gwas_data.columns:
        gwas_data["P"] = norm.sf(abs(gwas_data["Z"])) * 2
    gwas_data["P"] = gwas_data["P"].clip(lower=1e-300)
    return gwas_data


def load_snp_gene_pairs(prefix: Path) -> pd.DataFrame:
    files = _read_chr_files(prefix.as_posix(), suffix=".feather")
    if not files:
        raise FileNotFoundError(f"No SNP-gene pair files found with prefix: {prefix}")
    return pd.concat([pd.read_feather(file) for file in files])


def normalize_if_needed(adata):
    if "log1p" not in adata.uns.keys() and adata.X.max() > 14:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return adata


def compute_gene_diagnostic_info(
    obs: pd.DataFrame,
    mk_score: pd.DataFrame,
    trait_ldsc_result: pd.DataFrame,
    annotation: str,
) -> pd.DataFrame:
    common_spots = obs.index.astype(str).intersection(mk_score.index).intersection(trait_ldsc_result.index)
    mk_score = mk_score.loc[common_spots]
    trait_ldsc_result = trait_ldsc_result.loc[common_spots]
    obs = obs.loc[common_spots]

    has_variation = (~mk_score.eq(mk_score.iloc[0], axis=1)).any()
    mk_score = mk_score.loc[:, has_variation]

    corr = mk_score.corrwith(trait_ldsc_result["logp"])
    corr.name = "PCC"

    grouped_mk_score = mk_score.groupby(obs[annotation].astype(str)).median()
    max_annotations = grouped_mk_score.idxmax()
    gene_info = pd.DataFrame(
        {
            "Gene": max_annotations.index,
            "Annotation": max_annotations.values,
            "Median_GSS": grouped_mk_score.max().values,
        }
    ).merge(corr, left_on="Gene", right_index=True)
    gene_info = gene_info[["Gene", "Annotation", "Median_GSS", "PCC"]].dropna(subset=["Gene"])
    gene_info.sort_values("PCC", ascending=False, inplace=True)
    return gene_info.reset_index(drop=True)


def compute_cauchy_table(obs: pd.DataFrame, trait_ldsc_result: pd.DataFrame, annotation: str) -> pd.DataFrame:
    common_spots = obs.index.astype(str).intersection(trait_ldsc_result.index)
    ldsc = trait_ldsc_result.loc[common_spots].copy()
    ldsc["annotation"] = obs.loc[common_spots, annotation].astype(str).to_list()

    rows = []
    for ct in ldsc["annotation"].unique():
        p_values = ldsc.loc[ldsc["annotation"] == ct, "p"].clip(lower=1e-300, upper=1)
        p_values_log = -np.log10(p_values)
        median_log = np.median(p_values_log)
        iqr_log = np.percentile(p_values_log, 75) - np.percentile(p_values_log, 25)
        p_values_filtered = p_values[p_values_log < median_log + 3 * iqr_log]
        n_removed = len(p_values) - len(p_values_filtered)
        test_values = (
            p_values_filtered
            if 0 < n_removed < max(len(p_values) * 0.01, 20)
            else p_values
        )
        rows.append(
            {
                "annotation": ct,
                "p_cauchy": acat_test(test_values),
                "p_median": np.median(p_values),
            }
        )

    result = pd.DataFrame(rows)
    result.sort_values("p_cauchy", inplace=True)
    return result.reset_index(drop=True)


def expression_series(adata, gene: str) -> pd.Series:
    values = adata[:, gene].X
    if issparse(values):
        values = values.toarray()
    values = np.asarray(values).flatten()
    threshold = np.quantile(values, 0.9999)
    values[values > threshold] = threshold
    return pd.Series(values, index=adata.obs.index.astype(str), name="Expression")


def write_plotly(fig, html_path: Path, png_path: Path | None = None) -> str:
    fig.write_html(html_path)
    if png_path is not None:
        try:
            fig.write_image(png_path)
        except Exception as exc:
            logger.warning("Could not write PNG %s: %s", png_path, exc)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def relative(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def apply_color_range(fig, color_min: float, color_max: float):
    if color_max > color_min:
        fig.update_coloraxes(cmin=color_min, cmax=color_max)
    return fig


def generate_manhattan_plot(
    gwas_data: pd.DataFrame,
    snp_gene_pair: pd.DataFrame,
    gene_diagnostic_info: pd.DataFrame,
    highlight_gene_list: list[str],
    title: str,
    output_path: Path,
) -> str:
    gwas_data_with_gene = snp_gene_pair.merge(gwas_data, on="SNP", how="inner").rename(
        columns={"gene_name": "GENE"}
    )
    annotated = gwas_data_with_gene.merge(
        gene_diagnostic_info, left_on="GENE", right_on="Gene", how="left"
    )
    annotated = annotated[~annotated["Annotation"].isna()]
    if annotated.empty:
        raise ValueError(f"No annotated SNPs are available for {title}")

    annotated_sort = annotated.sort_values("P")
    pass_suggestive_line_mask = annotated_sort["P"] < 1e-5
    if pass_suggestive_line_mask.sum() > 100_000:
        snps2plot = annotated_sort.loc[pass_suggestive_line_mask, "SNP"]
    else:
        snps2plot = annotated_sort.head(100_000)["SNP"]

    data = annotated[annotated["SNP"].isin(snps2plot)].reset_index(drop=True)
    data["Annotation_text"] = (
        "PCC: "
        + data["PCC"].round(2).astype(str)
        + "<br>Annotation: "
        + data["Annotation"].astype(str)
    )

    fig = ManhattanPlot(
        dataframe=data,
        title=title,
        point_size=3,
        highlight_gene_list=highlight_gene_list,
        suggestiveline_value=-np.log10(1e-5),
        annotation="Annotation_text",
    )
    fig.write_html(output_path)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s")
    args = parse_args()
    paths = make_paths(args)
    ensure_inputs(paths, args.sumstats_file)
    for dirname in ["output_dir", "gsmap_dir", "manhattan_dir", "gss_dir"]:
        paths[dirname].mkdir(parents=True, exist_ok=True)

    logger.info("Loading existing gsMap outputs...")
    adata = normalize_if_needed(sc.read_h5ad(paths["h5ad"]))
    adata.obs.index = adata.obs.index.astype(str)
    if args.split_key not in adata.obs:
        raise KeyError(f"{args.split_key!r} not found in adata.obs. Available keys: {list(adata.obs.columns)}")
    if args.annotation not in adata.obs:
        raise KeyError(f"{args.annotation!r} not found in adata.obs. Available keys: {list(adata.obs.columns)}")

    mk_score = load_marker_score(paths["mk_score"])
    trait_ldsc_result = load_ldsc(paths["ldsc"])
    gwas_data = load_gwas_data(args.sumstats_file)
    snp_gene_pair = load_snp_gene_pairs(paths["snp_gene_prefix"])

    if paths["overall_gene_info"].exists():
        overall_gene_info = pd.read_csv(paths["overall_gene_info"])
    else:
        logger.info("Overall Gene Diagnostic Info not found. Computing it from all spots...")
        overall_gene_info = compute_gene_diagnostic_info(
            adata.obs, mk_score, trait_ldsc_result, args.annotation
        )
        overall_gene_info.to_csv(paths["overall_gene_info"], index=False)

    global_top_genes = overall_gene_info["Gene"].head(args.top_corr_genes).tolist()
    (paths["gss_dir"] / "plot_genes.csv").write_text("\n".join(global_top_genes))

    split_series = adata.obs[args.split_key].astype(str)
    split_values = args.split_values or list(pd.unique(split_series))
    if args.fig_width and args.fig_height and args.point_size:
        pixel_width, pixel_height, point_size = args.fig_width, args.fig_height, args.point_size
    else:
        (pixel_width, pixel_height), point_size = estimate_point_size_for_plot(adata.obsm["spatial"])
    global_logp_max = trait_ldsc_result["logp"].max()
    global_gss_max = mk_score[global_top_genes].max().max()

    batch_infos = []
    cauchy_frames = []
    gene_info_frames = []
    gene_plot_paths = {gene: {} for gene in global_top_genes}

    for split_value in split_values:
        split_value = str(split_value)
        safe_split = safe_name(split_value)
        logger.info("Generating report diagnostics for %s=%s", args.split_key, split_value)

        batch_mask = split_series == split_value
        batch_spots = adata.obs.index[batch_mask].astype(str)
        common_spots = batch_spots.intersection(trait_ldsc_result.index).intersection(mk_score.index)
        if common_spots.empty:
            logger.warning("Skipping %s=%s because no common spots were found.", args.split_key, split_value)
            continue

        adata_batch = adata[common_spots].copy()
        trait_ldsc_batch = trait_ldsc_result.loc[common_spots]
        mk_score_batch = mk_score.loc[common_spots]

        gene_info = compute_gene_diagnostic_info(
            adata_batch.obs, mk_score_batch, trait_ldsc_batch, args.annotation
        )
        gene_info.insert(0, "batch_name", split_value)
        gene_info_frames.append(gene_info)

        cauchy = compute_cauchy_table(adata_batch.obs, trait_ldsc_batch, args.annotation)
        cauchy.insert(0, "batch_name", split_value)
        cauchy_frames.append(cauchy)

        gsmap_data = load_st_coord(adata_batch, trait_ldsc_batch["logp"].rename("logp"), args.annotation)
        gsmap_fig = draw_scatter(
            gsmap_data,
            title=f"{args.trait_name} (gsMap) - {split_value}",
            point_size=point_size,
            width=pixel_width,
            height=pixel_height,
            annotation=args.annotation,
        )
        apply_color_range(gsmap_fig, 0, global_logp_max)
        gsmap_html_path = paths["gsmap_dir"] / f"{args.sample_name}_{args.trait_name}_{safe_split}_gsMap_plot.html"
        gsmap_html = write_plotly(gsmap_fig, gsmap_html_path, gsmap_html_path.with_suffix(".png"))
        gsmap_data.to_csv(gsmap_html_path.with_suffix(".csv"))

        manhattan_path = (
            paths["manhattan_dir"]
            / f"{args.sample_name}_{args.trait_name}_{safe_split}_Diagnostic_Manhattan_Plot.html"
        )
        manhattan_html = generate_manhattan_plot(
            gwas_data,
            snp_gene_pair,
            gene_info.drop(columns=["batch_name"]),
            gene_info["Gene"].head(args.top_corr_genes).tolist(),
            f"gsMap Diagnosis Manhattan Plot - {split_value}",
            manhattan_path,
        )

        for gene in global_top_genes:
            if gene not in adata_batch.var_names or gene not in mk_score_batch.columns:
                logger.warning("Skipping missing gene %s for %s=%s", gene, args.split_key, split_value)
                continue

            expression = expression_series(adata_batch, gene)
            expression_data = load_st_coord(adata_batch, expression, args.annotation)
            expression_fig = draw_scatter(
                expression_data,
                title=f"{gene} (Expression) - {split_value}",
                annotation="annotation",
                color_by="Expression",
                point_size=point_size,
                width=pixel_width,
                height=pixel_height,
            )
            expression_html = paths["gss_dir"] / f"{args.sample_name}_{safe_split}_{gene}_Expression_Distribution.html"
            write_plotly(expression_fig, expression_html, expression_html.with_suffix(".png"))

            gss_data = load_st_coord(adata_batch, mk_score_batch[gene].rename("GSS"), args.annotation)
            gss_fig = draw_scatter(
                gss_data,
                title=f"{gene} (GSS) - {split_value}",
                annotation="annotation",
                color_by="GSS",
                point_size=point_size,
                width=pixel_width,
                height=pixel_height,
            )
            apply_color_range(gss_fig, 0, global_gss_max)
            gss_html = paths["gss_dir"] / f"{args.sample_name}_{safe_split}_{gene}_GSS_Distribution.html"
            write_plotly(gss_fig, gss_html, gss_html.with_suffix(".png"))

            gene_plot_paths[gene][safe_split] = {
                "expression_plot": relative(expression_html, paths["output_dir"]),
                "gss_plot": relative(gss_html, paths["output_dir"]),
            }

        batch_infos.append(
            {
                "label": split_value,
                "safe_label": safe_split,
                "gsmap_html": gsmap_html,
                "manhattan_html": manhattan_html,
            }
        )

    if not batch_infos:
        raise ValueError("No batch groups were generated.")

    cauchy_all = pd.concat(cauchy_frames, ignore_index=True)
    gene_info_all = pd.concat(gene_info_frames, ignore_index=True)
    cauchy_csv = paths["output_dir"] / f"{args.sample_name}_{args.trait_name}_{safe_name(args.split_key)}_Cauchy.csv"
    gene_info_csv = (
        paths["output_dir"]
        / f"{args.sample_name}_{args.trait_name}_{safe_name(args.split_key)}_Gene_Diagnostic_Info.csv"
    )
    cauchy_all.to_csv(cauchy_csv, index=False)
    gene_info_all.to_csv(gene_info_csv, index=False)

    batch_json = json.dumps(
        [{"label": item["label"], "safe_label": item["safe_label"]} for item in batch_infos]
    )
    gene_plot_json = json.dumps(gene_plot_paths)
    parameters = {
        "Sample Name": args.sample_name,
        "Trait Name": args.trait_name,
        "Summary Statistics File": args.sumstats_file,
        "Annotation": args.annotation,
        "Split Key": args.split_key,
        "Split Values": ", ".join(item["label"] for item in batch_infos),
        "HDF5 Path": paths["h5ad"],
        "Marker Score File": paths["mk_score"],
        "Spatial LDSC File": paths["ldsc"],
        "Output Directory": paths["output_dir"],
        "Top Correlated Genes": args.top_corr_genes,
        "Report Generation Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    report_file = paths["output_dir"] / f"{args.sample_name}_{args.trait_name}_multi_sample_Report.html"
    html = Template(REPORT_TEMPLATE).render(
        title=f"{args.sample_name} {args.trait_name} Multi-sample Genetic Spatial Mapping Report",
        split_key=args.split_key,
        batches=batch_infos,
        cauchy_table=cauchy_all.to_dict(orient="records"),
        gene_diagnostic_info=gene_info_all.to_dict(orient="records"),
        gene_plots=[{"name": gene} for gene in global_top_genes],
        gene_plot_json=gene_plot_json,
        batch_json=batch_json,
        parameters=parameters,
    )
    report_file.write_text(html)
    logger.info("Multi-sample report generated: %s", report_file)


if __name__ == "__main__":
    main()
