#!/usr/bin/env python3
"""Convert OpenGWAS VCF summary statistics to gsMap sumstats format.

Output columns are the minimal gsMap/LDSC-style fields:
SNP, A1, A2, Z, N.
"""

from __future__ import annotations

import argparse
import gzip
import math
from pathlib import Path


def open_text(path: Path, mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode)
    return path.open(mode)


def split_number_a(value: str, alt_index: int) -> str:
    if value in {"", "."}:
        return ""
    parts = value.split(",")
    if alt_index < len(parts):
        return parts[alt_index]
    if len(parts) == 1:
        return parts[0]
    return ""


def parse_float(value: str) -> float | None:
    if value in {"", ".", "NA", "nan", "NaN"}:
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return x


def get_sample_value(format_keys: list[str], sample_values: list[str], key: str, alt_index: int) -> str:
    try:
        idx = format_keys.index(key)
    except ValueError:
        return ""
    if idx >= len(sample_values):
        return ""
    return split_number_a(sample_values[idx], alt_index)


def convert_vcf(vcf_path: Path, out_path: Path, default_n: float | None = None) -> tuple[int, int]:
    written = 0
    skipped = 0

    with open_text(vcf_path, "rt") as fin, gzip.open(out_path, "wt") as fout:
        fout.write("SNP\tA1\tA2\tZ\tN\n")
        for line in fin:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                skipped += 1
                continue

            _chrom, _pos, snp, ref, alt_field = fields[:5]
            format_keys = fields[8].split(":")
            sample_values = fields[9].split(":")
            alts = alt_field.split(",")

            for alt_index, alt in enumerate(alts):
                if not snp.startswith("rs"):
                    skipped += 1
                    continue

                z_value = parse_float(get_sample_value(format_keys, sample_values, "EZ", alt_index))
                if z_value is None:
                    beta = parse_float(get_sample_value(format_keys, sample_values, "ES", alt_index))
                    se = parse_float(get_sample_value(format_keys, sample_values, "SE", alt_index))
                    if beta is None or se is None or se == 0:
                        skipped += 1
                        continue
                    z_value = beta / se

                n_value = parse_float(get_sample_value(format_keys, sample_values, "SS", alt_index))
                if n_value is None:
                    n_value = default_n
                if n_value is None or n_value <= 0:
                    skipped += 1
                    continue

                fout.write(f"{snp}\t{alt}\t{ref}\t{z_value:.6g}\t{n_value:.6g}\n")
                written += 1

    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", type=float, default=None)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written, skipped = convert_vcf(args.vcf, args.out, args.n)
    print(f"wrote={written} skipped={skipped} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
