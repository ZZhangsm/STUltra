#!/usr/bin/env python3
"""Convert OpenGWAS/GWAS-VCF files to gsMap sumstats format."""

from __future__ import annotations

import argparse
import gzip
import math
from pathlib import Path


VALID_SNPS = {"AC", "AG", "CA", "CT", "GA", "GT", "TC", "TG"}
MISSING = {"", ".", "NA", "nan"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", required=True, type=float)
    return parser.parse_args()


def value_at(format_keys: list[str], values: list[str], key: str) -> str | None:
    try:
        return values[format_keys.index(key)]
    except ValueError:
        return None


def first_number(value: str | None) -> float | None:
    if value is None or value in MISSING:
        return None
    value = value.split(",")[0]
    if value in MISSING:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def convert(vcf_path: Path, out_path: Path, default_n: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0

    with gzip.open(vcf_path, "rt") as src, gzip.open(out_path, "wt") as dst:
        dst.write("SNP\tA1\tA2\tZ\tN\n")
        for line in src:
            if not line or line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                skipped += 1
                continue

            _chrom, _pos, variant_id, ref, alt = fields[:5]
            format_keys = fields[8].split(":")
            sample_values = fields[9].split(":")

            if "," in alt or len(ref) != 1 or len(alt) != 1:
                skipped += 1
                continue

            ref = ref.upper()
            alt = alt.upper()
            if ref + alt not in VALID_SNPS:
                skipped += 1
                continue

            snp = value_at(format_keys, sample_values, "ID") or variant_id
            if snp in MISSING or not snp.startswith("rs"):
                skipped += 1
                continue

            z = first_number(value_at(format_keys, sample_values, "EZ"))
            if z is None:
                beta = first_number(value_at(format_keys, sample_values, "ES"))
                se = first_number(value_at(format_keys, sample_values, "SE"))
                if beta is None or se is None or se == 0:
                    skipped += 1
                    continue
                z = beta / se

            n = first_number(value_at(format_keys, sample_values, "SS")) or default_n
            if n <= 0:
                skipped += 1
                continue

            # GWAS-VCF ES is relative to ALT, so ALT is the effect allele.
            dst.write(f"{snp}\t{alt}\t{ref}\t{z:.6g}\t{n:.3f}\n")
            kept += 1

    print(f"{vcf_path.name}: wrote {kept} variants to {out_path}; skipped {skipped}")


def main() -> None:
    args = parse_args()
    convert(args.vcf, args.out, args.n)


if __name__ == "__main__":
    main()
