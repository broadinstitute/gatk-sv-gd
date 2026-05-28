"""
Output writers for GD CNV inference results.

Functions for writing posterior tables, locus metadata, and ploidy estimates
to disk after model inference.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from gatk_sv_gd._util import get_sample_columns
from gatk_sv_gd.bins import LocusBinMapping
from gatk_sv_gd.models import GDLocus


def write_posterior_tables(
    combined_data,
    map_estimates: dict,
    cn_posterior: dict,
    mappings: List[LocusBinMapping],
    output_dir: str,
):
    """
    Write comprehensive posterior tables to disk.

    Args:
        combined_data: DepthData object with all bins
        map_estimates: Dictionary with MAP estimates from model
        cn_posterior: Dictionary with CN posterior probabilities
        mappings: List of LocusBinMapping objects
        output_dir: Output directory for tables
    """
    print("\n" + "=" * 80)
    print("WRITING POSTERIOR TABLES")
    print("=" * 80)

    # 1. Copy state probabilities for all bins and samples
    print("\nWriting copy state posteriors...")

    def _normalize_state_tensor(array, n_bins: int, n_samples: int):
        """Return state posteriors with shape (n_bins, n_samples, n_states)."""
        arr = np.asarray(array)
        if arr.ndim > 3:
            arr = np.squeeze(arr)
        if arr.ndim == 1:
            if n_bins == 1 and n_samples == 1:
                arr = arr.reshape(1, 1, arr.shape[0])
            else:
                raise ValueError(f"Expected 3D state tensor, got shape {arr.shape}")
        if arr.ndim == 2:
            arr = arr.reshape(arr.shape[0], arr.shape[1], 1)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D state tensor, got shape {arr.shape}")

        if arr.shape[0] == n_bins and arr.shape[1] == n_samples:
            return arr
        if arr.shape[0] == n_samples and arr.shape[1] == n_bins:
            return np.transpose(arr, (1, 0, 2))
        if arr.shape[1] == n_bins and arr.shape[2] == n_samples:
            return np.transpose(arr, (1, 2, 0))

        raise ValueError(
            "State tensor shape does not match bins/samples: "
            f"shape={arr.shape}, n_bins={n_bins}, n_samples={n_samples}"
        )

    def _normalize_bin_sample_matrix(array, n_bins: int, n_samples: int):
        """Return a per-bin, per-sample matrix with shape (n_bins, n_samples)."""
        arr = np.asarray(array).squeeze()
        if arr.ndim == 0:
            return np.full((n_bins, n_samples), float(arr), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D bin/sample matrix, got shape {arr.shape}")
        if arr.shape == (n_bins, n_samples):
            return arr
        if arr.shape == (n_samples, n_bins):
            return np.transpose(arr, (1, 0))
        raise ValueError(
            "Bin/sample matrix shape does not match bins/samples: "
            f"shape={arr.shape}, n_bins={n_bins}, n_samples={n_samples}"
        )

    def _normalize_bin_state_matrix(array, n_bins: int):
        """Return a per-bin, per-state matrix with shape (n_bins, n_states)."""
        arr = np.asarray(array)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim == 1:
            if n_bins == 1:
                return arr.reshape(1, -1)
            if arr.shape[0] == n_bins:
                return arr.reshape(n_bins, 1)
            raise ValueError(
                "Bin/state matrix shape does not match bins: "
                f"shape={arr.shape}, n_bins={n_bins}"
            )
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D bin/state matrix, got shape {arr.shape}")
        if arr.shape[0] == n_bins:
            return arr
        if arr.shape[1] == n_bins:
            return np.transpose(arr, (1, 0))
        raise ValueError(
            "Bin/state matrix shape does not match bins: "
            f"shape={arr.shape}, n_bins={n_bins}"
        )

    cn_post = _normalize_state_tensor(
        cn_posterior["cn_posterior"],
        combined_data.n_bins,
        combined_data.n_samples,
    )
    pair_post = cn_posterior.get("pair_state_posterior")
    pair_state_labels = cn_posterior.get("pair_state_labels")
    if pair_post is not None:
        pair_post = _normalize_state_tensor(
            pair_post,
            combined_data.n_bins,
            combined_data.n_samples,
        )
    null_post = cn_posterior.get("null_posterior")
    if null_post is None:
        null_post = np.zeros((combined_data.n_bins, combined_data.n_samples), dtype=np.float32)
    else:
        null_post = _normalize_bin_sample_matrix(
            null_post,
            combined_data.n_bins,
            combined_data.n_samples,
        )
    cn_map = _normalize_bin_sample_matrix(
        map_estimates["cn"],
        combined_data.n_bins,
        combined_data.n_samples,
    )
    pair_map = map_estimates.get("pair_state")
    if pair_map is not None:
        pair_map = _normalize_bin_sample_matrix(
            pair_map,
            combined_data.n_bins,
            combined_data.n_samples,
        )
    depth = np.asarray(combined_data.depth.cpu().numpy())  # shape: (n_bins, n_samples)
    baf_median = None
    minor_baf = None
    baf_var = None
    baf_n_sites = None
    if getattr(combined_data, "has_baf", False):
        baf_median = np.asarray(combined_data.baf_median.cpu().numpy())
        minor_baf = np.asarray(combined_data.minor_baf_median.cpu().numpy())
        baf_var = np.asarray(combined_data.baf_variance.cpu().numpy())
        baf_n_sites = np.asarray(combined_data.baf_n_sites.cpu().numpy())

    # Ensure proper dimensions
    if depth.ndim == 1:
        depth = depth.reshape(-1, 1)

    cn_rows = []
    for bin_idx in range(combined_data.n_bins):
        mapping = mappings[bin_idx]

        for sample_idx, sample_id in enumerate(combined_data.sample_ids):
            row = {
                "cluster": mapping.cluster,
                "interval": mapping.interval_name,
                "chr": mapping.chrom,
                "start": mapping.start,
                "end": mapping.end,
                "sample": sample_id,
                "depth": depth[bin_idx, sample_idx].tolist() if isinstance(depth[bin_idx, sample_idx], np.ndarray) else float(depth[bin_idx, sample_idx]),
                "prob_null": float(null_post[bin_idx, sample_idx]),
            }

            # Add probability for each CN state
            for cn_state in range(cn_post.shape[2]):
                prob_val = cn_post[bin_idx, sample_idx, cn_state]
                row[f"prob_cn_{cn_state}"] = prob_val.tolist() if isinstance(prob_val, np.ndarray) else float(prob_val)

            # Add MAP estimate
            map_val = cn_map[bin_idx, sample_idx]
            row["cn_map"] = int(map_val.tolist() if isinstance(map_val, np.ndarray) else map_val)

            if pair_map is not None and pair_state_labels is not None:
                pair_idx = int(pair_map[bin_idx, sample_idx])
                h1, h2 = pair_state_labels[pair_idx]
                row["pair_state_map"] = pair_idx
                row["pair_h1_map"] = h1
                row["pair_h2_map"] = h2

            if minor_baf is not None:
                row["baf_median"] = float(baf_median[bin_idx, sample_idx])
                row["minor_baf_median"] = float(minor_baf[bin_idx, sample_idx])
                row["baf_variance"] = float(baf_var[bin_idx, sample_idx])
                row["baf_n_sites"] = int(baf_n_sites[bin_idx, sample_idx])

            # Optional pair-state marginals (for future diploid model use)
            if pair_post is not None and pair_state_labels is not None:
                for pair_idx, pair_label in enumerate(pair_state_labels):
                    prob_val = pair_post[bin_idx, sample_idx, pair_idx]
                    row[f"prob_pair_{pair_label[0]}_{pair_label[1]}"] = (
                        prob_val.tolist() if isinstance(prob_val, np.ndarray) else float(prob_val)
                    )

            cn_rows.append(row)

    cn_df = pd.DataFrame(cn_rows)
    cn_output = os.path.join(output_dir, "cn_posteriors.tsv.gz")
    cn_df.to_csv(cn_output, sep="\t", index=False, compression="gzip")
    print("  Saved copy-state posterior table")
    print(f"  Rows: {len(cn_df):,} ({combined_data.n_bins:,} bins × {combined_data.n_samples} samples)")

    # 2. Sample-specific variable posteriors
    print("\nWriting sample-specific variable posteriors...")
    sample_rows = []

    # Convert to numpy array and squeeze extra dimensions
    sample_var = np.asarray(map_estimates["sample_var"]).squeeze()
    baf_temperature = map_estimates.get("baf_temperature")
    if baf_temperature is not None:
        baf_temperature = np.asarray(baf_temperature, dtype=np.float64).squeeze()
    length_scale_var = map_estimates.get("length_scale_var")
    if length_scale_var is not None:
        length_scale_var = np.asarray(length_scale_var, dtype=np.float64).squeeze()

    # Ensure it's at least 1D
    if sample_var.ndim == 0:
        sample_var = sample_var.reshape(1)
    if baf_temperature is not None and baf_temperature.ndim == 0:
        baf_temperature = np.full(combined_data.n_samples, float(baf_temperature), dtype=np.float64)
    elif baf_temperature is not None and baf_temperature.size == 1:
        baf_temperature = np.full(combined_data.n_samples, float(baf_temperature.reshape(-1)[0]), dtype=np.float64)
    if length_scale_var is not None and length_scale_var.ndim == 0:
        length_scale_var = np.full(combined_data.n_samples, float(length_scale_var), dtype=np.float64)
    elif length_scale_var is not None and length_scale_var.size == 1:
        length_scale_var = np.full(combined_data.n_samples, float(length_scale_var.reshape(-1)[0]), dtype=np.float64)

    for sample_idx, sample_id in enumerate(combined_data.sample_ids):
        var_val = sample_var[sample_idx]
        row = {
            "sample": sample_id,
            "sample_var_map": var_val.tolist() if isinstance(var_val, np.ndarray) else float(var_val),
        }
        if baf_temperature is not None:
            temp_val = baf_temperature[sample_idx]
            temp_val = temp_val.tolist() if isinstance(temp_val, np.ndarray) else float(temp_val)
            row["baf_temperature_map"] = temp_val
            row["baf_variance_scale_map"] = temp_val
        if length_scale_var is not None:
            ls_val = length_scale_var[sample_idx]
            row["length_scale_var_map"] = (
                ls_val.tolist() if isinstance(ls_val, np.ndarray) else float(ls_val)
            )
        sample_rows.append(row)

    sample_df = pd.DataFrame(sample_rows)
    sample_output = os.path.join(output_dir, "sample_posteriors.tsv.gz")
    sample_df.to_csv(sample_output, sep="\t", index=False, compression="gzip")
    print("  Saved sample-level posterior table")
    print(f"  Rows: {len(sample_df):,} ({combined_data.n_samples} samples)")

    # 3. Bin-specific variable posteriors
    print("\nWriting bin-specific variable posteriors...")
    bin_rows = []

    # Convert to numpy arrays and ensure proper shape
    bin_bias = np.asarray(map_estimates["bin_bias"]).squeeze()
    bin_var = np.asarray(map_estimates["bin_var"]).squeeze()
    cn_probs = _normalize_bin_state_matrix(map_estimates["cn_probs"], combined_data.n_bins)
    pair_state_priors = map_estimates.get("effective_pair_state_probs")
    if pair_state_priors is None:
        pair_state_priors = map_estimates.get("pair_state_probs")
    null_state_prior = map_estimates.get("null_state_prior")
    pair_state_labels = cn_posterior.get("pair_state_labels")
    if pair_state_priors is not None:
        pair_state_priors = _normalize_bin_state_matrix(pair_state_priors, combined_data.n_bins)
    if null_state_prior is None:
        null_state_prior = np.zeros(combined_data.n_bins, dtype=np.float32)
    else:
        null_state_prior = np.asarray(null_state_prior, dtype=np.float32).squeeze()

    # Ensure we have the right number of dimensions
    if bin_bias.ndim == 0:
        bin_bias = bin_bias.reshape(1)
    if bin_var.ndim == 0:
        bin_var = bin_var.reshape(1)
    if null_state_prior.ndim == 0:
        null_state_prior = np.full(combined_data.n_bins, float(null_state_prior), dtype=np.float32)

    for bin_idx in range(combined_data.n_bins):
        mapping = mappings[bin_idx]

        row = {
            "cluster": mapping.cluster,
            "interval": mapping.interval_name,
            "chr": mapping.chrom,
            "start": mapping.start,
            "end": mapping.end,
            "bin_bias_map": bin_bias[bin_idx].tolist() if isinstance(bin_bias[bin_idx], np.ndarray) else float(bin_bias[bin_idx]),
            "bin_var_map": bin_var[bin_idx].tolist() if isinstance(bin_var[bin_idx], np.ndarray) else float(bin_var[bin_idx]),
            "null_prior": float(null_state_prior[bin_idx]),
        }

        # Add CN probability priors (per-bin learned from data)
        for cn_state in range(cn_probs.shape[1]):
            prob_val = cn_probs[bin_idx, cn_state]
            row[f"cn_prior_{cn_state}"] = prob_val.tolist() if isinstance(prob_val, np.ndarray) else float(prob_val)

        if pair_state_priors is not None and pair_state_labels is not None:
            for pair_idx, pair_label in enumerate(pair_state_labels):
                prob_val = pair_state_priors[bin_idx, pair_idx]
                row[f"pair_prior_{pair_label[0]}_{pair_label[1]}"] = (
                    prob_val.tolist() if isinstance(prob_val, np.ndarray) else float(prob_val)
                )

        bin_rows.append(row)

    bin_df = pd.DataFrame(bin_rows)
    bin_output = os.path.join(output_dir, "bin_posteriors.tsv.gz")
    bin_df.to_csv(bin_output, sep="\t", index=False, compression="gzip")
    print("  Saved bin-level posterior table")
    print(f"  Rows: {len(bin_df):,} ({combined_data.n_bins:,} bins)")

    print("\n" + "=" * 80)


def write_locus_metadata(
    included_loci: Dict[str, GDLocus],
    mappings: List[LocusBinMapping],
    output_dir: str,
):
    """
    Write locus metadata and bin mappings for use by downstream scripts.

    Args:
        included_loci: Dict of cluster -> GDLocus objects
        mappings: List of LocusBinMapping objects
        output_dir: Output directory
    """
    print("\nWriting locus metadata...")

    # 1. Write bin-to-interval mappings
    bin_mapping_rows = []
    for mapping in mappings:
        bin_mapping_rows.append({
            "cluster": mapping.cluster,
            "interval": mapping.interval_name,
            "chr": mapping.chrom,
            "start": mapping.start,
            "end": mapping.end,
            "array_idx": mapping.array_idx,
        })

    bin_mapping_df = pd.DataFrame(bin_mapping_rows)
    bin_mapping_output = os.path.join(output_dir, "bin_mappings.tsv.gz")
    bin_mapping_df.to_csv(bin_mapping_output, sep="\t", index=False, compression="gzip")
    print("  Saved bin mapping table")
    print(f"  Rows: {len(bin_mapping_df):,} bins")

    # 2. Write locus definitions with interval coordinates
    locus_rows = []
    for cluster, locus in included_loci.items():
        # Get intervals for this locus
        for start, end, name in locus.get_intervals():
            locus_rows.append({
                "cluster": cluster,
                "interval": name,
                "chr": locus.chrom,
                "start": start,
                "end": end,
            })

    locus_df = pd.DataFrame(locus_rows)
    locus_output = os.path.join(output_dir, "locus_intervals.tsv.gz")
    locus_df.to_csv(locus_output, sep="\t", index=False, compression="gzip")
    print("  Saved locus interval table")
    print(f"  Rows: {len(locus_df):,} intervals")

    # 3. Write GD-entry → interval mapping
    #
    # This is the crucial bridge between the GD table (identified by GD_ID)
    # and the posteriors / bin_mappings (identified by cluster + interval).
    #
    # A single GD entry may span *multiple* sub-intervals in the posteriors.
    # For example, a BP1→BP3 entry in a cluster that also defines BP2 will
    # cover both the "1-2" and "2-3" intervals, neither of which is named
    # "1-3".  Without this file users have no way to map a GD_ID to the
    # posterior rows that carry its copy-number signal.
    #
    # Columns:
    #   GD_ID        – identifier from the input GD table
    #   cluster      – cluster key (matches cn_posteriors.cluster)
    #   svtype       – DEL / DUP
    #   BP1, BP2     – breakpoint names for this GD entry
    #   interval     – sub-interval name (matches cn_posteriors.interval)
    #   chr          – chromosome
    #   interval_start, interval_end – genomic coordinates of the sub-interval
    gd_entry_rows = []
    for cluster, locus in included_loci.items():
        for entry in locus.gd_entries:
            covered = locus.get_intervals_between(entry["BP1"], entry["BP2"])
            for iv_start, iv_end, iv_name in covered:
                gd_entry_rows.append({
                    "GD_ID": entry["GD_ID"],
                    "cluster": cluster,
                    "svtype": entry["svtype"],
                    "BP1": entry["BP1"],
                    "BP2": entry["BP2"],
                    "interval": iv_name,
                    "chr": locus.chrom,
                    "interval_start": iv_start,
                    "interval_end": iv_end,
                })

    gd_entry_df = pd.DataFrame(gd_entry_rows)
    gd_entry_output = os.path.join(output_dir, "gd_entry_intervals.tsv.gz")
    gd_entry_df.to_csv(gd_entry_output, sep="\t", index=False, compression="gzip")
    print("  Saved GD-entry interval mapping table")
    print(f"  Rows: {len(gd_entry_df):,} (GD entry × interval)")


def estimate_ploidy(
    df: pd.DataFrame,
    output_dir: str,
) -> pd.DataFrame:
    """
    Estimate ploidy for each sample/contig pair from the filtered bin set.

    Uses the median normalized depth across all filtered bins on each
    chromosome for each sample. Since the data is normalized so that CN=2
    corresponds to a depth of 2.0, the rounded median gives the ploidy.

    Args:
        df: Filtered DataFrame with bins as rows and samples as columns.
            Expected to already be normalized so diploid depth ≈ 2.0.
        output_dir: Directory to write the ploidy table.

    Returns:
        DataFrame with columns: sample, contig, median_depth, ploidy
    """
    sample_cols = get_sample_columns(df)

    print(f"\n{'=' * 80}")
    print("ESTIMATING PLOIDY PER SAMPLE / CONTIG")
    print(f"{'=' * 80}")

    rows = []
    for contig, contig_df in df.groupby("Chr"):
        depths = contig_df[sample_cols].values  # bins × samples
        medians = np.median(depths, axis=0)     # per-sample median
        for sample_id, med in zip(sample_cols, medians):
            ploidy = int(np.round(med))
            rows.append({
                "sample": sample_id,
                "contig": contig,
                "median_depth": float(med),
                "ploidy": ploidy,
            })

    ploidy_df = pd.DataFrame(rows)

    # Summary
    n_samples = len(sample_cols)
    n_contigs = ploidy_df["contig"].nunique()
    print(f"  Samples: {n_samples}")
    print(f"  Contigs: {n_contigs}")
    counts = ploidy_df["ploidy"].value_counts().sort_index()
    dist_str = ", ".join(f"ploidy {p}: {n}" for p, n in counts.items())
    print(f"  Ploidy distribution across sample/contig pairs: {dist_str}")

    # Write to disk
    output_path = os.path.join(output_dir, "ploidy_estimates.tsv")
    ploidy_df.to_csv(output_path, sep="\t", index=False)
    print("  Saved ploidy estimate table")
    print(f"  Rows: {len(ploidy_df):,}")
    print(f"{'=' * 80}\n")

    return ploidy_df


def build_ploidy_map(ploidy_df: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    """Build a ``{(sample, chrom): ploidy}`` lookup from the ploidy table.

    This is the format consumed by the quality-filtering helpers in
    :mod:`gatk_sv_gd.bins` for ploidy-adjusted median/MAD computation.
    """
    return {
        (str(row["sample"]), str(row["contig"])): int(row["ploidy"])
        for _, row in ploidy_df.iterrows()
    }
