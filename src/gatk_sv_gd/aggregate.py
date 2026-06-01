"""Aggregate one or more gatk-sv-gd runs into a PDF report."""

import argparse
import math
import os
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from gatk_sv_gd.annotations import FlankCompressor
from gatk_sv_gd._util import get_logger, setup_logging
from gatk_sv_gd.models import GDTable
from gatk_sv_gd.plot import (
    _render_pdf_sample_page,
    _select_baf_plot_support_columns,
)

_CONFIDENCE_COLUMNS = ("qual_score", "confidence_score", "log_prob_score")
_REQUIRED_CALL_COLUMNS = {
    "sample",
    "cluster",
    "GD_ID",
    "chrom",
    "start",
    "end",
    "svtype",
    "is_carrier",
    "is_best_match",
    "calling_method",
    "call_criteria_mean_coverage",
    "call_criteria_interval_confidence",
    "call_criteria_flank_non_event_confidence",
}
_REQUIRED_PLOIDY_COLUMNS = {"sample", "contig", "ploidy"}
_REQUIRED_GD_TABLE_COLUMNS = {
    "chr",
    "start_GRCh38",
    "end_GRCh38",
    "GD_ID",
    "svtype",
    "cluster",
}

_OPTIONAL_ARTIFACTS = (
    ("sample_posteriors", ("infer", "sample_posteriors.tsv.gz")),
    ("bin_posteriors", ("infer", "bin_posteriors.tsv.gz")),
    ("event_marginals", ("call", "event_marginals.tsv.gz")),
    ("eval_report", ("eval", "truth_evaluation_report.tsv")),
    ("carrier_summary_png", ("plot", "carrier_summary.png")),
    ("confidence_distribution_png", ("plot", "confidence_distribution.png")),
    ("carrier_plots_pdf", ("plot", "carrier_plots.pdf")),
    ("true_positives_pdf", ("plot", "true_positives.pdf")),
    ("false_positives_pdf", ("plot", "false_positives.pdf")),
    ("false_negatives_pdf", ("plot", "false_negatives.pdf")),
    ("anomalous_discrepancies_pdf", ("plot", "anomalous_discrepancies.pdf")),
)

_SUMMARY_COLUMNS = ["metric", "value"]
_INVENTORY_COLUMNS = [
    "batch_id",
    "batch_label",
    "work_dir",
    "sample_count",
    "modeled_gd_count",
    "modeled_locus_count",
    "call_record_count",
    "carrier_record_count",
    "high_confidence_carrier_record_count",
    "low_confidence_carrier_record_count",
    "eval_report_present",
    "required_artifacts_present",
    "optional_artifacts_present",
    "optional_artifacts_missing",
]
_CASE_COLUMNS = [
    "batch_id",
    "batch_label",
    "work_dir",
    "sample",
    "sample_key",
    "carrier_category",
    "GD_ID",
    "cluster",
    "chrom",
    "start",
    "end",
    "svtype",
    "BP1",
    "BP2",
    "is_terminal",
    "n_bins",
    "mean_depth",
    "sample_ploidy",
    "matched_haplotype",
    "hap_cn_state",
    "matched_seg_start",
    "matched_seg_end",
    "matched_seg_n_bins",
    "matched_interval_bp",
    "interval_coverage",
    "reciprocal_overlap",
    "min_interval_confidence",
    "left_flank_non_event_median",
    "right_flank_non_event_median",
    "left_flank_status",
    "right_flank_status",
    "min_flank_non_event_confidence",
    "confidence_column",
    "confidence_value",
    "log_prob_score",
    "confidence_score",
    "qual_score",
    "calling_method",
]
_LOCUS_SUMMARY_COLUMNS = [
    "batch_id",
    "batch_label",
    "work_dir",
    "GD_ID",
    "cluster",
    "chrom",
    "start",
    "end",
    "svtype",
    "modeled_sample_count",
    "call_record_count",
    "carrier_record_count",
    "carrier_sample_count",
    "high_confidence_carrier_record_count",
    "high_confidence_carrier_sample_count",
    "low_confidence_carrier_record_count",
    "low_confidence_carrier_sample_count",
    "max_confidence",
    "median_confidence",
]
_MISSING_COLUMNS = ["batch_label", "work_dir", "artifact", "path", "reason"]

_INTERNAL_BATCH_COLUMNS = ["batch_id", "batch_label"]

_TRUE_VALUES = {"true", "t", "1", "yes", "y"}
_FALSE_VALUES = {"false", "f", "0", "no", "n", ""}

_PAGE_SIZE_IN = (8.5, 11.0)
_MARGIN_L = 0.75 / 8.5
_MARGIN_R = 1.0 - 0.75 / 8.5
_HEADER_RULE_Y = 1.0 - 0.65 / 11.0
_HEADER_TEXT_Y = 1.0 - 0.45 / 11.0
_FOOTER_RULE_Y = 0.55 / 11.0
_FOOTER_TEXT_Y = 0.35 / 11.0
_BODY_TOP = 1.0 - 0.85 / 11.0
_BODY_BOTTOM = 0.70 / 11.0
_INK = "#222222"
_MUTED = "#5A5A5A"
_RULE = "#808080"
_BAND = "#ECECEC"
_STRIPE = "#F5F5F5"
_REPORT_TITLE = "GATK-SV GD Aggregate Report"
_PLOT_MIN_GENE_LABEL_SPACING = 0.05
_PLOT_FLANK_SCALE = 0.20

_CASE_SECTIONS = (
    ("high_confidence_carrier", "Confident GD Calls"),
    ("low_confidence_carrier", "Non-confident GD Calls"),
)


@dataclass
class RunData:
    """Loaded artifacts for one standard gatk-sv-gd work directory."""

    batch_id: int
    batch_label: str
    work_dir: Path
    work_dir_input: str
    calls_df: pd.DataFrame
    ploidy_df: pd.DataFrame
    gd_table_df: pd.DataFrame
    eval_df: Optional[pd.DataFrame]
    optional_artifact_status: Dict[str, bool]
    missing_artifacts: List[Dict[str, str]]


@dataclass
class TocLinkSpec:
    target_page: int
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class PlotRunContext:
    loci_by_cluster: Dict[str, Any]
    depth_by_cluster: Dict[str, pd.DataFrame]
    minor_baf_by_cluster: Dict[str, pd.DataFrame]
    baf_variance_by_cluster: Dict[str, pd.DataFrame]
    baf_sites_by_cluster: Dict[str, pd.DataFrame]
    event_del_by_cluster: Dict[str, pd.DataFrame]
    event_dup_by_cluster: Dict[str, pd.DataFrame]
    baf_temperature_by_sample: Dict[str, float]
    unavailable_reason: Optional[str] = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the aggregate subcommand."""
    parser = argparse.ArgumentParser(
        description="Aggregate one or more gatk-sv-gd run directories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "work_dirs",
        nargs="+",
        help="One or more standard work directories produced by run_gd.sh",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory for the aggregate report and sidecar tables",
    )
    parser.add_argument(
        "--output-name",
        default="aggregate_report.pdf",
        help="Filename for the aggregate PDF report",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=50.0,
        help="Lower bound for including non-confident best-match calls in aggregate outputs",
    )
    parser.add_argument(
        "--batch-label",
        action="append",
        default=None,
        help="Optional batch label. Repeat once per work directory.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Validate aggregate arguments before doing filesystem work."""
    output_name = Path(args.output_name)
    if output_name.name != args.output_name or output_name.suffix.lower() != ".pdf":
        raise ValueError("--output-name must be a PDF filename, not a path")
    if args.min_confidence < 0:
        raise ValueError("--min-confidence must be non-negative")
    if args.batch_label is not None and len(args.batch_label) != len(args.work_dirs):
        raise ValueError("--batch-label must be provided once per work directory")


def _default_batch_labels(work_dirs: Iterable[str]) -> List[str]:
    """Return deterministic default batch labels from work directory names."""
    labels: List[str] = []
    seen: Dict[str, int] = {}
    for index, work_dir in enumerate(work_dirs, start=1):
        name = Path(work_dir).name or "batch_{}".format(index)
        count = seen.get(name, 0) + 1
        seen[name] = count
        labels.append(name if count == 1 else "{}_{}".format(name, count))
    return labels


def _artifact_path(work_dir: Path, parts: Sequence[str]) -> Path:
    path = work_dir
    for part in parts:
        path = path / part
    return path


def _missing_row(
    batch_label: str,
    work_dir: str,
    artifact: str,
    path: Path,
    reason: str = "not found",
) -> Dict[str, str]:
    return {
        "batch_label": batch_label,
        "work_dir": work_dir,
        "artifact": artifact,
        "path": str(path),
        "reason": reason,
    }


def _read_tsv(path: Path, *, compression: str = "infer") -> pd.DataFrame:
    """Read a TSV with a normalized error message."""
    try:
        return pd.read_csv(path, sep="\t", compression=compression)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError("Could not read TSV {}: {}".format(path, exc)) from exc


def _validate_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(
            "{} is missing required columns: {}".format(path, ", ".join(missing))
        )


def _get_confidence_column(calls_df: pd.DataFrame) -> str:
    """Return the preferred confidence column present in a calls table."""
    for column in _CONFIDENCE_COLUMNS:
        if column in calls_df.columns:
            return column
    raise ValueError(
        "Calls table is missing 'qual_score', 'confidence_score', and 'log_prob_score'."
    )


def _to_bool_value(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return bool(value)


def _normalize_bool_columns(calls_df: pd.DataFrame) -> pd.DataFrame:
    out = calls_df.copy()
    for column in ("is_carrier", "is_best_match", "is_terminal"):
        if column in out.columns:
            out[column] = out[column].map(_to_bool_value)
    return out


def _add_batch_columns(df: pd.DataFrame, run: RunData) -> pd.DataFrame:
    """Attach batch/run identifiers to a DataFrame."""
    out = df.copy()
    out.insert(0, "batch_id", run.batch_id)
    out.insert(1, "batch_label", run.batch_label)
    out.insert(2, "work_dir", run.work_dir_input)
    if "sample" in out.columns:
        out.insert(3, "sample_key", out["sample"].map(lambda sample: "{}/{}".format(run.batch_label, sample)))
    return out


def _load_optional_eval_report(
    work_dir: Path,
    batch_label: str,
    work_dir_input: str,
    missing: List[Dict[str, str]],
) -> Optional[pd.DataFrame]:
    path = work_dir / "eval" / "truth_evaluation_report.tsv"
    if not path.exists():
        return None
    try:
        eval_df = _read_tsv(path)
    except ValueError as exc:
        missing.append(_missing_row(batch_label, work_dir_input, "eval_report", path, str(exc)))
        return None
    if "GD_ID" not in eval_df.columns:
        missing.append(_missing_row(batch_label, work_dir_input, "eval_report", path, "missing GD_ID column"))
        return None
    return eval_df


def _scan_optional_artifacts(
    work_dir: Path,
    batch_label: str,
    work_dir_input: str,
) -> Tuple[Dict[str, bool], List[Dict[str, str]]]:
    status: Dict[str, bool] = {}
    missing: List[Dict[str, str]] = []
    for artifact, parts in _OPTIONAL_ARTIFACTS:
        path = _artifact_path(work_dir, parts)
        present = path.exists()
        status[artifact] = present
        if not present:
            missing.append(_missing_row(batch_label, work_dir_input, artifact, path))
    return status, missing


def _load_run_data(
    work_dir: str,
    *,
    batch_id: int,
    batch_label: str,
) -> RunData:
    """Load required and optional artifacts from one standard work directory."""
    root = Path(work_dir)
    call_path = root / "call" / "gd_cnv_calls.tsv.gz"
    ploidy_path = root / "preprocess" / "ploidy_estimates.tsv"
    gd_table_path = root / "preprocess" / "gd_table_filtered.tsv"

    for path, artifact in (
        (call_path, "calls"),
        (ploidy_path, "ploidy table"),
        (gd_table_path, "filtered GD table"),
    ):
        if not path.exists():
            raise FileNotFoundError("Missing required {} file: {}".format(artifact, path))

    calls_df = _normalize_bool_columns(_read_tsv(call_path))
    _validate_columns(calls_df, _REQUIRED_CALL_COLUMNS, call_path)
    _get_confidence_column(calls_df)

    ploidy_df = _read_tsv(ploidy_path)
    _validate_columns(ploidy_df, _REQUIRED_PLOIDY_COLUMNS, ploidy_path)

    gd_table_df = _read_tsv(gd_table_path)
    _validate_columns(gd_table_df, _REQUIRED_GD_TABLE_COLUMNS, gd_table_path)

    optional_status, missing_artifacts = _scan_optional_artifacts(root, batch_label, work_dir)
    eval_df = _load_optional_eval_report(root, batch_label, work_dir, missing_artifacts)
    optional_status["eval_report"] = eval_df is not None

    return RunData(
        batch_id=batch_id,
        batch_label=batch_label,
        work_dir=root,
        work_dir_input=work_dir,
        calls_df=calls_df,
        ploidy_df=ploidy_df,
        gd_table_df=gd_table_df,
        eval_df=eval_df,
        optional_artifact_status=optional_status,
        missing_artifacts=missing_artifacts,
    )


def _load_runs(args: argparse.Namespace) -> List[RunData]:
    labels = args.batch_label or _default_batch_labels(args.work_dirs)
    return [
        _load_run_data(
            work_dir,
            batch_id=index,
            batch_label=label,
        )
        for index, (work_dir, label) in enumerate(zip(args.work_dirs, labels), start=1)
    ]


def _group_plot_frames_by_cluster(frame: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if frame is None or frame.empty or "Cluster" not in frame.columns:
        return {}
    return {
        str(cluster): group.sort_values("Start").reset_index(drop=True)
        for cluster, group in frame.groupby("Cluster", sort=False)
    }


def _pivot_plot_matrix(cn_posteriors_df: pd.DataFrame, value_column: str) -> Optional[pd.DataFrame]:
    if value_column not in cn_posteriors_df.columns:
        return None
    frame = cn_posteriors_df.pivot(
        index=["cluster", "chr", "start", "end"],
        columns="sample",
        values=value_column,
    ).reset_index()
    return frame.rename(columns={
        "cluster": "Cluster",
        "chr": "Chr",
        "start": "Start",
        "end": "End",
    })


def _load_plot_run_context(run: RunData) -> PlotRunContext:
    def _unavailable(reason: str) -> PlotRunContext:
        return PlotRunContext(
            loci_by_cluster={},
            depth_by_cluster={},
            minor_baf_by_cluster={},
            baf_variance_by_cluster={},
            baf_sites_by_cluster={},
            event_del_by_cluster={},
            event_dup_by_cluster={},
            baf_temperature_by_sample={},
            unavailable_reason=reason,
        )

    cn_posteriors_path = run.work_dir / "infer" / "cn_posteriors.tsv.gz"
    if not cn_posteriors_path.exists():
        return _unavailable("missing infer/cn_posteriors.tsv.gz")

    try:
        cn_posteriors_df = _read_tsv(cn_posteriors_path)
    except ValueError as exc:
        return _unavailable(str(exc))
    if cn_posteriors_df.empty:
        return _unavailable("infer/cn_posteriors.tsv.gz is empty")

    cn_posteriors_df = cn_posteriors_df.drop_duplicates(
        subset=["cluster", "chr", "start", "end", "sample"],
    )
    depth_df = _pivot_plot_matrix(cn_posteriors_df, "depth")
    if depth_df is None:
        return _unavailable("infer/cn_posteriors.tsv.gz is missing depth values")

    baf_temperature_by_sample: Dict[str, float] = {}
    sample_posteriors_path = run.work_dir / "infer" / "sample_posteriors.tsv.gz"
    if sample_posteriors_path.exists():
        try:
            sample_posteriors_df = _read_tsv(sample_posteriors_path)
        except ValueError:
            sample_posteriors_df = pd.DataFrame()
        baf_temperature_column = None
        if "baf_variance_scale_map" in sample_posteriors_df.columns:
            baf_temperature_column = "baf_variance_scale_map"
        elif "baf_temperature_map" in sample_posteriors_df.columns:
            baf_temperature_column = "baf_temperature_map"
        if "sample" in sample_posteriors_df.columns and baf_temperature_column is not None:
            valid_rows = sample_posteriors_df[["sample", baf_temperature_column]].dropna()
            baf_temperature_by_sample = {
                str(row["sample"]): float(row[baf_temperature_column])
                for _, row in valid_rows.iterrows()
            }

    minor_baf_df = _pivot_plot_matrix(cn_posteriors_df, "minor_baf_median")
    baf_variance_column, baf_site_count_column = _select_baf_plot_support_columns(cn_posteriors_df)
    baf_variance_df = _pivot_plot_matrix(cn_posteriors_df, baf_variance_column) if baf_variance_column else None
    baf_sites_df = _pivot_plot_matrix(cn_posteriors_df, baf_site_count_column) if baf_site_count_column else None

    event_del_by_cluster: Dict[str, pd.DataFrame] = {}
    event_dup_by_cluster: Dict[str, pd.DataFrame] = {}
    event_marginals_path = run.work_dir / "call" / "event_marginals.tsv.gz"
    if event_marginals_path.exists():
        try:
            event_marginals_df = _read_tsv(event_marginals_path)
        except ValueError:
            event_marginals_df = pd.DataFrame()
        if not event_marginals_df.empty:
            event_marginals_df = event_marginals_df.rename(columns={
                "cluster": "Cluster",
                "chrom": "Chr",
                "start": "Start",
                "end": "End",
            })
            if "prob_del_event" in event_marginals_df.columns:
                event_del_by_cluster = _group_plot_frames_by_cluster(
                    event_marginals_df.pivot(
                        index=["Cluster", "Chr", "Start", "End"],
                        columns="sample",
                        values="prob_del_event",
                    ).reset_index()
                )
            if "prob_dup_event" in event_marginals_df.columns:
                event_dup_by_cluster = _group_plot_frames_by_cluster(
                    event_marginals_df.pivot(
                        index=["Cluster", "Chr", "Start", "End"],
                        columns="sample",
                        values="prob_dup_event",
                    ).reset_index()
                )

    try:
        gd_table = GDTable(str(run.work_dir / "preprocess" / "gd_table_filtered.tsv"))
    except Exception as exc:
        return _unavailable("could not load GD table for plot rendering: {}".format(exc))

    return PlotRunContext(
        loci_by_cluster=gd_table.loci,
        depth_by_cluster=_group_plot_frames_by_cluster(depth_df),
        minor_baf_by_cluster=_group_plot_frames_by_cluster(minor_baf_df),
        baf_variance_by_cluster=_group_plot_frames_by_cluster(baf_variance_df),
        baf_sites_by_cluster=_group_plot_frames_by_cluster(baf_sites_df),
        event_del_by_cluster=event_del_by_cluster,
        event_dup_by_cluster=event_dup_by_cluster,
        baf_temperature_by_sample=baf_temperature_by_sample,
    )


def _low_confidence_carrier_mask(
    calls_df: pd.DataFrame,
    *,
    min_confidence: float,
) -> pd.Series:
    best_match_mask = calls_df["is_best_match"].map(_to_bool_value)
    confident_mask = calls_df["is_carrier"].map(_to_bool_value)
    confidence_mask = calls_df["confidence_value"].ge(float(min_confidence)).fillna(False)
    return best_match_mask & (~confident_mask) & confidence_mask


def _build_calls_table(runs: List[RunData], *, min_confidence: float) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for run in runs:
        confidence_column = _get_confidence_column(run.calls_df)
        calls = _add_batch_columns(run.calls_df, run)
        calls["confidence_column"] = confidence_column
        calls["confidence_value"] = pd.to_numeric(
            calls[confidence_column],
            errors="coerce",
        )
        high_mask = calls["is_carrier"].map(_to_bool_value)
        low_mask = _low_confidence_carrier_mask(calls, min_confidence=min_confidence)
        calls["is_reportable_call"] = high_mask | low_mask
        calls["is_high_confidence_carrier"] = high_mask
        calls["is_low_confidence_carrier"] = low_mask
        calls["is_visible_carrier"] = high_mask | low_mask
        calls["carrier_category"] = np.where(
            high_mask,
            "high_confidence_carrier",
            np.where(
                low_mask,
                "low_confidence_carrier",
                "non_carrier",
            ),
        )
        frames.append(calls)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _build_cases_table(calls_df: pd.DataFrame) -> pd.DataFrame:
    if calls_df.empty:
        return pd.DataFrame(columns=_CASE_COLUMNS)
    visible_categories = {category for category, _title in _CASE_SECTIONS}
    cases = calls_df[calls_df["carrier_category"].isin(visible_categories)].copy()
    flank_threshold = pd.to_numeric(
        cases.get("call_criteria_flank_non_event_confidence"),
        errors="coerce",
    )
    left_flank_median = pd.to_numeric(
        cases.get("left_flank_non_event_median"),
        errors="coerce",
    )
    right_flank_median = pd.to_numeric(
        cases.get("right_flank_non_event_median"),
        errors="coerce",
    )
    cases["left_flank_status"] = np.where(
        left_flank_median.isna() | flank_threshold.isna() | left_flank_median.ge(flank_threshold),
        "PASS",
        "FAIL",
    )
    cases["right_flank_status"] = np.where(
        right_flank_median.isna() | flank_threshold.isna() | right_flank_median.ge(flank_threshold),
        "PASS",
        "FAIL",
    )
    for column in _CASE_COLUMNS:
        if column not in cases.columns:
            cases[column] = ""
    sort_columns = ["carrier_category", "batch_label", "cluster", "GD_ID", "sample"]
    cases = cases.sort_values(sort_columns, kind="mergesort")
    return cases[_CASE_COLUMNS].reset_index(drop=True)


def _first_value(series: pd.Series) -> Any:
    values = series.dropna()
    if values.empty:
        return ""
    return values.iloc[0]


def _numeric_median(series: pd.Series) -> Any:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return ""
    return float(values.median())


def _numeric_max(series: pd.Series) -> Any:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return ""
    return float(values.max())


def _build_locus_summary_table(calls_df: pd.DataFrame) -> pd.DataFrame:
    if calls_df.empty:
        return pd.DataFrame(columns=_LOCUS_SUMMARY_COLUMNS)
    rows: List[Dict[str, Any]] = []
    group_columns = ["batch_id", "batch_label", "GD_ID", "cluster", "svtype"]
    for group_values, group in calls_df.groupby(group_columns, dropna=False, sort=True):
        batch_id, batch_label, gd_id, cluster, svtype = group_values
        carrier_group = group[group["carrier_category"].isin({category for category, _title in _CASE_SECTIONS})]
        if carrier_group.empty:
            continue
        high_group = group[group["is_high_confidence_carrier"]]
        low_group = group[group.get("is_low_confidence_carrier", False)]
        rows.append({
            "batch_id": batch_id,
            "batch_label": batch_label,
            "work_dir": _first_value(group["work_dir"]) if "work_dir" in group.columns else "",
            "GD_ID": gd_id,
            "cluster": cluster,
            "chrom": _first_value(group["chrom"]) if "chrom" in group.columns else "",
            "start": _first_value(group["start"]) if "start" in group.columns else "",
            "end": _first_value(group["end"]) if "end" in group.columns else "",
            "svtype": svtype,
            "modeled_sample_count": int(group["sample"].nunique()),
            "call_record_count": int(len(group)),
            "carrier_record_count": int(len(carrier_group)),
            "carrier_sample_count": int(carrier_group["sample"].nunique()),
            "high_confidence_carrier_record_count": int(len(high_group)),
            "high_confidence_carrier_sample_count": int(high_group["sample"].nunique()),
            "low_confidence_carrier_record_count": int(len(low_group)),
            "low_confidence_carrier_sample_count": int(low_group["sample"].nunique()),
            "max_confidence": _numeric_max(group["confidence_value"]),
            "median_confidence": _numeric_median(group["confidence_value"]),
        })
    return pd.DataFrame(rows, columns=_LOCUS_SUMMARY_COLUMNS)


def _build_eval_table(runs: List[RunData]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for run in runs:
        if run.eval_df is None:
            continue
        frames.append(_add_batch_columns(run.eval_df, run))
    if not frames:
        return pd.DataFrame(columns=["batch_id", "batch_label", "work_dir", "GD_ID"])
    return pd.concat(frames, ignore_index=True, sort=False)


def _build_inventory_table(
    runs: List[RunData],
    calls_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        run_calls = calls_df[calls_df["batch_id"] == run.batch_id]
        carriers = run_calls[run_calls["is_reportable_call"]]
        high_carriers = run_calls[run_calls["is_high_confidence_carrier"]]
        optional_present = sum(1 for present in run.optional_artifact_status.values() if present)
        optional_missing = len(run.optional_artifact_status) - optional_present
        cluster_series = run.gd_table_df["cluster"].fillna("").astype(str)
        rows.append({
            "batch_id": run.batch_id,
            "batch_label": run.batch_label,
            "work_dir": run.work_dir_input,
            "sample_count": int(run.ploidy_df["sample"].nunique()),
            "modeled_gd_count": int(run.gd_table_df["GD_ID"].nunique()),
            "modeled_locus_count": int(cluster_series.where(cluster_series != "", run.gd_table_df["GD_ID"].astype(str)).nunique()),
            "call_record_count": int(len(run_calls)),
            "carrier_record_count": int(len(carriers)),
            "high_confidence_carrier_record_count": int(len(high_carriers)),
            "low_confidence_carrier_record_count": int(len(carriers) - len(high_carriers)),
            "eval_report_present": bool(run.eval_df is not None),
            "required_artifacts_present": True,
            "optional_artifacts_present": int(optional_present),
            "optional_artifacts_missing": int(optional_missing),
        })
    return pd.DataFrame(rows, columns=_INVENTORY_COLUMNS)


def _summary_value_from_calls(calls_df: pd.DataFrame, column: str) -> Optional[Any]:
    if column not in calls_df.columns or calls_df.empty:
        return None
    values = [value for value in calls_df[column].dropna().tolist() if str(value) != ""]
    if not values:
        return None
    normalized = []
    for value in values:
        if isinstance(value, (int, float, np.integer, np.floating)):
            normalized.append(float(value))
        else:
            normalized.append(str(value))
    unique_values: List[Any] = []
    for value in normalized:
        if value not in unique_values:
            unique_values.append(value)
    if len(unique_values) == 1:
        return unique_values[0]
    return ", ".join(_format_pdf_value(value) for value in unique_values)


def _sum_numeric_column(df: pd.DataFrame, column: str) -> Optional[float]:
    if column not in df.columns or df.empty:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.sum())


def _build_summary_table(
    inventory_df: pd.DataFrame,
    calls_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    locus_summary_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    runs: List[RunData],
    *,
    min_confidence: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [
        {"metric": "n_batches", "value": int(len(runs))},
        {"metric": "n_samples", "value": int(inventory_df["sample_count"].sum())},
        {"metric": "n_modeled_gd_ids", "value": int(calls_df["GD_ID"].nunique()) if "GD_ID" in calls_df.columns else 0},
        {"metric": "n_modeled_loci", "value": int(calls_df["cluster"].nunique()) if "cluster" in calls_df.columns else 0},
        {"metric": "n_carrier_events", "value": int(len(cases_df))},
        {"metric": "n_carrier_samples", "value": int(cases_df["sample_key"].nunique()) if "sample_key" in cases_df.columns else 0},
        {
            "metric": "n_high_confidence_carrier_events",
            "value": int((cases_df["carrier_category"] == "high_confidence_carrier").sum()) if "carrier_category" in cases_df.columns else 0,
        },
        {
            "metric": "n_low_confidence_carrier_events",
            "value": int((cases_df["carrier_category"] == "low_confidence_carrier").sum()) if "carrier_category" in cases_df.columns else 0,
        },
        {"metric": "n_batches_with_eval_report", "value": int(sum(run.eval_df is not None for run in runs))},
        {"metric": "n_eval_rows", "value": int(len(eval_df))},
        {"metric": "n_locus_summary_rows", "value": int(len(locus_summary_df))},
        {"metric": "aggregate_min_confidence", "value": float(min_confidence)},
    ]

    if "svtype" in cases_df.columns:
        for svtype, count in cases_df.groupby("svtype", dropna=False).size().items():
            rows.append({
                "metric": "n_carrier_events_{}".format(str(svtype).lower()),
                "value": int(count),
            })

    for metric, column in (
        ("calling_modes", "calling_method"),
        ("call_mean_coverage_threshold", "call_criteria_mean_coverage"),
        ("call_interval_confidence_threshold", "call_criteria_interval_confidence"),
        ("call_flank_non_event_confidence_threshold", "call_criteria_flank_non_event_confidence"),
    ):
        value = _summary_value_from_calls(calls_df, column)
        if value is not None:
            rows.append({"metric": metric, "value": value})

    total_tp = _sum_numeric_column(eval_df, "TP")
    total_fp = _sum_numeric_column(eval_df, "FP")
    total_fn = _sum_numeric_column(eval_df, "FN")
    if total_tp is not None or total_fp is not None or total_fn is not None:
        tp_value = total_tp or 0.0
        fp_value = total_fp or 0.0
        fn_value = total_fn or 0.0
        rows.extend([
            {"metric": "aggregate_TP", "value": int(tp_value)},
            {"metric": "aggregate_FP", "value": int(fp_value)},
            {"metric": "aggregate_FN", "value": int(fn_value)},
            {
                "metric": "aggregate_sensitivity",
                "value": round(tp_value / (tp_value + fn_value), 4) if (tp_value + fn_value) > 0 else "NA",
            },
            {
                "metric": "aggregate_precision",
                "value": round(tp_value / (tp_value + fp_value), 4) if (tp_value + fp_value) > 0 else "NA",
            },
        ])

    return pd.DataFrame(rows, columns=_SUMMARY_COLUMNS)


def _build_report_tables(
    runs: List[RunData],
    *,
    min_confidence: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calls_df = _build_calls_table(runs, min_confidence=min_confidence)
    cases_df = _build_cases_table(calls_df)
    locus_summary_df = _build_locus_summary_table(calls_df)
    eval_df = _build_eval_table(runs)
    inventory_df = _build_inventory_table(runs, calls_df)
    missing_rows = [row for run in runs for row in run.missing_artifacts]
    missing_df = pd.DataFrame(missing_rows, columns=_MISSING_COLUMNS)
    summary_df = _build_summary_table(
        inventory_df,
        calls_df,
        cases_df,
        locus_summary_df,
        eval_df,
        runs,
        min_confidence=min_confidence,
    )
    return (
        summary_df,
        inventory_df,
        calls_df,
        cases_df,
        locus_summary_df,
        eval_df,
        missing_df,
    )


def _write_sidecars(
    output_dir: Path,
    summary_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    calls_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    locus_summary_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> List[Path]:
    outputs = {
        "aggregate_summary.tsv": summary_df,
        "aggregate_inventory.tsv": inventory_df.drop(columns=_INTERNAL_BATCH_COLUMNS, errors="ignore"),
        "aggregate_calls.tsv": calls_df.drop(columns=_INTERNAL_BATCH_COLUMNS, errors="ignore"),
        "aggregate_cases.tsv": cases_df.drop(columns=_INTERNAL_BATCH_COLUMNS, errors="ignore"),
        "aggregate_locus_summary.tsv": locus_summary_df.drop(columns=_INTERNAL_BATCH_COLUMNS, errors="ignore"),
        "aggregate_eval.tsv": eval_df.drop(columns=_INTERNAL_BATCH_COLUMNS, errors="ignore"),
        "aggregate_missing_artifacts.tsv": missing_df.drop(columns=["batch_label"], errors="ignore"),
    }
    paths: List[Path] = []
    for name, table in outputs.items():
        path = output_dir / name
        table.to_csv(path, sep="\t", index=False)
        paths.append(path)
    return paths


def _format_pdf_value(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)):
            return "{:.4g}".format(float(value))
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _humanize_metric(name: str) -> str:
    overrides = {
        "n_batches": "Batches",
        "n_samples": "Samples",
        "n_modeled_gd_ids": "Modeled GD IDs",
        "n_modeled_loci": "Modeled loci",
        "n_carrier_events": "Carrier events",
        "n_carrier_samples": "Carrier samples",
        "n_high_confidence_carrier_events": "Confident carrier events",
        "n_low_confidence_carrier_events": "Non-confident carrier events",
        "n_batches_with_eval_report": "Batches with eval report",
        "n_eval_rows": "Evaluation rows",
        "n_locus_summary_rows": "Locus burden rows",
        "aggregate_min_confidence": "Aggregate non-confident min confidence",
        "aggregate_TP": "Aggregate TP",
        "aggregate_FP": "Aggregate FP",
        "aggregate_FN": "Aggregate FN",
        "aggregate_sensitivity": "Aggregate sensitivity",
        "aggregate_precision": "Aggregate precision",
        "calling_modes": "Calling mode(s)",
        "call_mean_coverage_threshold": "Call min mean coverage",
        "call_interval_confidence_threshold": "Call min interval confidence",
        "call_flank_non_event_confidence_threshold": "Call min flank non-event confidence",
    }
    if name in overrides:
        return overrides[name]
    if name.startswith("n_carrier_events_"):
        return "Carrier events ({})".format(name.replace("n_carrier_events_", "").upper())
    return name.replace("_", " ").capitalize()


def _humanize_category(category: str) -> str:
    labels = {
        "high_confidence_carrier": "Confident GD call",
        "low_confidence_carrier": "Non-confident GD call",
        "non_carrier": "Non-carrier",
    }
    return labels.get(str(category), str(category).replace("_", " ").title())


def _new_page(pdf_state: Dict[str, Any], *, header: Optional[str] = None) -> plt.Figure:
    """Create a report page with ploidy-style running header and footer."""
    fig = plt.figure(figsize=_PAGE_SIZE_IN)
    fig.patch.set_facecolor("white")
    pdf_state["page"] += 1
    page_num = int(pdf_state["page"])

    if header:
        fig.text(
            _MARGIN_L, _HEADER_TEXT_Y, header,
            fontsize=8, color=_MUTED, ha="left", va="center",
            family="sans-serif",
        )
    fig.text(
        _MARGIN_R, _HEADER_TEXT_Y, _REPORT_TITLE,
        fontsize=8, color=_MUTED, ha="right", va="center",
        family="sans-serif",
    )
    fig.add_artist(Line2D(
        [_MARGIN_L, _MARGIN_R], [_HEADER_RULE_Y, _HEADER_RULE_Y],
        color=_RULE, linewidth=0.6, transform=fig.transFigure,
    ))
    fig.add_artist(Line2D(
        [_MARGIN_L, _MARGIN_R], [_FOOTER_RULE_Y, _FOOTER_RULE_Y],
        color=_RULE, linewidth=0.6, transform=fig.transFigure,
    ))
    fig.text(
        _MARGIN_L, _FOOTER_TEXT_Y, pdf_state.get("footer_left", ""),
        fontsize=7, color=_MUTED, ha="left", va="center",
        family="sans-serif",
    )
    fig.text(
        _MARGIN_R, _FOOTER_TEXT_Y, "Page {}".format(page_num),
        fontsize=7, color=_MUTED, ha="right", va="center",
        family="sans-serif",
    )
    return fig


def _save_page(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig)
    plt.close(fig)


def _section_band(fig: plt.Figure, y: float, title: str, *, eyebrow: Optional[str] = None) -> float:
    if eyebrow:
        fig.text(
            _MARGIN_L, y, eyebrow.upper(),
            fontsize=7, color=_MUTED, ha="left", va="top",
            family="sans-serif",
        )
        y -= 0.012
    fig.text(
        _MARGIN_L, y, title,
        fontsize=12, fontweight="bold", color=_INK, ha="left", va="top",
        family="sans-serif",
    )
    y -= 0.018
    fig.add_artist(Line2D(
        [_MARGIN_L, _MARGIN_R], [y, y],
        color=_INK, linewidth=0.8, transform=fig.transFigure,
    ))
    return y - 0.014


def _wrap_text(value: Any, max_chars: int) -> str:
    text = _format_pdf_value(value)
    if max_chars <= 1 or len(text) <= max_chars:
        return text
    lines: List[str] = []
    for line in text.splitlines() or [""]:
        wrapped = textwrap.wrap(
            line,
            width=max_chars,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        lines.extend(wrapped or [""])
    return "\n".join(lines)


def _draw_paragraph(
    fig: plt.Figure,
    text: str,
    *,
    start_y: float,
    fontsize: int = 8,
    color: str = _INK,
    line_height: float = 0.018,
) -> float:
    y = start_y
    for line in str(text).split("\n"):
        fig.text(
            _MARGIN_L, y, line,
            fontsize=fontsize, color=color, ha="left", va="top",
            family="sans-serif",
        )
        y -= line_height
    return y


def _draw_kv_block(
    fig: plt.Figure,
    items: Sequence[Tuple[str, str]],
    *,
    start_y: float,
    columns: int = 2,
    line_height: float = 0.022,
    label_fraction: float = 0.48,
) -> float:
    if not items:
        return start_y
    width = _MARGIN_R - _MARGIN_L
    gutter = 0.020
    col_width = (width - gutter * (columns - 1)) / columns
    rows = int(math.ceil(len(items) / float(columns)))
    for index, (label, value) in enumerate(items):
        col = index // rows
        row = index % rows
        x = _MARGIN_L + col * (col_width + gutter)
        y = start_y - row * line_height
        label_width = col_width * label_fraction
        fig.text(
            x, y, _wrap_text(label, 28),
            fontsize=8, color=_MUTED, ha="left", va="top",
            family="sans-serif",
        )
        fig.text(
            x + label_width, y, _wrap_text(value, 34),
            fontsize=8, color=_INK, ha="left", va="top",
            family="sans-serif", fontweight="semibold",
        )
    return start_y - rows * line_height - 0.006


def _prepare_table_display(
    table_df: pd.DataFrame,
    columns: Sequence[str],
    column_labels: Optional[Dict[str, str]],
    col_widths: Sequence[float],
) -> pd.DataFrame:
    available = [column for column in columns if column in table_df.columns]
    display_df = table_df.loc[:, available].copy()
    if hasattr(display_df, "map"):
        display_df = display_df.map(_format_pdf_value)
    else:
        display_df = display_df.applymap(_format_pdf_value)

    total_width = _MARGIN_R - _MARGIN_L
    width_sum = float(sum(col_widths)) if col_widths else 1.0
    normalized = [float(width) / width_sum for width in col_widths]
    for column, width_fraction in zip(available, normalized):
        chars = max(6, int((total_width * width_fraction) / 0.0085))
        display_df[column] = display_df[column].map(lambda value, chars=chars: _wrap_text(value, chars))
    if column_labels:
        display_df = display_df.rename(columns=column_labels)
    wrapped_labels = []
    for column_name, width_fraction in zip(display_df.columns, normalized):
        chars = max(7, int((total_width * width_fraction) / 0.0105))
        wrapped_labels.append(_wrap_text(column_name, chars))
    display_df.columns = wrapped_labels
    return display_df


def _draw_table(
    fig: plt.Figure,
    table_df: pd.DataFrame,
    columns: Sequence[str],
    *,
    start_y: float,
    max_rows: int = 24,
    col_widths: Optional[Sequence[float]] = None,
    column_labels: Optional[Dict[str, str]] = None,
    font_size: int = 7,
    row_height: float = 0.027,
) -> float:
    available = [column for column in columns if column in table_df.columns]
    if table_df.empty or not available:
        fig.text(
            _MARGIN_L, start_y - 0.005, "No rows.",
            fontsize=8, color=_MUTED, ha="left", va="top", style="italic",
            family="sans-serif",
        )
        return start_y - 0.034

    rows = table_df.head(max_rows)
    if col_widths is None:
        col_widths = [1.0 / len(available)] * len(available)
    display_df = _prepare_table_display(rows, available, column_labels, col_widths)
    n_rows = len(display_df) + 1
    table_height = min(0.70, max(0.08, n_rows * row_height))
    bottom = max(_BODY_BOTTOM + 0.025, start_y - table_height)
    ax = fig.add_axes([_MARGIN_L, bottom, _MARGIN_R - _MARGIN_L, start_y - bottom])
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=list(display_df.columns),
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.0, 1.0, 1.0],
        colWidths=list(col_widths),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    for (row_index, _column_index), cell in table.get_celld().items():
        cell.PAD = 0.015
        cell.set_linewidth(0)
        if row_index == 0:
            cell.set_facecolor(_BAND)
            cell.visible_edges = "B"
            cell.set_edgecolor(_INK)
            cell.set_linewidth(0.6)
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(_STRIPE if row_index % 2 == 0 else "white")
            cell.visible_edges = ""
        cell.get_text().set_color(_INK)
        cell.get_text().set_ha("left")
        cell.get_text().set_va("top")

    if len(table_df) > max_rows:
        fig.text(
            _MARGIN_L, bottom - 0.010,
            "Showing first {} of {} rows; full data are in the TSV sidecar.".format(max_rows, len(table_df)),
            fontsize=7, color=_MUTED, ha="left", va="top", style="italic",
            family="sans-serif",
        )
        bottom -= 0.014
    return bottom - 0.016


def _draw_table_section_pages(
    pdf: PdfPages,
    pdf_state: Dict[str, Any],
    table_df: pd.DataFrame,
    columns: Sequence[str],
    *,
    header: str,
    title: str,
    eyebrow: str,
    max_rows: int,
    col_widths: Optional[Sequence[float]] = None,
    column_labels: Optional[Dict[str, str]] = None,
    font_size: int = 7,
    row_height: float = 0.027,
) -> None:
    page_count = max(1, int(math.ceil(len(table_df) / float(max_rows)))) if not table_df.empty else 1
    for page_index in range(page_count):
        fig = _new_page(pdf_state, header=header)
        section_title = title if page_index == 0 else "{} (continued)".format(title)
        y = _section_band(fig, _BODY_TOP, section_title, eyebrow=eyebrow)
        page_df = table_df.iloc[page_index * max_rows:(page_index + 1) * max_rows]
        _draw_table(
            fig,
            page_df,
            columns,
            start_y=y,
            max_rows=max_rows,
            col_widths=col_widths,
            column_labels=column_labels,
            font_size=font_size,
            row_height=row_height,
        )
        _save_page(pdf, fig)


def _summary_metrics_for_cover(summary_df: pd.DataFrame) -> List[Tuple[str, str]]:
    wanted = [
        "n_batches",
        "n_samples",
        "n_modeled_gd_ids",
        "n_modeled_loci",
        "n_carrier_events",
        "n_carrier_samples",
        "n_high_confidence_carrier_events",
        "n_low_confidence_carrier_events",
        "n_batches_with_eval_report",
    ]
    summary = summary_df.set_index("metric") if not summary_df.empty else pd.DataFrame()
    items: List[Tuple[str, str]] = []
    for metric in wanted:
        value = summary.loc[metric, "value"] if metric in summary.index else ""
        items.append((_humanize_metric(metric), _format_pdf_value(value)))
    return items


def _add_cover_page(
    pdf: PdfPages,
    pdf_state: Dict[str, Any],
    runs: List[RunData],
    summary_df: pd.DataFrame,
    toc_entries: Sequence[Tuple[str, int]],
) -> List[TocLinkSpec]:
    fig = _new_page(pdf_state)
    fig.text(
        0.5, 0.83, _REPORT_TITLE,
        fontsize=22, fontweight="bold", color=_INK, ha="center", va="center",
        family="sans-serif",
    )
    fig.text(
        0.5, 0.79, "Genomic Disorder Copy-Number Cohort Summary",
        fontsize=11, color=_MUTED, ha="center", va="center", style="italic",
        family="sans-serif",
    )
    for offset, linewidth in ((0.770, 1.2), (0.766, 0.4)):
        fig.add_artist(Line2D(
            [_MARGIN_L + 0.08, _MARGIN_R - 0.08], [offset, offset],
            color=_INK, linewidth=linewidth, transform=fig.transFigure,
        ))

    summary = summary_df.set_index("metric") if not summary_df.empty else pd.DataFrame()
    cover_settings: List[Tuple[str, str]] = [
        ("Report generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Batches analysed", str(len(runs))),
    ]
    for metric in (
        "aggregate_min_confidence",
        "calling_modes",
        "call_mean_coverage_threshold",
        "call_interval_confidence_threshold",
        "call_flank_non_event_confidence_threshold",
    ):
        if metric in summary.index:
            cover_settings.append(
                (_humanize_metric(metric), _format_pdf_value(summary.loc[metric, "value"]))
            )

    y = _draw_kv_block(
        fig,
        cover_settings,
        start_y=0.71,
        columns=2,
        line_height=0.024,
    )
    y = _section_band(fig, y - 0.012, "Summary", eyebrow="Front matter")
    y = _draw_kv_block(fig, _summary_metrics_for_cover(summary_df), start_y=y, columns=2)
    y = _section_band(fig, y - 0.018, "Contents", eyebrow="Front matter")
    link_specs: List[TocLinkSpec] = []
    for title, page in toc_entries:
        fig.text(
            _MARGIN_L, y, title,
            fontsize=9, color="#0B57D0", ha="left", va="top",
            family="sans-serif",
        )
        fig.text(
            _MARGIN_R, y, str(page),
            fontsize=9, color="#0B57D0", ha="right", va="top",
            family="sans-serif",
        )
        fig.add_artist(Line2D(
            [_MARGIN_L, _MARGIN_R], [y - 0.014, y - 0.014],
            color="#0B57D0", linewidth=0.5, transform=fig.transFigure,
        ))
        link_specs.append(
            TocLinkSpec(
                target_page=int(page),
                x0=_MARGIN_L,
                y0=y - 0.020,
                x1=_MARGIN_R,
                y1=y + 0.002,
            )
        )
        y -= 0.022
    _save_page(pdf, fig)
    return link_specs


def _add_pdf_internal_links(
    pdf: PdfPages,
    source_page: int,
    link_specs: Sequence[TocLinkSpec],
) -> None:
    if not link_specs:
        return
    from matplotlib.backends.backend_pdf import Name

    pdf_file = pdf._ensure_file()
    if source_page < 1 or source_page > len(pdf_file._annotations):
        return
    page_annotations = pdf_file._annotations[source_page - 1][1]
    page_width = 72.0 * float(_PAGE_SIZE_IN[0])
    page_height = 72.0 * float(_PAGE_SIZE_IN[1])
    for spec in link_specs:
        if spec.target_page < 1 or spec.target_page > len(pdf_file.pageList):
            continue
        page_annotations.append(
            {
                "Type": Name("Annot"),
                "Subtype": Name("Link"),
                "Rect": [
                    page_width * spec.x0,
                    page_height * spec.y0,
                    page_width * spec.x1,
                    page_height * spec.y1,
                ],
                "Border": [0, 0, 0],
                "A": {
                    "S": Name("GoTo"),
                    "D": [pdf_file.pageList[spec.target_page - 1], Name("Fit")],
                },
            }
        )


def _add_inventory_pages(pdf: PdfPages, pdf_state: Dict[str, Any], inventory_df: pd.DataFrame) -> None:
    _draw_table_section_pages(
        pdf,
        pdf_state,
        inventory_df,
        [
            "batch_id", "batch_label", "sample_count", "modeled_gd_count",
            "modeled_locus_count", "carrier_record_count", "eval_report_present",
            "optional_artifacts_present", "optional_artifacts_missing",
        ],
        header="Inputs - Batch Inventory",
        title="Batch Inventory",
        eyebrow="Inputs",
        max_rows=18,
        col_widths=[0.07, 0.19, 0.10, 0.10, 0.10, 0.11, 0.09, 0.12, 0.12],
        column_labels={
            "batch_id": "ID",
            "batch_label": "Batch",
            "sample_count": "Samples",
            "modeled_gd_count": "GD IDs",
            "modeled_locus_count": "Loci",
            "carrier_record_count": "Carriers",
            "eval_report_present": "Eval",
            "optional_artifacts_present": "Optional present",
            "optional_artifacts_missing": "Optional missing",
        },
        row_height=0.031,
    )


def _add_summary_pages(
    pdf: PdfPages,
    pdf_state: Dict[str, Any],
    summary_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    locus_summary_df: pd.DataFrame,
) -> None:
    summary_display = summary_df.copy()
    if not summary_display.empty:
        summary_display["metric"] = summary_display["metric"].map(_humanize_metric)
    _draw_table_section_pages(
        pdf,
        pdf_state,
        summary_display,
        ["metric", "value"],
        header="Section 1 - Cohort Summary",
        title="Cohort Summary",
        eyebrow="Section 1",
        max_rows=28,
        col_widths=[0.72, 0.28],
        column_labels={"metric": "Metric", "value": "Value"},
        font_size=8,
    )

    case_display = cases_df.copy()
    if not case_display.empty:
        case_display["carrier_category"] = case_display["carrier_category"].map(_humanize_category)
    _draw_table_section_pages(
        pdf,
        pdf_state,
        case_display,
        [
            "carrier_category", "batch_label", "sample", "GD_ID", "cluster",
            "svtype", "left_flank_status", "right_flank_status", "confidence_value",
        ],
        header="Section 2 - Case Index",
        title="Case Index",
        eyebrow="Section 2",
        max_rows=16,
        col_widths=[0.18, 0.12, 0.11, 0.18, 0.15, 0.06, 0.08, 0.08, 0.04],
        column_labels={
            "carrier_category": "Category",
            "batch_label": "Batch",
            "sample": "Sample",
            "GD_ID": "GD ID",
            "cluster": "Locus",
            "svtype": "Type",
            "left_flank_status": "Left flank",
            "right_flank_status": "Right flank",
            "confidence_value": "Confidence",
        },
        row_height=0.045,
    )

    _draw_table_section_pages(
        pdf,
        pdf_state,
        locus_summary_df,
        [
            "batch_label", "GD_ID", "cluster", "svtype", "modeled_sample_count",
            "carrier_sample_count", "high_confidence_carrier_sample_count",
            "low_confidence_carrier_sample_count", "max_confidence",
        ],
        header="Section 3 - Locus Burden",
        title="Locus Burden",
        eyebrow="Section 3",
        max_rows=22,
        col_widths=[0.14, 0.15, 0.16, 0.08, 0.11, 0.11, 0.10, 0.10, 0.05],
        column_labels={
            "batch_label": "Batch",
            "GD_ID": "GD ID",
            "cluster": "Locus",
            "svtype": "Type",
            "modeled_sample_count": "Modeled samples",
            "carrier_sample_count": "Carrier samples",
            "high_confidence_carrier_sample_count": "Confident",
            "low_confidence_carrier_sample_count": "Non-confident",
            "max_confidence": "Max conf.",
        },
    )


def _case_section_start_pages(cases_df: pd.DataFrame, first_case_page: int) -> List[Tuple[str, int]]:
    entries: List[Tuple[str, int]] = []
    page = first_case_page
    for category, title in _CASE_SECTIONS:
        section_cases = cases_df[cases_df["carrier_category"] == category] if not cases_df.empty else pd.DataFrame()
        if section_cases.empty:
            continue
        entries.append((title, page))
        page += 1 + (2 * len(section_cases))
    return entries


def _draw_case_evidence_plot(fig: plt.Figure, case: pd.Series, *, left: float, bottom: float, width: float, height: float) -> None:
    metrics = [
        ("Call confidence", case.get("confidence_value")),
        ("Body interval confidence", case.get("min_interval_confidence")),
        ("Flank non-event confidence", case.get("min_flank_non_event_confidence")),
    ]
    qual_values = [(label, pd.to_numeric(value, errors="coerce")) for label, value in metrics]
    qual_values = [(label, float(value)) for label, value in qual_values if pd.notna(value)]

    overlap_metrics = [
        ("Interval coverage", case.get("interval_coverage")),
        ("Reciprocal overlap", case.get("reciprocal_overlap")),
    ]
    overlap_values = [(label, pd.to_numeric(value, errors="coerce")) for label, value in overlap_metrics]
    overlap_values = [(label, float(value)) for label, value in overlap_values if pd.notna(value)]

    if not qual_values and not overlap_values:
        fig.text(
            left, bottom + height - 0.02,
            "No numeric evidence fields available for plotting.",
            fontsize=8, color=_MUTED, ha="left", va="top", family="sans-serif",
        )
        return

    ax1 = fig.add_axes([left, bottom + height * 0.48, width, height * 0.44])
    if qual_values:
        labels = [label for label, _ in qual_values]
        values = [value for _, value in qual_values]
        ax1.barh(labels, values, color="#4C78A8")
        ax1.set_xlim(0, max(100.0, max(values) * 1.10))
        ax1.set_xlabel("QUAL/confidence", fontsize=7)
    else:
        ax1.text(0.0, 0.5, "No confidence metrics", fontsize=8, color=_MUTED)
        ax1.set_axis_off()
    ax1.tick_params(axis="both", labelsize=7)
    ax1.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    ax1.set_title("Call Evidence", fontsize=9, loc="left")
    for spine in ax1.spines.values():
        spine.set_visible(False)

    ax2 = fig.add_axes([left, bottom, width, height * 0.34])
    if overlap_values:
        labels = [label for label, _ in overlap_values]
        values = [value for _, value in overlap_values]
        ax2.barh(labels, values, color="#59A14F")
        ax2.set_xlim(0, 1.0)
        ax2.set_xlabel("Fraction", fontsize=7)
    else:
        ax2.text(0.0, 0.5, "No coverage/overlap metrics", fontsize=8, color=_MUTED)
        ax2.set_axis_off()
    ax2.tick_params(axis="both", labelsize=7)
    ax2.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    for spine in ax2.spines.values():
        spine.set_visible(False)


def _add_case_plot_unavailable_page(
    pdf: PdfPages,
    pdf_state: Dict[str, Any],
    case: pd.Series,
    *,
    section_number: int,
    section_title: str,
    reason: str,
) -> None:
    fig = _new_page(pdf_state, header="Section {} - {}".format(section_number, section_title))
    y = _section_band(fig, _BODY_TOP, "Per-sample Plot", eyebrow="Diagnostic")
    y = _draw_paragraph(
        fig,
        "The plot-style per-sample review page could not be rendered for this case.",
        start_y=y,
        fontsize=9,
    )
    _draw_kv_block(
        fig,
        [
            ("Sample", _format_pdf_value(case.get("sample"))),
            ("Batch", _format_pdf_value(case.get("batch_label"))),
            ("GD ID", _format_pdf_value(case.get("GD_ID"))),
            ("Locus", _format_pdf_value(case.get("cluster"))),
            ("Reason", reason),
        ],
        start_y=y - 0.010,
        columns=1,
        line_height=0.022,
        label_fraction=0.18,
    )
    _save_page(pdf, fig)


def _add_case_plot_page(
    pdf: PdfPages,
    pdf_state: Dict[str, Any],
    case: pd.Series,
    run: RunData,
    plot_context: PlotRunContext,
    *,
    section_number: int,
    section_title: str,
) -> None:
    if plot_context.unavailable_reason:
        _add_case_plot_unavailable_page(
            pdf,
            pdf_state,
            case,
            section_number=section_number,
            section_title=section_title,
            reason=plot_context.unavailable_reason,
        )
        return

    cluster = str(case.get("cluster"))
    sample_id = str(case.get("sample"))
    locus = plot_context.loci_by_cluster.get(cluster)
    region_df = plot_context.depth_by_cluster.get(cluster)
    if locus is None or region_df is None or region_df.empty:
        _add_case_plot_unavailable_page(
            pdf,
            pdf_state,
            case,
            section_number=section_number,
            section_title=section_title,
            reason="no locus-specific depth data were available for this cluster",
        )
        return
    if sample_id not in region_df.columns:
        _add_case_plot_unavailable_page(
            pdf,
            pdf_state,
            case,
            section_number=section_number,
            section_title=section_title,
            reason="the sample is not present in the inferred depth matrix for this locus",
        )
        return

    region_start = int(region_df["Start"].min())
    region_end = int(region_df["End"].max())
    xform = FlankCompressor(
        region_start,
        region_end,
        locus.start,
        locus.end,
        flank_scale=_PLOT_FLANK_SCALE,
    )
    cluster_calls_df = run.calls_df[run.calls_df["cluster"].astype(str) == cluster]
    confidence_column = _get_confidence_column(run.calls_df)

    pdf_state["page"] += 1
    rendered = _render_pdf_sample_page(
        pdf,
        sample_id,
        cluster,
        locus,
        region_df,
        cluster_calls_df,
        confidence_column,
        None,
        None,
        _PLOT_MIN_GENE_LABEL_SPACING,
        xform,
        None,
        plot_context.minor_baf_by_cluster.get(cluster),
        plot_context.baf_variance_by_cluster.get(cluster),
        plot_context.baf_sites_by_cluster.get(cluster),
        plot_context.event_del_by_cluster.get(cluster),
        plot_context.event_dup_by_cluster.get(cluster),
        None,
        baf_temperature_by_sample=plot_context.baf_temperature_by_sample,
        target_gd_id=str(case.get("GD_ID")),
        title_suffix="Aggregate case report",
    )
    if not rendered:
        pdf_state["page"] -= 1
        _add_case_plot_unavailable_page(
            pdf,
            pdf_state,
            case,
            section_number=section_number,
            section_title=section_title,
            reason="the plot renderer could not produce a page for this sample/locus combination",
        )


def _add_section_divider(
    pdf: PdfPages,
    pdf_state: Dict[str, Any],
    section_number: int,
    title: str,
    n_cases: int,
) -> None:
    fig = _new_page(pdf_state, header="Section {}".format(section_number))
    fig.text(0.5, 0.58, "Section {}".format(section_number), fontsize=11, color=_MUTED, ha="center")
    fig.text(0.5, 0.52, title, fontsize=24, fontweight="bold", color=_INK, ha="center")
    fig.add_artist(Line2D([0.30, 0.70], [0.495, 0.495], color=_INK, linewidth=0.8, transform=fig.transFigure))
    fig.text(
        0.5, 0.47,
        "{} case{} in this section".format(n_cases, "" if n_cases == 1 else "s"),
        fontsize=10, color=_MUTED, ha="center", va="top", style="italic",
    )
    _save_page(pdf, fig)


def _add_case_page(
    pdf: PdfPages,
    pdf_state: Dict[str, Any],
    case: pd.Series,
    *,
    section_number: int,
    section_title: str,
) -> None:
    fig = _new_page(pdf_state, header="Section {} - {}".format(section_number, section_title))
    y = _BODY_TOP
    fig.text(_MARGIN_L, y, "CASE", fontsize=7, color=_MUTED, ha="left", va="top", family="sans-serif")
    y -= 0.014
    fig.text(
        _MARGIN_L, y, _format_pdf_value(case.get("sample")),
        fontsize=16, fontweight="bold", color=_INK, ha="left", va="top",
        family="sans-serif",
    )
    fig.text(
        _MARGIN_R, y, _humanize_category(case.get("carrier_category")),
        fontsize=10, color=_MUTED, ha="right", va="top", style="italic",
        family="sans-serif",
    )
    y -= 0.024
    fig.add_artist(Line2D([_MARGIN_L, _MARGIN_R], [y, y], color=_INK, linewidth=0.8, transform=fig.transFigure))
    y -= 0.016

    y = _draw_kv_block(
        fig,
        [
            ("Batch", _format_pdf_value(case.get("batch_label"))),
            ("Sample key", _format_pdf_value(case.get("sample_key"))),
            ("GD ID", _format_pdf_value(case.get("GD_ID"))),
            ("Locus", _format_pdf_value(case.get("cluster"))),
            ("Event", "{} {}:{}-{}".format(
                _format_pdf_value(case.get("svtype")),
                _format_pdf_value(case.get("chrom")),
                _format_pdf_value(case.get("start")),
                _format_pdf_value(case.get("end")),
            )),
            ("Breakpoints", "{}-{}".format(_format_pdf_value(case.get("BP1")), _format_pdf_value(case.get("BP2")))),
        ],
        start_y=y,
        columns=1,
        line_height=0.020,
        label_fraction=0.28,
    )

    y = _section_band(fig, y - 0.010, "Call Metrics", eyebrow="Findings")
    y = _draw_kv_block(
        fig,
        [
            ("Confidence", _format_pdf_value(case.get("confidence_value"))),
            ("Confidence column", _format_pdf_value(case.get("confidence_column"))),
            ("Calling method", _format_pdf_value(case.get("calling_method"))),
            ("Sample ploidy", _format_pdf_value(case.get("sample_ploidy"))),
            ("Mean depth", _format_pdf_value(case.get("mean_depth"))),
            ("N bins", _format_pdf_value(case.get("n_bins"))),
            ("Interval coverage", _format_pdf_value(case.get("interval_coverage"))),
            ("Reciprocal overlap", _format_pdf_value(case.get("reciprocal_overlap"))),
        ],
        start_y=y,
        columns=1,
        line_height=0.020,
        label_fraction=0.28,
    )

    evidence_df = pd.DataFrame([
        {
            "metric": "Body interval confidence",
            "value": case.get("min_interval_confidence"),
        },
        {
            "metric": "Left flank non-event median",
            "value": case.get("left_flank_non_event_median"),
        },
        {
            "metric": "Right flank non-event median",
            "value": case.get("right_flank_non_event_median"),
        },
        {
            "metric": "Flank non-event confidence",
            "value": case.get("min_flank_non_event_confidence"),
        },
        {
            "metric": "Matched segment",
            "value": "{}-{}".format(_format_pdf_value(case.get("matched_seg_start")), _format_pdf_value(case.get("matched_seg_end"))),
        },
        {
            "metric": "Matched haplotype / CN",
            "value": "{} / {}".format(_format_pdf_value(case.get("matched_haplotype")), _format_pdf_value(case.get("hap_cn_state"))),
        },
    ])
    y = _section_band(fig, y - 0.010, "Breakpoint Evidence", eyebrow="Evidence")
    y = _draw_table(
        fig,
        evidence_df,
        ["metric", "value"],
        start_y=y,
        max_rows=8,
        col_widths=[0.48, 0.52],
        column_labels={"metric": "Metric", "value": "Value"},
        font_size=7,
        row_height=0.024,
    )

    _save_page(pdf, fig)


def _add_case_pages(
    pdf: PdfPages,
    pdf_state: Dict[str, Any],
    cases_df: pd.DataFrame,
    runs: List[RunData],
) -> None:
    runs_by_batch_id = {run.batch_id: run for run in runs}
    plot_context_by_batch_id: Dict[int, PlotRunContext] = {}
    section_number = 4
    for category, title in _CASE_SECTIONS:
        section_cases = cases_df[cases_df["carrier_category"] == category] if not cases_df.empty else pd.DataFrame()
        if section_cases.empty:
            continue
        _add_section_divider(pdf, pdf_state, section_number, title, len(section_cases))
        for _, case in section_cases.sort_values(["batch_label", "cluster", "GD_ID", "sample"]).iterrows():
            _add_case_page(pdf, pdf_state, case, section_number=section_number, section_title=title)
            batch_id = int(case.get("batch_id"))
            run = runs_by_batch_id.get(batch_id)
            if run is None:
                _add_case_plot_unavailable_page(
                    pdf,
                    pdf_state,
                    case,
                    section_number=section_number,
                    section_title=title,
                    reason="the aggregate run metadata for this batch were unavailable",
                )
                continue
            if batch_id not in plot_context_by_batch_id:
                plot_context_by_batch_id[batch_id] = _load_plot_run_context(run)
            _add_case_plot_page(
                pdf,
                pdf_state,
                case,
                run,
                plot_context_by_batch_id[batch_id],
                section_number=section_number,
                section_title=title,
            )
        section_number += 1


def _add_eval_pages(pdf: PdfPages, pdf_state: Dict[str, Any], eval_df: pd.DataFrame) -> None:
    _draw_table_section_pages(
        pdf,
        pdf_state,
        eval_df,
        ["batch_label", "GD_ID", "TP", "FP", "FN", "sensitivity", "precision"],
        header="Evaluation Summary",
        title="Evaluation Summary",
        eyebrow="Evaluation",
        max_rows=18,
        col_widths=[0.14, 0.28, 0.08, 0.08, 0.08, 0.17, 0.17],
        column_labels={"batch_label": "Batch", "GD_ID": "GD ID"},
    )


def _add_missing_pages(pdf: PdfPages, pdf_state: Dict[str, Any], missing_df: pd.DataFrame) -> None:
    _draw_table_section_pages(
        pdf,
        pdf_state,
        missing_df,
        ["batch_label", "artifact", "reason", "path"],
        header="Appendix - Missing Optional Artifacts",
        title="Missing Optional Artifacts",
        eyebrow="Appendix",
        max_rows=18,
        col_widths=[0.18, 0.24, 0.20, 0.38],
        column_labels={"batch_label": "Batch", "artifact": "Artifact", "reason": "Reason", "path": "Path"},
    )


def _field_guide_table() -> pd.DataFrame:
    rows = [
        ("Cover page", "Summary", "Front-matter summary of cohort size, modeled loci, carrier events, and evaluation availability."),
        ("Cover page", "Calling mode(s)", "Calling mode and call-stage selection criteria read from the call outputs for the aggregated runs."),
        ("Cover page", "Aggregate non-confident min confidence", "Lower score bound for showing non-confident best-match calls in the aggregate outputs; confident calls are unaffected."),
        ("Batch Inventory", "Samples", "Number of distinct samples in each input work directory ploidy table."),
        ("Cohort Summary", "Metric", "Cohort-wide aggregate count or scalar metric."),
        ("Case Index", "Category", "Confident when emitted by the call step as a carrier; non-confident when selected as the best-match candidate but not emitted as a carrier and meeting the aggregate non-confident minimum confidence."),
        ("Case Index", "Left flank / Right flank", "PASS when the flank non-event median meets the call-time flank threshold, or when no flank value was available; otherwise FAIL."),
        ("Locus Burden", "Carrier samples", "Number of unique carrier samples for one batch/GD_ID/locus/type group."),
        ("Case detail", "Call Metrics", "Per-call identifiers, confidence, method, ploidy, depth, interval coverage, and reciprocal overlap."),
        ("Case detail", "Breakpoint Evidence", "Detailed evidence fields emitted by the call step for the selected GD breakpoint match."),
        ("Case detail", "Per-sample plot", "Full review page reused from the plot tool, showing the sample's locus annotations, depth profile, BAF panel, and event-marginal panel for the selected call."),
        ("Evaluation Summary", "TP/FP/FN", "Optional truth-set comparison fields computed by eval from the confident carriers emitted by the call step."),
    ]
    return pd.DataFrame(rows, columns=["display_element", "displayed_label", "definition"])


def _case_section_page_count(cases_df: pd.DataFrame) -> int:
    if cases_df.empty:
        return 0
    total = 0
    for category, _title in _CASE_SECTIONS:
        section_cases = cases_df[cases_df["carrier_category"] == category]
        if section_cases.empty:
            continue
        total += 1 + (2 * len(section_cases))
    return total


def _add_field_guide_pages(pdf: PdfPages, pdf_state: Dict[str, Any]) -> None:
    _draw_table_section_pages(
        pdf,
        pdf_state,
        _field_guide_table(),
        ["display_element", "displayed_label", "definition"],
        header="Appendix - Report Field Guide",
        title="Report Field Guide",
        eyebrow="Appendix",
        max_rows=16,
        col_widths=[0.24, 0.22, 0.54],
        column_labels={
            "display_element": "Table or row group",
            "displayed_label": "Displayed row/column",
            "definition": "Definition",
        },
    )


def _build_toc_entries(
    inventory_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    locus_summary_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> List[Tuple[str, int]]:
    def _page_count(n_rows: int, max_rows: int) -> int:
        return max(1, int(math.ceil(float(n_rows) / float(max_rows)))) if n_rows else 1

    inventory_start = 2
    inventory_pages = _page_count(len(inventory_df), 18)
    cohort_start = inventory_start + inventory_pages
    cohort_pages = _page_count(len(summary_df), 28)
    case_index_start = cohort_start + cohort_pages
    case_index_pages = _page_count(len(cases_df), 16)
    locus_start = case_index_start + case_index_pages
    locus_pages = _page_count(len(locus_summary_df), 22)

    entries: List[Tuple[str, int]] = [
        ("Batch Inventory", inventory_start),
        ("Cohort Summary", cohort_start),
        ("Case Index", case_index_start),
        ("Locus Burden", locus_start),
    ]
    case_entries = _case_section_start_pages(cases_df, locus_start + locus_pages)
    entries.extend(case_entries)
    page = locus_start + locus_pages + _case_section_page_count(cases_df)
    eval_pages = _page_count(len(eval_df), 18)
    entries.extend([
        ("Evaluation Summary", page),
        ("Report Field Guide", page + eval_pages),
    ])
    return entries


def _write_pdf_report(
    report_path: Path,
    runs: List[RunData],
    summary_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    locus_summary_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> None:
    rc_overrides = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "axes.edgecolor": _INK,
        "text.color": _INK,
        "axes.labelcolor": _INK,
        "xtick.color": _INK,
        "ytick.color": _INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    pdf_state: Dict[str, Any] = {
        "page": 0,
        "footer_left": "Generated {} - {} batch{}".format(
            datetime.now().strftime("%Y-%m-%d"),
            len(runs),
            "" if len(runs) == 1 else "es",
        ),
    }
    with plt.rc_context(rc_overrides), PdfPages(report_path) as pdf:
        link_specs = _add_cover_page(
            pdf,
            pdf_state,
            runs,
            summary_df,
            _build_toc_entries(
                inventory_df,
                summary_df,
                cases_df,
                locus_summary_df,
                eval_df,
                missing_df,
            ),
        )
        _add_inventory_pages(pdf, pdf_state, inventory_df)
        _add_summary_pages(pdf, pdf_state, summary_df, cases_df, locus_summary_df)
        _add_case_pages(pdf, pdf_state, cases_df, runs)
        _add_eval_pages(pdf, pdf_state, eval_df)
        _add_field_guide_pages(pdf, pdf_state)
        _add_pdf_internal_links(pdf, 1, link_specs)


def _run_aggregate(args: argparse.Namespace) -> None:
    """Run aggregation after CLI logging is configured."""
    _validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / args.output_name

    runs = _load_runs(args)
    (
        summary_df,
        inventory_df,
        calls_df,
        cases_df,
        locus_summary_df,
        eval_df,
        missing_df,
    ) = _build_report_tables(runs, min_confidence=args.min_confidence)
    sidecar_paths = _write_sidecars(
        output_dir,
        summary_df,
        inventory_df,
        calls_df,
        cases_df,
        locus_summary_df,
        eval_df,
        missing_df,
    )
    _write_pdf_report(
        report_path,
        runs,
        summary_df,
        inventory_df,
        cases_df,
        locus_summary_df,
        eval_df,
        missing_df,
    )

    logger = get_logger("aggregate")
    logger.info(
        "Aggregate report complete: batches=%d calls=%d cases=%d loci=%d eval_rows=%d missing_optional_artifacts=%d",
        len(runs),
        len(calls_df),
        len(cases_df),
        len(locus_summary_df),
        len(eval_df),
        len(missing_df),
    )
    logger.info("Aggregate output artifact count: %d", len(sidecar_paths) + 1)


def main() -> None:
    """Entry point for ``gatk-sv-gd aggregate``."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(
        args.output_dir,
        filename="aggregate_log.txt",
        command="aggregate",
        args=args,
    )
    _run_aggregate(args)


if __name__ == "__main__":
    main()