import numpy as np
import pandas as pd
import pytest

from gatk_sv_gd import nominate_mosaic as nm
from gatk_sv_gd import plot_mosaic as pm

BIN = 10_000
GENOME = 3_000_000
LAM = 400
# Cluster c1: BP1 (1.40-1.41M), BP2 (2.00-2.01M), BP3 (2.40-2.41M).
# GD_A = BP1-BP2 (DEL+DUP rows), GD_B = BP2-BP3 (DEL only).
A = (1_400_000, 2_010_000)
B = (2_000_000, 2_410_000)


def _table(path):
    rows = [
        ("chr1", A[0], A[1], "A_DEL", "DEL", "yes", "no", "c1", "1", "2"),
        ("chr1", A[0], A[1], "A_DUP", "DUP", "yes", "no", "c1", "1", "2"),
        ("chr1", B[0], B[1], "B_DEL", "DEL", "yes", "no", "c1", "2", "3"),
        ("chrX", 1_000_000, 1_400_000, "X_DEL", "DEL", "yes", "no", "", "1", "2"),
    ]
    pd.DataFrame(rows, columns=[
        "chr", "start_GRCh38", "end_GRCh38", "GD_ID", "svtype", "NAHR", "terminal",
        "cluster", "BP1", "BP2"]).to_csv(path, sep="\t", index=False)


def _counts(path):
    rng = np.random.RandomState(3)
    starts = np.arange(0, GENOME, BIN)
    mid = starts + BIN / 2
    in_a = (mid >= A[0] + 10_000) & (mid < A[1] - 10_000)
    left_flank = (mid >= A[0] - 600_000) & (mid < A[0])
    profiles = {f"n{i}": np.ones(len(starts)) for i in range(25)}
    profiles["mos_del"] = np.where(in_a, 0.75, 1.0)
    profiles["mos_dup"] = np.where(in_a, 1.25, 1.0)
    first_60 = in_a & (mid < A[0] + 0.6 * (A[1] - A[0]))
    profiles["part_del"] = np.where(first_60, 0.75, 1.0)
    profiles["het_del"] = np.where(in_a, 0.5, 1.0)
    # mosaic-range interval depth, but a large event in the left flank
    profiles["mos_del_bigflank"] = np.where(in_a, 0.75, 1.0) * np.where(left_flank, 0.5, 1.0)
    cols = {s: rng.poisson(LAM * p) for s, p in profiles.items()}
    df = pd.DataFrame({"#Chr": "chr1", "Start": starts, "End": starts + BIN, **cols})
    df.to_csv(path, sep="\t", index=False, compression="gzip")


def _segdup_bed(path):
    pd.DataFrame([("chr1", 100_000, 120_000)]).to_csv(path, sep="\t", header=False, index=False)


def _run(tmp_path, extra=()):
    gd, counts, sd = tmp_path / "gd.tsv", tmp_path / "c.tsv.gz", tmp_path / "sd.bed"
    _table(gd), _counts(counts), _segdup_bed(sd)
    nm.main(["nominate", "--batch", "b1", "--gd-table", str(gd), "--low-res-counts",
             str(counts), "--segdup-bed", str(sd), "--out-prefix", str(tmp_path / "o"),
             *extra])
    cand = pd.read_csv(tmp_path / "o.candidates.tsv.gz", sep="\t")
    summ = pd.read_csv(tmp_path / "o.interval_summary.tsv.gz", sep="\t")
    return cand, summ


def test_nominates_only_mosaic_range_normal_flank_samples(tmp_path):
    cand, summ = _run(tmp_path)
    a = cand[cand["start"] == A[0]].set_index("sample")
    assert set(a.index) == {"mos_del", "mos_dup"}
    assert a.loc["mos_del", "gd_id"] == "A_DEL" and a.loc["mos_del", "mosaic_type"] == "DEL"
    assert a.loc["mos_dup", "gd_id"] == "A_DUP" and a.loc["mos_dup", "mosaic_type"] == "DUP"
    assert 1.2 <= a.loc["mos_del", "median_depth"] <= 1.8
    assert 2.2 <= a.loc["mos_dup", "median_depth"] <= 2.8
    assert (cand["batch"] == "b1").all()
    # het DEL (~1.0) is outside both windows; flank-abnormal sample is excluded
    assert "het_del" not in set(cand["sample"])
    assert "mos_del_bigflank" not in set(cand["sample"])
    row = summ[summ["start"] == A[0]].iloc[0]
    assert row["n_abnormal"] >= 1 and row["n_candidates"] == 2
    # one summary row per distinct autosomal interval; chrX skipped
    assert len(summ) == 2 and set(summ["chr"]) == {"chr1"}


def test_median_ranges_are_configurable(tmp_path):
    cand, _ = _run(tmp_path, ["--del-min", "0.9", "--del-max", "1.1"])
    assert "het_del" in set(cand["sample"])


def test_interval_median_excludes_segdup_bins():
    starts = np.arange(0, 100_000, 10_000)
    depth = pd.DataFrame({"Chr": "chr1", "Start": starts, "End": starts + 10_000,
                          "s": [1.0] * 5 + [2.0] * 5})
    sd = {"chr1": (np.array([50_000]), np.array([100_000]))}
    med, n = nm.interval_medians(depth, "chr1", 0, 100_000, sd)
    assert n == 5 and med["s"] == 1.0
    med, n = nm.interval_medians(depth, "chr1", 0, 100_000, None)
    assert n == 10 and med["s"] == 1.5


def test_min_bins_suppresses_short_intervals(tmp_path):
    cand, _ = _run(tmp_path, ["--min-bins", "10000"])
    assert len(cand) == 0


def test_merge_streams_and_handles_empty(tmp_path):
    cand, _ = _run(tmp_path)
    p = tmp_path / "o.candidates.tsv.gz"
    (tmp_path / "list.txt").write_text(f"{p}\n{p}\n")
    nm.main(["merge", "--input-list", str(tmp_path / "list.txt"),
             "--out", str(tmp_path / "m.tsv.gz")])
    merged = pd.read_csv(tmp_path / "m.tsv.gz", sep="\t")
    assert len(merged) == 2 * len(cand) and list(merged.columns) == nm.CANDIDATE_COLUMNS
    (tmp_path / "empty.txt").write_text("")
    nm.main(["merge", "--input-list", str(tmp_path / "empty.txt"),
             "--out", str(tmp_path / "e.tsv.gz")])
    assert list(pd.read_csv(tmp_path / "e.tsv.gz", sep="\t").columns) == nm.CANDIDATE_COLUMNS


def test_chrom_factors_put_chromosome_median_at_cn2():
    counts = pd.DataFrame({
        "Chr": ["chr1"] * 4 + ["chr2"] * 4,
        "Start": list(range(4)) * 2, "End": list(range(1, 5)) * 2,
        "s": [100, 100, 100, 100, 150, 150, 150, 150],   # chr2 shifted 1.5x
        "z": [100] * 4 + [0] * 4,                          # no chr2 signal
    })
    med = {"s": 125.0, "z": 50.0}
    f = nm.chrom_scale_factors(counts, med, ["s", "z"])
    assert f["chr1"]["s"] == pytest.approx(1.25) and f["chr2"]["s"] == pytest.approx(125 / 150)
    assert f["chr2"]["z"] == 1.0                           # zero median -> unchanged


def test_chromosome_wide_shift_is_normalized_out(tmp_path):
    # chr1 of one sample sits at 1.3x of the batch everywhere (whole-chromosome
    # gain relative to its other chromosomes); its interval looks CN 2 once
    # normalized to the chromosome, but "mosaic" without chrom normalization.
    gd, counts, sd = tmp_path / "gd.tsv", tmp_path / "c.tsv.gz", tmp_path / "sd.bed"
    _table(gd), _segdup_bed(sd)
    rng = np.random.RandomState(5)
    starts = np.arange(0, GENOME, BIN)
    frames = []
    for chrom in ("chr1", "chr2", "chr3"):
        cols = {f"n{i}": rng.poisson(LAM, len(starts)) for i in range(25)}
        cols["shifted"] = rng.poisson(LAM * (1.3 if chrom == "chr1" else 1.0), len(starts))
        frames.append(pd.DataFrame({"#Chr": chrom, "Start": starts, "End": starts + BIN, **cols}))
    pd.concat(frames).to_csv(counts, sep="\t", index=False, compression="gzip")
    args = ["nominate", "--batch", "b", "--gd-table", str(gd), "--low-res-counts", str(counts),
            "--segdup-bed", str(sd), "--out-prefix", str(tmp_path / "o"),
            "--max-deviation", "100"]   # isolate normalization from the abnormal-flank filter
    nm.main(args)
    assert "shifted" not in set(pd.read_csv(tmp_path / "o.candidates.tsv.gz", sep="\t")["sample"])
    nm.main(args + ["--no-chrom-normalize"])
    raw = pd.read_csv(tmp_path / "o.candidates.tsv.gz", sep="\t")
    assert "shifted" in set(raw["sample"])


def test_consistent_fraction_rejects_partial_and_spikes():
    full = np.full(10, 1.5)
    part = np.r_[np.full(3, 1.5), np.full(7, 2.0)]
    up = np.full(10, 2.5)
    vals = np.column_stack([full, part, up])
    f = nm.consistent_fraction(vals, np.array(["DEL", "DEL", "DUP"]), 0.1)
    assert list(f) == [1.0, 0.3, 1.0]


def test_partial_interval_event_needs_consistency(tmp_path):
    # 0.75x over 60% of GD_A: median is in the DEL window, but only ~60% of bins
    # are on the event side, below the 0.7 default
    cand, _ = _run(tmp_path)
    assert "part_del" not in set(cand["sample"])
    cand, _ = _run(tmp_path, ["--min-consistent-fraction", "0.4"])
    assert "part_del" in set(cand["sample"])


def test_collapse_keeps_best_interval_per_cluster():
    rows = [("A_DEL", "s", "b", 1.5, "DEL", "chr1", 1, 2, "DEL", 30, "c1", 1.5, 0.9, 1),
            ("B_DEL", "s", "b", 1.5, "DEL", "chr1", 2, 3, "DEL", 40, "c1", 1.5, 0.8, 1),
            ("C_DEL", "s", "b", 1.5, "DEL", "chr1", 5, 6, "DEL", 40, "c2", 1.5, 0.8, 1)]
    out = nm.collapse_by_cluster(pd.DataFrame(rows, columns=nm.CANDIDATE_COLUMNS))
    assert list(out["gd_id"]) == ["A_DEL", "C_DEL"]
    assert list(out["n_cluster_intervals"]) == [2, 1]


def test_end_to_end_adds_columns_and_collapses(tmp_path):
    cand, summ = _run(tmp_path)
    assert list(cand.columns) == nm.CANDIDATE_COLUMNS
    assert not cand.duplicated(["sample", "cluster"]).any()
    assert (cand["consistent_fraction"] >= 0.7).all()
    assert "batch_median_depth" in summ.columns
