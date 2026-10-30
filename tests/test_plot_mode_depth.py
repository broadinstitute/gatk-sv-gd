import numpy as np
import pandas as pd
import pytest

from gatk_sv_gd import plot_mode_depth as pmd

BIN = 10_000
N_BINS = 40  # locus body bins


def _write_gd_table(path):
    rows = [
        ("chr1", 1_000_000, 1_400_000, "L1_DEL", "DEL", "yes", "no", "", "1", "2"),
        ("chr1", 1_000_000, 1_400_000, "L1_DUP", "DUP", "yes", "no", "", "1", "2"),
        ("chr2", 2_000_000, 2_400_000, "L2_DEL", "DEL", "yes", "no", "", "1", "2"),
        ("chr3", 3_000_000, 3_400_000, "L3_DEL", "DEL", "no", "no", "", "1", "2"),
        ("chrX", 4_000_000, 4_400_000, "LX_DEL", "DEL", "yes", "no", "", "1", "2"),
        ("chr4", 5_000_000, 5_400_000, "L4_DUP", "DUP", "yes", "no", "", "1", "2"),
    ]
    pd.DataFrame(rows, columns=[
        "chr", "start_GRCh38", "end_GRCh38", "GD_ID", "svtype", "NAHR",
        "terminal", "cluster", "BP1", "BP2"]).to_csv(path, sep="\t", index=False)


def _locus_key(chrom, start, end):
    return f"{chrom}:{start}-{end}"


# sample category -> function(rng) giving per-bin true depth at the locus body
def _profiles(rng):
    noise = lambda: rng.normal(0, 0.04, N_BINS)
    half = np.r_[np.ones(N_BINS // 2), np.zeros(N_BINS - N_BINS // 2)].astype(bool)
    # Partial events cover 70% of bins so one state clearly dominates.
    mostly = np.r_[np.ones(28), np.zeros(12)].astype(bool)
    few = np.r_[np.ones(12), np.zeros(28)].astype(bool)
    base = 2 + noise()
    return {
        "noncarrier": base,
        "het_del": 1 + noise(),
        "het_dup": 3 + noise(),
        "mosaic_del": 1.5 + noise(),
        "mosaic_dup": 2.5 + noise(),
        "partial_del_major": np.where(mostly, 1.0, 2.0) + noise(),
        "partial_del_minor": np.where(few, 1.0, 2.0) + noise(),
        "partial_dup_major": np.where(mostly, 3.0, 2.0) + noise(),
        "partial_dup_minor": np.where(few, 3.0, 2.0) + noise(),
    }


CATEGORIES = list(_profiles(np.random.RandomState(0)))


def make_posteriors(path, per_cat=15, seed=1):
    rng = np.random.RandomState(seed)
    rows = []
    cluster_keys = [_locus_key("chr1", 1_000_000, 1_400_000),
                    _locus_key("chr2", 2_000_000, 2_400_000),
                    _locus_key("chr3", 3_000_000, 3_400_000),
                    _locus_key("chrX", 4_000_000, 4_400_000),
                    _locus_key("chr4", 5_000_000, 5_400_000)]
    for cat in CATEGORIES:
        for i in range(per_cat):
            sample = f"{cat}_{i}"
            for cl in cluster_keys:
                prof = _profiles(rng)[cat] if cl == cluster_keys[0] else 2 + rng.normal(0, .04, N_BINS)
                for b, d in enumerate(prof):
                    rows.append((cl, "body", "chr1", b * BIN, (b + 1) * BIN, sample, d))
                # flank bins carry a misleading value; they must be ignored
                for b in range(10):
                    rows.append((cl, "left_flank", "chr1", -b, -b + BIN, sample, 0.3))
    df = pd.DataFrame(rows, columns=["cluster", "interval", "chr", "start", "end",
                                     "sample", "depth"])
    df.to_csv(path, sep="\t", index=False, compression="gzip")


@pytest.fixture
def setup(tmp_path):
    gd = tmp_path / "gd.tsv"
    _write_gd_table(gd)
    post = tmp_path / "post.tsv.gz"
    make_posteriors(post)
    return gd, post, tmp_path


def _summarize(gd, post, out, batch="b1"):
    pmd.main(["summarize", "--batch", batch, "--gd-table", str(gd),
              "--cn-posteriors", str(post), "--out", str(out)])
    return pd.read_csv(out, sep="\t")


def test_locus_order_merges_del_dup_skips_non_nahr_keeps_sex(tmp_path):
    gd = tmp_path / "gd.tsv"
    _write_gd_table(gd)
    order = pmd.locus_order(str(gd))
    assert list(order["label"]) == ["L1", "L2", "LX", "L4"]
    assert list(order["x"]) == [0, 1, 2, 3]


def test_mode_of_rounded_tie_goes_nearest_two():
    s = pd.Series({1.0: 5, 2.0: 5})
    assert pmd.mode_of_rounded(s) == 2.0
    assert pmd.mode_of_rounded(pd.Series({3.0: 4, 2.0: 4, 1.0: 1})) == 2.0


def test_category_mode_depths(setup):
    gd, post, tmp = setup
    out = _summarize(gd, post, tmp / "b1.tsv.gz")
    key = _locus_key("chr1", 1_000_000, 1_400_000)
    sub = out[out["cluster"] == key].copy()
    sub["cat"] = sub["sample"].str.rsplit("_", n=1).str[0]
    grp = sub.groupby("cat")["median_depth"]

    assert grp.apply(lambda v: (abs(v - 2) <= 0.2).all())["noncarrier"]
    assert (abs(grp.get_group("het_del") - 1) <= 0.2).all()
    assert (abs(grp.get_group("het_dup") - 3) <= 0.2).all()
    assert grp.get_group("mosaic_del").between(1.2, 1.8).all()
    assert grp.get_group("mosaic_dup").between(2.2, 2.8).all()
    # partial overlap: near 1 or near 2 (3), never intermediate
    for cat, target in [("partial_del_major", 1), ("partial_del_minor", 2),
                        ("partial_dup_major", 3), ("partial_dup_minor", 2)]:
        assert (abs(grp.get_group(cat) - target) <= 0.2).all(), cat
    # flank bins (depth 0.3) are ignored
    assert out["median_depth"].min() > 0.5


def test_non_nahr_excluded_sex_included(setup):
    gd, post, tmp = setup
    out = _summarize(gd, post, tmp / "b1.tsv.gz")
    assert _locus_key("chr3", 3_000_000, 3_400_000) not in set(out["cluster"])
    assert _locus_key("chrX", 4_000_000, 4_400_000) in set(out["cluster"])


def test_mosaic_flags_and_loci(setup):
    gd, post, tmp = setup
    out = _summarize(gd, post, tmp / "b1.tsv.gz")
    flagged = pmd.probable_mosaic_loci([out], min_samples=1)
    assert flagged == [_locus_key("chr1", 1_000_000, 1_400_000)]
    assert pmd.probable_mosaic_loci([out], min_samples=100) == []
    only_clean = out[~out["sample"].str.startswith("mosaic")]
    assert pmd.probable_mosaic_loci([only_clean], min_samples=1) == []


def test_merge_and_plot_single_and_stacked(setup):
    gd, post, tmp = setup
    a = _summarize(gd, post, tmp / "b1.tsv.gz", "b1")
    _summarize(gd, post, tmp / "b2.tsv.gz", "b2")
    merged = tmp / "merged.tsv.gz"
    pmd.main(["merge", "--inputs", str(tmp / "b1.tsv.gz"), str(tmp / "b2.tsv.gz"),
              "--out", str(merged)])
    m = pd.read_csv(merged, sep="\t")
    assert len(m) == 2 * len(a) and set(m["batch"]) == {"b1", "b2"}

    pdf1 = tmp / "one.pdf"
    pmd.main(["plot", "--gd-table", str(gd), "--data", str(merged),
              "--out-pdf", str(pdf1)])
    assert pdf1.read_bytes().startswith(b"%PDF")

    pdf2 = tmp / "two.pdf"
    pmd.main(["plot", "--gd-table", str(gd), "--data", str(merged), str(tmp / "b1.tsv.gz"),
              "--labels", "All", "Batch 1", "--out-pdf", str(pdf2)])
    assert pdf2.read_bytes().startswith(b"%PDF")


def test_plot_labels_only_mosaic_loci(setup):
    gd, post, tmp = setup
    out = _summarize(gd, post, tmp / "b1.tsv.gz")
    order = pmd.locus_order(str(gd))
    shown = pmd.draw_mode_depth_plot([out], ["x"], order, str(tmp / "p.pdf"))
    assert shown == ["L1"]


def test_plot_label_mismatch_errors(setup):
    gd, post, tmp = setup
    _summarize(gd, post, tmp / "b1.tsv.gz")
    with pytest.raises(SystemExit):
        pmd.main(["plot", "--gd-table", str(gd), "--data", str(tmp / "b1.tsv.gz"),
                  "--labels", "a", "b", "--out-pdf", str(tmp / "x.pdf")])


def test_locus_label_strips_suffix():
    assert pmd.locus_label("15q11.2_DEL") == "15q11.2"
    assert pmd.locus_label("1q21.1_distal_DUP") == "1q21.1_distal"
    assert pmd.locus_label("plain") == "plain"


def test_median_of_actual_values_in_mode_bin():
    rows = [("c", "body", "chr1", 0, 1, "s", d)
            for d in [1.52, 1.54, 1.56, 1.46, 2.04]]
    chunk = pd.DataFrame(rows, columns=["cluster", "interval", "chr", "start",
                                        "end", "sample", "depth"])
    out = pmd.summarize_depth([chunk], ["c"], "b")
    row = out.iloc[0]
    assert row["mode_depth"] == 1.5          # 1.52, 1.54 -> 1.5 ; 1.56 -> 1.6; 1.46 -> 1.5
    assert row["median_depth"] == pytest.approx(1.52)  # median(1.46, 1.52, 1.54)
    assert row["mode_bins"] == 3 and row["n_bins"] == 5


def test_ymax_capped_at_observed_max(setup):
    gd, post, tmp = setup
    out = _summarize(gd, post, tmp / "b1.tsv.gz")
    out = out[out["median_depth"] < 2.5]
    order = pmd.locus_order(str(gd))
    pmd.draw_mode_depth_plot([out], ["x"], order, str(tmp / "p.pdf"))
    assert (tmp / "p.pdf").stat().st_size > 0
