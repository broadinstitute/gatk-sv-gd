import numpy as np
import pandas as pd
import pytest

from gatk_sv_gd import plot_mode_depth as pmd

BIN = 10_000


def _locus_key(chrom, start, end):
    return f"{chrom}:{start}-{end}"


BIN = 10_000
GENOME_END = 3_000_000
# L1 (chr1): BP1 ends 1,010,000; BP2 starts 1,400,000 -> 39 body bins.
BODY = (1_010_000, 1_400_000)
L1 = _locus_key("chr1", 1_000_000, 1_400_000)
LAMBDA = 400


def _categories():
    # (name, true body depth profile as fraction of diploid per body bin)
    n = (BODY[1] - BODY[0]) // BIN
    mostly = np.r_[np.ones(28), np.zeros(n - 28)].astype(bool)
    few = np.r_[np.ones(12), np.zeros(n - 12)].astype(bool)
    f = lambda v: np.full(n, v)
    return {
        "noncarrier": f(1.0),
        "het_del": f(0.5),
        "het_dup": f(1.5),
        "mosaic_del": f(0.75),
        "mosaic_del2": f(0.8),
        "mosaic_dup": f(1.25),
        "partial_del_major": np.where(mostly, 0.5, 1.0),
        "partial_del_minor": np.where(few, 0.5, 1.0),
        "partial_dup_major": np.where(mostly, 1.5, 1.0),
        "partial_dup_minor": np.where(few, 1.5, 1.0),
    }


CATEGORIES = list(_categories())


def make_counts(path, per_cat=12, seed=1):
    """Low-res count matrix: Poisson counts, per-sample library-size factors,
    a CNV in the L1 body per category, plus chr2 and chrX bins."""
    rng = np.random.RandomState(seed)
    starts = np.arange(0, GENOME_END, BIN)
    cats = _categories()
    cols = {}
    body_idx = np.where((starts + BIN / 2 >= BODY[0]) & (starts + BIN / 2 < BODY[1]))[0]
    for cat, prof in cats.items():
        for i in range(per_cat):
            lib = rng.uniform(0.6, 1.6)  # library-size differences to normalize away
            per_chrom = []
            for chrom in ("chr1", "chr2", "chrX"):
                lam = np.full(len(starts), LAMBDA * lib)
                if chrom == "chr1":
                    lam[body_idx] *= prof
                per_chrom.append(rng.poisson(lam))
            cols[f"{cat}_{i}"] = np.concatenate(per_chrom)
    chroms = np.repeat(["chr1", "chr2", "chrX"], len(starts))
    df = pd.DataFrame({"#Chr": chroms, "Start": np.tile(starts, 3),
                       "End": np.tile(starts + BIN, 3), **cols})
    df.to_csv(path, sep="\t", index=False, compression="gzip")


def _gd_table(path):
    rows = [
        ("chr1", 1_000_000, 1_400_000, "L1_DEL", "DEL", "yes", "no", "", "1", "2"),
        ("chr1", 1_000_000, 1_400_000, "L1_DUP", "DUP", "yes", "no", "", "1", "2"),
        ("chr2", 2_000_000, 2_400_000, "L2_DEL", "DEL", "yes", "no", "", "1", "2"),
        ("chr1", 2_500_000, 2_900_000, "L3_DEL", "DEL", "no", "no", "", "1", "2"),
        ("chrX", 1_000_000, 1_400_000, "LX_DEL", "DEL", "yes", "no", "", "1", "2"),
    ]
    pd.DataFrame(rows, columns=[
        "chr", "start_GRCh38", "end_GRCh38", "GD_ID", "svtype", "NAHR", "terminal",
        "cluster", "BP1", "BP2"]).to_csv(path, sep="\t", index=False)


@pytest.fixture
def setup(tmp_path):
    gd = tmp_path / "gd.tsv"
    _gd_table(gd)
    counts = tmp_path / "counts.tsv.gz"
    make_counts(counts)
    return gd, counts, tmp_path


def _summarize(gd, counts, out, batch="b1"):
    pmd.main(["summarize", "--batch", batch, "--gd-table", str(gd),
              "--low-res-counts", str(counts), "--out", str(out)])
    return pd.read_csv(out, sep="\t")


def test_locus_order_merges_del_dup_skips_non_nahr_and_sex(tmp_path):
    gd = tmp_path / "gd.tsv"
    _gd_table(gd)
    order = pmd.locus_order(str(gd))
    assert list(order["label"]) == ["L1", "L2"]
    assert list(order["x"]) == [0, 1]


def test_locus_label_strips_suffix():
    assert pmd.locus_label("15q11.2_DEL") == "15q11.2"
    assert pmd.locus_label("1q21.1_distal_DUP") == "1q21.1_distal"
    assert pmd.locus_label("plain") == "plain"


def test_mode_and_median():
    # 1.52, 1.54, 1.46 round to 1.5 (mode); 1.56 -> 1.6; 2.04 -> 2.0
    m, med, n, k = pmd.mode_and_median(np.array([1.52, 1.54, 1.56, 1.46, 2.04]))
    assert m == 1.5 and med == pytest.approx(1.52) and (n, k) == (5, 3)
    # tie goes to the value nearest 2
    assert pmd.mode_and_median(np.array([1.0, 1.0, 2.0, 2.0]))[0] == 2.0
    assert pmd.mode_and_median(np.array([np.nan])) is None


def test_category_depths(setup):
    gd, counts, tmp = setup
    out = _summarize(gd, counts, tmp / "b1.tsv.gz")
    sub = out[out["cluster"] == L1].copy()
    sub["cat"] = sub["sample"].str.rsplit("_", n=1).str[0]
    g = sub.groupby("cat")["median_depth"]

    near = lambda cat, v, tol=0.2: (abs(g.get_group(cat) - v) <= tol).all()
    assert near("noncarrier", 2.0)             # library size normalized away
    assert near("het_del", 1.0)
    assert near("het_dup", 3.0)
    for cat in ("mosaic_del", "mosaic_del2"):
        assert g.get_group(cat).between(1.2, 1.8).all(), cat
    assert g.get_group("mosaic_dup").between(2.2, 2.8).all()
    # partial overlap: near 1 or near 2 (3 / 2), never intermediate
    assert near("partial_del_major", 1.0)
    assert near("partial_del_minor", 2.0)
    assert near("partial_dup_major", 3.0)
    assert near("partial_dup_minor", 2.0)


def test_non_carriers_unaffected_at_other_locus(setup):
    gd, counts, tmp = setup
    out = _summarize(gd, counts, tmp / "b1.tsv.gz")
    other = out[out["cluster"] == _locus_key("chr2", 2_000_000, 2_400_000)]
    assert len(other) == len(out[out["cluster"] == L1])
    assert (abs(other["median_depth"] - 2.0) <= 0.2).all()


def test_non_nahr_and_sex_loci_excluded(setup):
    gd, counts, tmp = setup
    out = _summarize(gd, counts, tmp / "b1.tsv.gz")
    assert set(out["cluster"]) == {L1, _locus_key("chr2", 2_000_000, 2_400_000)}


def test_mosaic_flags_and_loci(setup):
    gd, counts, tmp = setup
    out = _summarize(gd, counts, tmp / "b1.tsv.gz")
    flagged = pmd.probable_mosaic_loci([out], min_samples=1)
    assert flagged == [L1]
    assert pmd.probable_mosaic_loci([out], min_samples=100) == []
    only_clean = out[~out["sample"].str.startswith("mosaic")]
    assert pmd.probable_mosaic_loci([only_clean], min_samples=1) == []


def test_merge_and_plot_single_and_stacked(setup):
    gd, counts, tmp = setup
    a = _summarize(gd, counts, tmp / "b1.tsv.gz", "b1")
    _summarize(gd, counts, tmp / "b2.tsv.gz", "b2")
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
    gd, counts, tmp = setup
    out = _summarize(gd, counts, tmp / "b1.tsv.gz")
    order = pmd.locus_order(str(gd))
    shown = pmd.draw_mode_depth_plot([out], ["x"], order, str(tmp / "p.pdf"))
    assert shown == ["L1"]


def test_plot_label_mismatch_errors(setup):
    gd, counts, tmp = setup
    _summarize(gd, counts, tmp / "b1.tsv.gz")
    with pytest.raises(SystemExit):
        pmd.main(["plot", "--gd-table", str(gd), "--data", str(tmp / "b1.tsv.gz"),
                  "--labels", "a", "b", "--out-pdf", str(tmp / "x.pdf")])


def test_ymax_capped_at_observed_max(setup):
    gd, counts, tmp = setup
    out = _summarize(gd, counts, tmp / "b1.tsv.gz")
    out = out[out["median_depth"] < 2.5]
    order = pmd.locus_order(str(gd))
    pmd.draw_mode_depth_plot([out], ["x"], order, str(tmp / "p.pdf"))
    assert (tmp / "p.pdf").stat().st_size > 0


def test_pdf_size_independent_of_point_count(tmp_path):
    gd = tmp_path / "gd.tsv"
    _gd_table(gd)
    order = pmd.locus_order(str(gd))
    r = np.random.RandomState(0)
    n = 20000
    df = pd.DataFrame({"batch": "b", "cluster": r.choice(order["cluster"], n),
                       "sample": [f"s{i}" for i in range(n)],
                       "mode_depth": 2.0, "median_depth": 2 + r.normal(0, .05, n),
                       "n_bins": 30, "mode_bins": 25})
    pmd.draw_mode_depth_plot([df], ["x"], order, str(tmp_path / "big.pdf"))
    assert (tmp_path / "big.pdf").stat().st_size < 2_000_000


HR_BIN = 100
SMALL = _locus_key("chr2", 2_500_000, 2_540_000)  # body ~20 kb: 2 low-res bins


def test_small_body_uses_highres_bins(tmp_path, monkeypatch):
    """A body shorter than 10x the low-res bin size is read from high-res bins."""
    from gatk_sv_gd import plot as plot_mod

    rng = np.random.RandomState(5)
    gd = tmp_path / "gd.tsv"
    pd.DataFrame([("chr2", 2_500_000, 2_540_000, "S1_DEL", "DEL", "yes", "no", "", "1", "2")],
                 columns=["chr", "start_GRCh38", "end_GRCh38", "GD_ID", "svtype", "NAHR",
                          "terminal", "cluster", "BP1", "BP2"]).to_csv(gd, sep="\t", index=False)
    locus = pmd.GDTable(str(gd)).loci[SMALL] if SMALL in pmd.GDTable(str(gd)).loci \
        else next(iter(pmd.GDTable(str(gd)).loci.values()))
    (b0, b1), = pmd.body_intervals(locus)

    n_samples = 6
    lib = rng.uniform(0.6, 1.6, n_samples)
    names = [f"s{i}" for i in range(n_samples)]
    starts = np.arange(0, GENOME_END, BIN)
    low = pd.DataFrame({"Chr": "chr2", "Start": starts, "End": starts + BIN})
    for i, n in enumerate(names):
        low[n] = rng.poisson(LAMBDA * lib[i], len(starts))
    hr_starts = np.arange(b0 - 5_000, b1 + 5_000, HR_BIN)
    hr = pd.DataFrame({"Chr": "chr2", "Start": hr_starts, "End": hr_starts + HR_BIN})
    for i, n in enumerate(names):
        hr[n] = rng.poisson(LAMBDA * lib[i] * HR_BIN / BIN * 0.5, len(hr_starts))  # het DEL

    def fake_query(path, chrom, start, end, sample_cols, max_bins=None):
        sub = hr[(hr["End"] > start) & (hr["Start"] < end)].copy()
        sub["source_file"] = "highres"
        sub.index = sub["Chr"] + ":" + sub["Start"].astype(str) + "-" + sub["End"].astype(str)
        sub.index.name = "Bin"
        return sub[["Chr", "Start", "End", "source_file"] + list(sample_cols)]

    monkeypatch.setattr(plot_mod, "query_highres_bins", fake_query)

    # Low-res alone: the body holds no bin midpoint (or a diluted one).
    base = pmd.summarize_matrix(low, str(gd), "b1")
    out = pmd.summarize_matrix(low, str(gd), "b1", highres_path="hr.tsv.gz")
    assert len(out) == n_samples
    assert out["n_bins"].min() >= (b1 - b0) // HR_BIN - 1
    assert out["median_depth"].between(0.5, 1.5).all()   # het DEL ~1, not ~2
    assert len(base) <= len(out)


def test_flag_mosaics_off_labels_all_no_band(setup):
    gd, counts, tmp = setup
    out = _summarize(gd, counts, tmp / "b1.tsv.gz")
    order = pmd.locus_order(str(gd))
    on = pmd.draw_mode_depth_plot([out], ["a"], order, str(tmp / "on.pdf"))
    off = pmd.draw_mode_depth_plot([out], ["a"], order, str(tmp / "off.pdf"),
                                   flag_mosaics=False)
    assert len(on) < len(order)
    assert off == list(order["label"])
