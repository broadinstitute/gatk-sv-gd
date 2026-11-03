import gzip
import json

import numpy as np
import pandas as pd
import pytest

from gatk_sv_gd import plot_mosaic as pm

CHROM = "chr15"
BIN = 10_000
REGION = (1_000_000, 3_000_000)
# BP1, BP2, BP3 (start, end); GD_A = BP1-BP2, GD_B = BP2-BP3.
BPS = {"1": (1_400_000, 1_410_000), "2": (2_000_000, 2_010_000), "3": (2_070_000, 2_080_000)}


def _write_gd_table(path):
    rows = [
        (CHROM, BPS["1"][0], BPS["2"][1], "GD_A", "DEL", "yes", "no", "c1", "1", "2"),
        (CHROM, BPS["2"][0], BPS["3"][1], "GD_B", "DEL", "yes", "no", "c1", "2", "3"),
        (CHROM, 5_000_000, 5_500_000, "GD_SOLO", "DUP", "yes", "no", "", "1", "2"),
    ]
    pd.DataFrame(rows, columns=[
        "chr", "start_GRCh38", "end_GRCh38", "GD_ID", "svtype", "NAHR",
        "terminal", "cluster", "BP1", "BP2"]).to_csv(path, sep="\t", index=False)


def _batch(tmp_path, name, samples, carriers, rng):
    """carriers: {sample: [GD_ID, ...]}; returns dict of file paths."""
    starts = np.arange(*REGION, BIN)
    counts = {}
    for s in samples:
        depth = np.full(len(starts), 100.0)
        if s.startswith("mos"):
            body = (starts >= BPS["1"][1]) & (starts < BPS["2"][0])
            depth[body] *= 0.75
        counts[s] = rng.poisson(depth)
    lowres = pd.DataFrame({"Chr": CHROM, "Start": starts, "End": starts + BIN, **counts})
    chr1 = pd.DataFrame({"Chr": "chr1", "Start": starts, "End": starts + BIN,
                         **{s: rng.poisson(100, len(starts)) for s in samples}})
    lowres = pd.concat([lowres, chr1], ignore_index=True)
    lowres_path = tmp_path / f"{name}.lowres.tsv.gz"
    lowres.to_csv(lowres_path, sep="\t", index=False, compression="gzip")

    post = pd.DataFrame([(("c1"), CHROM, a, a + BIN, s, 2.0)
                         for s in samples for a in starts],
                        columns=["cluster", "chr", "start", "end", "sample", "depth"])
    post_path = tmp_path / f"{name}.cn_posteriors.tsv.gz"
    post.to_csv(post_path, sep="\t", index=False, compression="gzip")

    call_rows = [(s, "c1", gd, True) for s, gds in carriers.items() for gd in gds]
    calls = pd.DataFrame(call_rows, columns=["sample", "cluster", "GD_ID", "is_carrier"])
    calls_path = tmp_path / f"{name}.calls.tsv.gz"
    calls.to_csv(calls_path, sep="\t", index=False, compression="gzip")
    return dict(lowres=str(lowres_path), post=str(post_path),
                calls=str(calls_path))


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.RandomState(0)
    _write_gd_table(tmp_path / "gd.tsv")
    (tmp_path / "sd.bed").write_text(
        f"{CHROM}\t1350000\t1400000\tsd\t0\t+\n{CHROM}\t2010000\t2070000\tsd\t0\t+\n")
    b1 = [f"b1_s{i}" for i in range(30)] + ["mos1"]
    b2 = [f"b2_s{i}" for i in range(30)] + ["mos2"]
    b3 = [f"b3_s{i}" for i in range(30)]
    c1 = {"b1_s0": ["GD_A"], "b1_s1": ["GD_A"], "b1_s2": ["GD_A", "GD_B"], "b1_s3": ["GD_B"]}
    c2 = {"b2_s0": ["GD_A"], "b2_s1": ["GD_A"]}
    c3 = {"b3_s0": ["GD_A"]}
    return tmp_path, {
        "b1": _batch(tmp_path, "b1", b1, c1, rng),
        "b2": _batch(tmp_path, "b2", b2, c2, rng),
        "b3": _batch(tmp_path, "b3", b3, c3, rng),
    }


def test_classify_excludes_other_gd_carriers(dataset):
    tmp_path, files = dataset
    pm.main(["classify", "--batch", "b1", "--gd-table", str(tmp_path / "gd.tsv"),
             "--gd-id", "GD_A", "--calls", files["b1"]["calls"],
             "--cn-posteriors", files["b1"]["post"],
             "--out-prefix", str(tmp_path / "b1")])
    classes = pd.read_csv(tmp_path / "b1.classes.tsv", sep="\t").set_index("sample")
    assert classes.loc["b1_s0", "status"] == pm.STATUS_GERMLINE
    assert classes.loc["b1_s2", "status"] == pm.STATUS_OTHER  # also GD_B in cluster
    assert classes.loc["b1_s3", "status"] == pm.STATUS_OTHER
    assert classes.loc["b1_s10", "status"] == pm.STATUS_NON_CARRIER
    region = json.load(open(tmp_path / "b1.region.json"))
    assert region["cluster"] == "c1" and region["start"] == REGION[0]


def test_select_draws_only_from_mosaic_batches(dataset):
    tmp_path, files = dataset
    for b in files:
        pm.main(["classify", "--batch", b, "--gd-table", str(tmp_path / "gd.tsv"),
                 "--gd-id", "GD_A", "--calls", files[b]["calls"],
                 "--cn-posteriors", files[b]["post"],
                 "--out-prefix", str(tmp_path / b)])
    (tmp_path / "mosaic.txt").write_text("mos1\nmos2\n")
    pm.main(["select", "--classes"] + [str(tmp_path / f"{b}.classes.tsv") for b in files]
            + ["--regions"] + [str(tmp_path / f"{b}.region.json") for b in files]
            + ["--mosaic-samples", str(tmp_path / "mosaic.txt"),
               "--n-germline", "3", "--n-non-carrier", "20", "--seed", "1",
               "--out-prefix", str(tmp_path / "sel")])
    sel = pd.read_csv(tmp_path / "sel.selected.tsv", sep="\t")
    assert set(sel["batch"]) == {"b1", "b2"}
    g = sel[sel["group"] == pm.GROUP_GERMLINE]
    assert len(g) == 3 and set(g["sample"]) <= {"b1_s0", "b1_s1", "b2_s0", "b2_s1"}
    assert len(sel[sel["group"] == pm.GROUP_NON_CARRIER]) == 20
    assert not {"b1_s2", "b1_s3"} & set(sel["sample"])
    flags = (tmp_path / "sel.use_batch.txt").read_text().split()
    assert flags == ["true", "true", "false"]


def test_select_reports_missing_mosaic_sample():
    classes = pd.DataFrame({"sample": ["a"], "batch": ["b"], "status": ["non_carrier"],
                            "ploidy": [2]})
    with pytest.raises(SystemExit):
        pm.select_samples(classes, ["nope"], 1, 1, 0)


def test_end_to_end_pdf(dataset):
    tmp_path, files = dataset
    gd = str(tmp_path / "gd.tsv")
    for b in files:
        pm.main(["classify", "--batch", b, "--gd-table", gd, "--gd-id", "GD_A",
                 "--calls", files[b]["calls"], "--cn-posteriors", files[b]["post"],
                 "--out-prefix", str(tmp_path / b)])
    (tmp_path / "mosaic.txt").write_text("mos1\nmos2\n")
    pm.main(["select", "--classes"] + [str(tmp_path / f"{b}.classes.tsv") for b in files]
            + ["--regions"] + [str(tmp_path / f"{b}.region.json") for b in files]
            + ["--mosaic-samples", str(tmp_path / "mosaic.txt"),
               "--out-prefix", str(tmp_path / "sel")])
    for b in ("b1", "b2"):
        pm.main(["extract", "--batch", b, "--gd-table", gd, "--gd-id", "GD_A",
                 "--selected", str(tmp_path / "sel.selected.tsv"),
                 "--region", str(tmp_path / "sel.region.json"),
                 "--low-res-counts", files[b]["lowres"],
                 "--out-prefix", str(tmp_path / b)])
    depth = pd.read_csv(tmp_path / "b1.depth.tsv.gz", sep="\t")
    # Normalised to CN 2 and the mosaic sample dips in the BP1-BP2 body.
    body = depth[(depth["Start"] > BPS["1"][1]) & (depth["End"] < BPS["2"][0])]
    assert body["mos1"].median() == pytest.approx(1.5, abs=0.1)
    assert body["b1_s5"].median() == pytest.approx(2.0, abs=0.1)

    out = tmp_path / "plot.pdf"
    pm.main(["plot", "--gd-table", gd, "--gd-id", "GD_A",
             "--selected", str(tmp_path / "sel.selected.tsv"),
             "--region", str(tmp_path / "sel.region.json"),
             "--depth", str(tmp_path / "b1.depth.tsv.gz"), str(tmp_path / "b2.depth.tsv.gz"),
             "--segdup-bed", str(tmp_path / "sd.bed"), "--out-pdf", str(out)])
    assert out.read_bytes().startswith(b"%PDF")


def test_classify_handles_empty_calls_and_ploidy_table(dataset):
    tmp_path, files = dataset
    empty = tmp_path / "empty.tsv"
    empty.write_text("")
    ploidy = tmp_path / "ploidy.tsv"
    pd.DataFrame({"sample": ["b1_s0", "b1_s1"], "contig": [CHROM, CHROM],
                  "ploidy": [1, 2]}).to_csv(ploidy, sep="\t", index=False)
    pm.main(["classify", "--batch", "b1", "--gd-table", str(tmp_path / "gd.tsv"),
             "--gd-id", "GD_A", "--calls", str(empty),
             "--cn-posteriors", files["b1"]["post"], "--ploidy-table", str(ploidy),
             "--out-prefix", str(tmp_path / "e")])
    classes = pd.read_csv(tmp_path / "e.classes.tsv", sep="\t").set_index("sample")
    assert (classes["status"] == pm.STATUS_NON_CARRIER).all()
    assert classes.loc["b1_s0", "ploidy"] == 1 and classes.loc["b1_s5", "ploidy"] == 2


def test_classify_unknown_gd_id_fails(dataset):
    tmp_path, files = dataset
    with pytest.raises(SystemExit):
        pm.main(["classify", "--batch", "b1", "--gd-table", str(tmp_path / "gd.tsv"),
                 "--gd-id", "NOPE", "--calls", files["b1"]["calls"],
                 "--cn-posteriors", files["b1"]["post"],
                 "--out-prefix", str(tmp_path / "x")])


def test_select_matches_mosaic_ploidy_and_is_seeded():
    rows = [("m", "b", "non_carrier", 1)] + [(f"s{i}", "b", "non_carrier", 1 if i < 5 else 2)
                                             for i in range(10)]
    classes = pd.DataFrame(rows, columns=["sample", "batch", "status", "ploidy"])
    a = pm.select_samples(classes, ["m"], 0, 3, seed=7)
    b = pm.select_samples(classes, ["m"], 0, 3, seed=7)
    assert a.equals(b)
    comparators = a[a["group"] == pm.GROUP_NON_CARRIER]["sample"]
    assert len(comparators) == 3 and all(int(s[1:]) < 5 for s in comparators)


def test_select_fails_without_bins_in_mosaic_batch(tmp_path):
    classes = tmp_path / "c.tsv"
    pd.DataFrame({"sample": ["m"], "batch": ["b"], "status": ["non_carrier"],
                  "ploidy": [2]}).to_csv(classes, sep="\t", index=False)
    region = tmp_path / "r.json"
    region.write_text(json.dumps({"batch": "b", "cluster": "c", "chrom": "chr1",
                                  "start": None, "end": None, "n_samples": 1}))
    (tmp_path / "m.txt").write_text("m\n")
    with pytest.raises(SystemExit):
        pm.main(["select", "--classes", str(classes), "--regions", str(region),
                 "--mosaic-samples", str(tmp_path / "m.txt"),
                 "--out-prefix", str(tmp_path / "o")])


def test_body_ticks_are_round_and_bounded():
    ticks = pm._body_ticks_mb(1_400_000, 2_080_000)
    assert 2 <= len(ticks) <= 3 and all(t % 50_000 == 0 for t in ticks)


def test_plot_without_segdup_and_with_custom_title(dataset):
    tmp_path, _ = dataset
    locus = pm._resolve_locus(str(tmp_path / "gd.tsv"), "GD_A")
    sel = pd.DataFrame({"sample": ["a", "b"], "batch": ["x", "x"],
                        "group": [pm.GROUP_MOSAIC, pm.GROUP_NON_CARRIER]})
    starts = np.arange(*REGION, BIN)
    depth = pd.DataFrame({"Chr": CHROM, "Start": starts, "End": starts + BIN,
                          "a": 2.0, "b": 2.0})
    out = tmp_path / "t.pdf"
    pm.draw_mosaic_plot(locus, {"start": REGION[0], "end": REGION[1]}, sel, [depth],
                        None, str(out), title="custom")
    assert out.read_bytes().startswith(b"%PDF")


def _classes(rows):
    return pd.DataFrame(rows, columns=["sample", "batch", "status", "ploidy"])


def test_choose_extra_batches_minimal_and_random_ties():
    counts = {"a": 5, "b": 2, "c": 2, "d": 1}
    assert pm.choose_extra_batches(counts, 4, np.random.RandomState(0)) == ["a"]
    assert pm.choose_extra_batches(counts, 0, np.random.RandomState(0)) == []
    # need 6: a+b, a+c (k=2 ties; a+d = 6 too)
    seen = {tuple(pm.choose_extra_batches(counts, 6, np.random.RandomState(s)))
            for s in range(40)}
    assert seen == {("a", "b"), ("a", "c"), ("a", "d")}
    with pytest.raises(ValueError):
        pm.choose_extra_batches(counts, 11, np.random.RandomState(0))


def test_select_adds_fewest_extra_batches_for_germline():
    rows = [("mos", "m", "non_carrier", 2), ("g_m0", "m", "germline_carrier", 2)]
    rows += [(f"x{i}", "x", "germline_carrier", 2) for i in range(5)]
    rows += [(f"y{i}", "y", "germline_carrier", 2) for i in range(2)]
    rows += [(f"z{i}", "z", "germline_carrier", 2) for i in range(2)]
    rows += [(f"n{i}", "m", "non_carrier", 2) for i in range(5)]
    cls = _classes(rows)
    sel = pm.select_samples(cls, ["mos"], 4, 3, seed=1)
    g = sel[sel["group"] == pm.GROUP_GERMLINE]
    assert len(g) == 4 and "g_m0" in set(g["sample"])
    assert set(g["batch"]) == {"m", "x"}          # one extra batch (x) suffices
    # non-carriers still only from mosaic batches
    nc = sel[sel["group"] == pm.GROUP_NON_CARRIER]
    assert set(nc["batch"]) == {"m"}


def test_select_caps_germline_at_total_available():
    rows = [("mos", "m", "non_carrier", 2), ("g0", "m", "germline_carrier", 2),
            ("y0", "y", "germline_carrier", 2), ("y1", "y", "germline_carrier", 2)]
    sel = pm.select_samples(_classes(rows), ["mos"], 10, 0, seed=0)
    g = sel[sel["group"] == pm.GROUP_GERMLINE]
    assert set(g["sample"]) == {"g0", "y0", "y1"}


def test_select_cli_flags_extra_batches(dataset):
    tmp_path, files = dataset
    for b in files:
        pm.main(["classify", "--batch", b, "--gd-table", str(tmp_path / "gd.tsv"),
                 "--gd-id", "GD_A", "--calls", files[b]["calls"],
                 "--cn-posteriors", files[b]["post"],
                 "--out-prefix", str(tmp_path / b)])
    (tmp_path / "mosaic.txt").write_text("mos1\n")
    pm.main(["select", "--classes"] + [str(tmp_path / f"{b}.classes.tsv") for b in files]
            + ["--regions"] + [str(tmp_path / f"{b}.region.json") for b in files]
            + ["--mosaic-samples", str(tmp_path / "mosaic.txt"),
               "--n-germline", "3", "--n-non-carrier", "5", "--seed", "1",
               "--out-prefix", str(tmp_path / "sel")])
    sel = pd.read_csv(tmp_path / "sel.selected.tsv", sep="\t")
    g = sel[sel["group"] == pm.GROUP_GERMLINE]
    # b1 holds 2 clean germline carriers; 1 more needs b2 (2) or b3 (1): tie
    assert len(g) == 3 and (g["batch"] == "b1").sum() == 2
    used = (tmp_path / "sel.batches.txt").read_text().split()
    assert len(used) == 2 and "b1" in used
    flags = (tmp_path / "sel.use_batch.txt").read_text().split()
    assert flags.count("true") == 2


def _halve(path, sample, start, end):
    df = pd.read_csv(path, sep="\t")
    m = (df["Chr"] == CHROM) & (df["Start"] >= start) & (df["End"] <= end)
    df.loc[m, sample] = (df.loc[m, sample] / 2).round().astype(int)
    df.to_csv(path, sep="\t", index=False, compression="gzip")


def test_classify_flags_abnormal_flank_and_other_interval(dataset):
    tmp_path, files = dataset
    f = files["b1"]
    _halve(f["lowres"], "b1_s5", 1_100_000, 1_300_000)      # left flank
    _halve(f["lowres"], "b1_s6", 2_010_000, 2_070_000)      # GD_B interval, outside GD_A
    _halve(f["lowres"], "b1_s7", 1_100_000, 1_110_000)      # 1 flank bin: ignored
    pm.main(["classify", "--batch", "b1", "--gd-table", str(tmp_path / "gd.tsv"),
             "--gd-id", "GD_A", "--calls", f["calls"], "--cn-posteriors", f["post"],
             "--low-res-counts", f["lowres"], "--out-prefix", str(tmp_path / "b1")])
    c = pd.read_csv(tmp_path / "b1.classes.tsv", sep="\t").set_index("sample")
    assert c.loc["b1_s5", "abnormal"] and c.loc["b1_s5", "abnormal_region"] == "left_flank"
    assert c.loc["b1_s6", "abnormal"] and c.loc["b1_s6", "abnormal_region"] == "2-3"
    assert not c.loc["b1_s7", "abnormal"]
    assert not c.loc["b1_s10", "abnormal"] and not c.loc["mos1", "abnormal"]


def test_select_excludes_abnormal_comparators_but_keeps_mosaic():
    rows = [("m", "b1", pm.STATUS_NON_CARRIER, 2, True, "left_flank"),
            ("g1", "b1", pm.STATUS_GERMLINE, 2, True, "left_flank"),
            ("g2", "b1", pm.STATUS_GERMLINE, 2, False, ""),
            ("n1", "b1", pm.STATUS_NON_CARRIER, 2, True, "right_flank"),
            ("n2", "b1", pm.STATUS_NON_CARRIER, 2, False, "")]
    classes = pd.DataFrame(rows, columns=["sample", "batch", "status", "ploidy",
                                          "abnormal", "abnormal_region"])
    out = pm.select_samples(classes, ["m"], 10, 10, 1)
    assert set(out["sample"]) == {"m", "g2", "n2"}
