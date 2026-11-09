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


def test_common_polymorphism_not_flagged_but_rare_event_is(dataset):
    tmp_path, files = dataset
    f = files["b1"]
    for i in range(8):                                        # common: 8 of 31 samples
        _halve(f["lowres"], f"b1_s{10 + i}", 1_100_000, 1_300_000)
    _halve(f["lowres"], "b1_s20", 2_100_000, 2_300_000)        # rare: right flank
    pm.main(["classify", "--batch", "b1", "--gd-table", str(tmp_path / "gd.tsv"),
             "--gd-id", "GD_A", "--calls", f["calls"], "--cn-posteriors", f["post"],
             "--low-res-counts", f["lowres"], "--out-prefix", str(tmp_path / "b1")])
    c = pd.read_csv(tmp_path / "b1.classes.tsv", sep="\t").set_index("sample")
    assert not c.loc[[f"b1_s{10 + i}" for i in range(8)], "abnormal"].any()
    assert c.loc["b1_s20", "abnormal"] and c.loc["b1_s20", "abnormal_region"] == "right_flank"


def test_segdup_overlapping_bins_ignored(dataset):
    tmp_path, files = dataset
    f = files["b1"]
    _halve(f["lowres"], "b1_s5", 1_100_000, 1_300_000)       # left-flank dip
    bed = tmp_path / "flank_sd.bed"
    bed.write_text(f"{CHROM}\t1099999\t1300001\tsd\t0\t+\n")  # overlaps the whole dip
    base = ["classify", "--batch", "b1", "--gd-table", str(tmp_path / "gd.tsv"),
            "--gd-id", "GD_A", "--calls", f["calls"], "--cn-posteriors", f["post"],
            "--low-res-counts", f["lowres"]]
    pm.main(base + ["--out-prefix", str(tmp_path / "plain")])
    pm.main(base + ["--segdup-bed", str(bed), "--out-prefix", str(tmp_path / "sd")])
    plain = pd.read_csv(tmp_path / "plain.classes.tsv", sep="\t").set_index("sample")
    masked = pd.read_csv(tmp_path / "sd.classes.tsv", sep="\t").set_index("sample")
    assert plain.loc["b1_s5", "abnormal"]
    assert not masked.loc["b1_s5", "abnormal"]


def test_overlaps_segdup_one_bp_counts():
    sd = {"chr1": (np.array([100, 500]), np.array([200, 600]))}
    starts = np.array([0, 190, 200, 450, 599])
    ends = starts + 10
    # bins: [0,10) no, [190,200) yes, [200,210) no (touching), [450,460) no, [599,609) yes
    assert list(pm.overlaps_segdup(sd, "chr1", starts, ends)) == [False, True, False, False, True]
    assert not pm.overlaps_segdup(sd, "chr2", starts, ends).any()


def test_include_samples_restricts_selection():
    rows = [("m1", "b1", pm.STATUS_NON_CARRIER, 2), ("m2", "b1", pm.STATUS_NON_CARRIER, 2)]
    rows += [(f"g{i}", "b1", pm.STATUS_GERMLINE, 2) for i in range(6)]
    rows += [(f"n{i}", "b1", pm.STATUS_NON_CARRIER, 2) for i in range(6)]
    classes = pd.DataFrame(rows, columns=["sample", "batch", "status", "ploidy"])
    allowed = ["m1", "g0", "g1", "n0", "n1", "n2"]
    out = pm.select_samples(classes, ["m1", "m2"], 10, 10, 1, allowed)
    assert set(out["sample"]) == set(allowed)                 # m2 dropped, comparators capped
    with pytest.raises(SystemExit):
        pm.select_samples(classes, ["m2"], 10, 10, 1, allowed)


def test_include_samples_cli(dataset):
    tmp_path, files = dataset
    for b in ("b1", "b2", "b3"):
        pm.main(["classify", "--batch", b, "--gd-table", str(tmp_path / "gd.tsv"),
                 "--gd-id", "GD_A", "--calls", files[b]["calls"],
                 "--cn-posteriors", files[b]["post"], "--out-prefix", str(tmp_path / b)])
    (tmp_path / "mos.txt").write_text("mos1\n")
    keep = ["mos1", "b1_s0", "b1_s10", "b1_s11", "b2_s10"]
    (tmp_path / "keep.txt").write_text("\n".join(keep) + "\n")
    pm.main(["select", "--classes"] + [str(tmp_path / f"{b}.classes.tsv") for b in ("b1", "b2", "b3")]
            + ["--regions"] + [str(tmp_path / f"{b}.region.json") for b in ("b1", "b2", "b3")]
            + ["--mosaic-samples", str(tmp_path / "mos.txt"),
               "--include-samples", str(tmp_path / "keep.txt"),
               "--out-prefix", str(tmp_path / "sel")])
    sel = pd.read_csv(tmp_path / "sel.selected.tsv", sep="\t")
    assert set(sel["sample"]) <= set(keep) and "mos1" in set(sel["sample"])


def test_allocate_germline_rules():
    a = pm.allocate_germline
    assert a({"DEL": 40}, 10) == {"DEL": 10}
    assert a({"DEL": 4}, 10) == {"DEL": 4}
    assert a({"DEL": 50, "DUP": 2}, 10) == {"DEL": 8, "DUP": 2}          # <=3: all of it
    assert a({"DEL": 2, "DUP": 3}, 10) == {"DEL": 2, "DUP": 3}
    both = a({"DEL": 300, "DUP": 100}, 20)                                # 3:1
    assert both == {"DEL": 15, "DUP": 5}
    assert a({"DEL": 1000, "DUP": 4}, 10) == {"DEL": 7, "DUP": 3}         # >=3 of each
    assert a({"DEL": 1000, "DUP": 4}, 4) == {"DEL": 3, "DUP": 3}          # min 3 beats the total
    mixed = a({"DEL": 7, "DUP": 5}, 10)
    assert min(mixed.values()) >= 3 and sum(mixed.values()) == 10
    assert sum(a({"DEL": 7, "DUP": 5}, 10).values()) == 10
    assert a({"DEL": 5, "DUP": 5}, 100) == {"DEL": 5, "DUP": 5}           # capped at available


def test_classify_includes_partner_gd_as_germline():
    calls = pd.DataFrame([("a", "c1", "G_DEL_x", True), ("b", "c1", "G_DUP_x", True),
                          ("c", "c1", "G_DEL_x", True), ("c", "c1", "G_DUP_x", True),
                          ("d", "c1", "OTHER", True)],
                         columns=["sample", "cluster", "GD_ID", "is_carrier"])
    out = pm.classify_samples(calls, list("abcde"), "c1", "G_DEL_x", "G_DUP_x",
                              {"G_DEL_x": "DEL", "G_DUP_x": "DUP"}).set_index("sample")
    assert out.loc["a", "status"] == pm.STATUS_GERMLINE and out.loc["a", "germline_type"] == "DEL"
    assert out.loc["b", "status"] == pm.STATUS_GERMLINE and out.loc["b", "germline_type"] == "DUP"
    assert out.loc["c", "status"] == pm.STATUS_OTHER           # both DEL and DUP
    assert out.loc["d", "status"] == pm.STATUS_OTHER
    assert out.loc["e", "status"] == pm.STATUS_NON_CARRIER


def test_select_mixes_del_and_dup_germline():
    rows = [("m", "b1", pm.STATUS_NON_CARRIER, "", 2)]
    rows += [(f"del{i}", "b1", pm.STATUS_GERMLINE, "DEL", 2) for i in range(30)]
    rows += [(f"dup{i}", "b1", pm.STATUS_GERMLINE, "DUP", 2) for i in range(10)]
    rows += [(f"n{i}", "b1", pm.STATUS_NON_CARRIER, "", 2) for i in range(5)]
    classes = pd.DataFrame(rows, columns=["sample", "batch", "status", "germline_type", "ploidy"])
    out = pm.select_samples(classes, ["m"], 20, 5, 1)
    g = out[out["group"] == pm.GROUP_GERMLINE]
    assert g["carrier_type"].value_counts().to_dict() == {"DEL": 15, "DUP": 5}   # 3:1
    assert (out[out["group"] != pm.GROUP_GERMLINE]["carrier_type"] == "").all()


def test_partner_gd_id(tmp_path):
    _write_gd_table_pair = tmp_path / "gd.tsv"
    pd.DataFrame([("chr1", 100, 200, "G_DEL_chr1_100_200", "DEL", "yes", "no", "c", 1, 2),
                  ("chr1", 100, 200, "G_DUP_chr1_100_200", "DUP", "yes", "no", "c", 1, 2),
                  ("chr1", 500, 600, "H_DEL_chr1_500_600", "DEL", "yes", "no", "d", 1, 2)],
                 columns=["chr", "start_GRCh38", "end_GRCh38", "GD_ID", "svtype", "NAHR",
                          "terminal", "cluster", "BP1", "BP2"]).to_csv(
        _write_gd_table_pair, sep="\t", index=False)
    pm_locus = pm._resolve_locus(str(_write_gd_table_pair), "G_DEL_chr1_100_200")
    assert pm.partner_gd_id(pm_locus, "G_DEL_chr1_100_200") == "G_DUP_chr1_100_200"
    assert pm.partner_gd_id(pm_locus, "G_DUP_chr1_100_200") == "G_DEL_chr1_100_200"
    solo = pm._resolve_locus(str(_write_gd_table_pair), "H_DEL_chr1_500_600")
    assert pm.partner_gd_id(solo, "H_DEL_chr1_500_600") is None


def test_end_to_end_del_and_dup_germline_legend(dataset):
    tmp_path, files = dataset
    gd = tmp_path / "gd2.tsv"
    pd.DataFrame([
        (CHROM, BPS["1"][0], BPS["2"][1], "GD_DEL_A", "DEL", "yes", "no", "c1", "1", "2"),
        (CHROM, BPS["1"][0], BPS["2"][1], "GD_DUP_A", "DUP", "yes", "no", "c1", "1", "2"),
    ], columns=["chr", "start_GRCh38", "end_GRCh38", "GD_ID", "svtype", "NAHR",
                "terminal", "cluster", "BP1", "BP2"]).to_csv(gd, sep="\t", index=False)
    carriers = {
        "b1": {**{f"b1_s{i}": ["GD_DEL_A"] for i in range(0, 12)},
               **{f"b1_s{i}": ["GD_DUP_A"] for i in range(12, 16)}},
        "b2": {f"b2_s{i}": ["GD_DEL_A"] for i in range(0, 6)},
        "b3": {"b3_s0": ["GD_DUP_A"]},
    }
    for b, c in carriers.items():
        rows = [(s, "c1", g, True) for s, gds in c.items() for g in gds]
        pd.DataFrame(rows, columns=["sample", "cluster", "GD_ID", "is_carrier"]).to_csv(
            files[b]["calls"], sep="\t", index=False, compression="gzip")
        pm.main(["classify", "--batch", b, "--gd-table", str(gd), "--gd-id", "GD_DEL_A",
                 "--calls", files[b]["calls"], "--cn-posteriors", files[b]["post"],
                 "--out-prefix", str(tmp_path / b)])
    (tmp_path / "mosaic.txt").write_text("mos1\n")
    pm.main(["select", "--classes"] + [str(tmp_path / f"{b}.classes.tsv") for b in files]
            + ["--regions"] + [str(tmp_path / f"{b}.region.json") for b in files]
            + ["--mosaic-samples", str(tmp_path / "mosaic.txt"), "--n-germline", "10",
               "--out-prefix", str(tmp_path / "sel")])
    sel = pd.read_csv(tmp_path / "sel.selected.tsv", sep="\t").fillna("")
    counts = sel[sel["group"] == "germline"]["carrier_type"].value_counts().to_dict()
    # 17 DEL vs 5 DUP in the cohort: DUP > 3, so a proportional split (~8 / 2) of 10
    assert sum(counts.values()) == 10 and counts["DUP"] >= 1 and counts["DEL"] > counts["DUP"]
    used = sorted(sel["batch"].unique())
    for b in used:
        pm.main(["extract", "--batch", b, "--gd-table", str(gd), "--gd-id", "GD_DEL_A",
                 "--selected", str(tmp_path / "sel.selected.tsv"),
                 "--region", str(tmp_path / "sel.region.json"),
                 "--low-res-counts", files[b]["lowres"], "--out-prefix", str(tmp_path / b)])
    out = tmp_path / "plot2.pdf"
    pm.main(["plot", "--gd-table", str(gd), "--gd-id", "GD_DEL_A",
             "--selected", str(tmp_path / "sel.selected.tsv"),
             "--region", str(tmp_path / "sel.region.json"),
             "--depth"] + [str(tmp_path / f"{b}.depth.tsv.gz") for b in used]
            + ["--out-pdf", str(out)])
    assert out.read_bytes().startswith(b"%PDF")


def test_legend_key_labels():
    assert pm.key_label("Germline", 10, {"DEL": 8, "DUP": 2}) == "Germline (DEL n=8; DUP n=2)"
    assert pm.key_label("Germline", 8, {"DEL": 8}) == "Germline DEL (n=8)"
    assert pm.key_label("Germline", 8, {"": 8}) == "Germline (n=8)"
    assert pm.key_label("Mosaic", 3, {"DEL": 3}) == "Mosaic (n=3)"
    assert pm.key_label("Mosaic", 5, {"DEL": 3, "DUP": 2}) == "Mosaic (DEL n=3; DUP n=2)"


def test_mosaic_del_and_dup_split_in_plot(dataset):
    tmp_path, files = dataset
    gd = str(tmp_path / "gd.tsv")
    df = pd.read_csv(files["b2"]["lowres"], sep="\t")
    body = (df["Chr"] == CHROM) & (df["Start"] >= BPS["1"][1]) & (df["End"] <= BPS["2"][0])
    df.loc[body, "mos2"] = (df.loc[body, "mos2"] * 5 / 3).round().astype(int)   # 0.75x -> 1.25x
    df.to_csv(files["b2"]["lowres"], sep="\t", index=False, compression="gzip")
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
                 "--low-res-counts", files[b]["lowres"], "--out-prefix", str(tmp_path / b)])
    captured = {}
    orig = pm.key_label
    pm.key_label = lambda name, total, by_type, cohort=None: captured.setdefault(
        name, (total, dict(by_type))) and orig(name, total, by_type, cohort)
    try:
        pm.main(["plot", "--gd-table", gd, "--gd-id", "GD_A",
                 "--selected", str(tmp_path / "sel.selected.tsv"),
                 "--region", str(tmp_path / "sel.region.json"),
                 "--depth", str(tmp_path / "b1.depth.tsv.gz"), str(tmp_path / "b2.depth.tsv.gz"),
                 "--out-pdf", str(tmp_path / "p.pdf")])
    finally:
        pm.key_label = orig
    assert captured["Mosaic"] == (2, {"DEL": 1, "DUP": 1})


def test_legend_marks_selected_when_cohort_is_larger():
    assert pm.key_label("Non-carrier", 50, {}, {"": 300}) == "Non-carrier (n=50, selected)"
    assert pm.key_label("Non-carrier", 12, {}, {"": 12}) == "Non-carrier (n=12)"
    assert pm.key_label("Germline", 20, {"DEL": 20}, {"DEL": 45}) == "Germline DEL (n=20, selected)"
    assert pm.key_label("Germline", 10, {"DEL": 8, "DUP": 2}, {"DEL": 30, "DUP": 2}) \
        == "Germline (DEL n=8, selected; DUP n=2)"


def test_select_reports_cohort_totals():
    rows = [("m", "b1", pm.STATUS_NON_CARRIER, "", 2)]
    rows += [(f"del{i}", "b1", pm.STATUS_GERMLINE, "DEL", 2) for i in range(30)]
    rows += [(f"n{i}", "b1", pm.STATUS_NON_CARRIER, "", 2) for i in range(40)]
    classes = pd.DataFrame(rows, columns=["sample", "batch", "status", "germline_type", "ploidy"])
    out = pm.select_samples(classes, ["m"], 10, 25, 1)
    assert set(out.loc[out["group"] == pm.GROUP_GERMLINE, "cohort_total"]) == {30}
    assert set(out.loc[out["group"] == pm.GROUP_NON_CARRIER, "cohort_total"]) == {40}
    assert out.loc[out["group"] == pm.GROUP_MOSAIC, "cohort_total"].isna().all()


def test_mosaic_type_uses_target_gd_span_and_cutoff_2(dataset):
    tmp_path, _ = dataset
    locus = pm._resolve_locus(str(tmp_path / "gd.tsv"), "GD_A")
    starts = np.arange(*REGION, BIN)
    mids = starts + BIN / 2
    in_a = (mids >= BPS["1"][0]) & (mids < BPS["2"][1])
    # DEL inside GD_A; elevated elsewhere in the cluster (would flip a cluster-wide median)
    depth = pd.DataFrame({"Chr": CHROM, "Start": starts, "End": starts + BIN,
                          "m": np.where(in_a, 1.5, 3.0), "n": 2.0})
    sel = pd.DataFrame({"sample": ["m", "n"], "batch": "x",
                        "group": [pm.GROUP_MOSAIC, pm.GROUP_NON_CARRIER], "ploidy": [2, 2]})
    captured = {}
    orig = pm.key_label
    pm.key_label = lambda name, total, by_type, cohort=None: captured.setdefault(
        name, dict(by_type)) and orig(name, total, by_type, cohort)
    try:
        pm.draw_mosaic_plot(locus, {"start": REGION[0], "end": REGION[1]}, sel, [depth],
                            None, str(tmp_path / "t.pdf"), gd_id="GD_A")
    finally:
        pm.key_label = orig
    assert captured["Mosaic"] == {"DEL": 1}
