import numpy as np
import pandas as pd
import pytest

from gatk_sv_gd.models import GDTable, validate_gd_table_for_preprocess
import gatk_sv_gd.preprocess as preprocess_module


class _FakeLocus:
    def __init__(self):
        self.chrom = "chr1"
        self.start = 100
        self.end = 300
        self.breakpoints = [(100, 100), (200, 200), (300, 300)]
        self.gd_entries = [{"GD_ID": "GD1"}]
        self.svtypes = ["DEL"]
        self.is_nahr = True

    def get_intervals(self):
        return [
            (100, 200, "A-C"),
            (200, 300, "C-D"),
        ]


class _FakeGDTable:
    def __init__(self, locus):
        self.locus = locus

    def get_all_loci(self):
        return {"cluster1": self.locus}


class _ChrXLocus:
    chrom = "chrX"
    start = 100
    end = 200
    breakpoints = [(100, 100), (200, 200)]
    gd_entries = [{"GD_ID": "GDX"}]
    svtypes = ["DEL"]
    is_nahr = True

    def get_intervals(self):
        return [(100, 200, "1-2")]


class _IntervalMask:
    def __init__(self, chrom, start, end):
        self.chrom = chrom
        self.start = start
        self.end = end

    def get_overlap_fractions_batch(self, chrom, starts, ends):
        if chrom != self.chrom:
            return np.zeros(len(starts))
        starts = np.asarray(starts)
        ends = np.asarray(ends)
        overlap = np.maximum(0, np.minimum(ends, self.end) - np.maximum(starts, self.start))
        lengths = np.maximum(1, ends - starts)
        return overlap / lengths


def _install_collect_highres_stubs(monkeypatch, highres_df):
    filter_max_values = []
    original_filter = preprocess_module._filter_and_prepare_locus_bins

    def spy_filter(*args, **kwargs):
        filter_max_values.append(args[5])
        return original_filter(*args, **kwargs)

    monkeypatch.setattr(preprocess_module, "compute_flank_regions_from_bins", lambda *args, **kwargs: [])
    monkeypatch.setattr(preprocess_module, "query_highres_bins", lambda *args, **kwargs: highres_df.copy())
    monkeypatch.setattr(preprocess_module, "normalize_highres_bins", lambda raw_df, *args, **kwargs: raw_df.copy())
    monkeypatch.setattr(preprocess_module, "_filter_and_prepare_locus_bins", spy_filter)
    return filter_max_values


def _lowres_df():
    return pd.DataFrame([
        {"Chr": "chr1", "Start": 100, "End": 200, "sample1": 2.0},
        {"Chr": "chr1", "Start": 200, "End": 230, "sample1": 2.1},
        {"Chr": "chr1", "Start": 230, "End": 260, "sample1": 2.2},
        {"Chr": "chr1", "Start": 260, "End": 300, "sample1": 2.3},
    ])


def _highres_df(a_c_segments):
    rows = [
        {"Chr": "chr1", "Start": start, "End": end, "sample1": 2.0}
        for start, end in a_c_segments
    ]
    rows.extend([
        {"Chr": "chr1", "Start": 200, "End": 230, "sample1": 2.1},
        {"Chr": "chr1", "Start": 230, "End": 260, "sample1": 2.2},
        {"Chr": "chr1", "Start": 260, "End": 300, "sample1": 2.3},
    ])
    return pd.DataFrame(rows)


def _chr_x_flank_df():
    return pd.DataFrame([
        {"Chr": "chrX", "Start": 0, "End": 20, "sample1": 2.0},
        {"Chr": "chrX", "Start": 20, "End": 40, "sample1": 2.0},
        {"Chr": "chrX", "Start": 80, "End": 100, "sample1": 2.0},
        {"Chr": "chrX", "Start": 100, "End": 150, "sample1": 2.0},
        {"Chr": "chrX", "Start": 150, "End": 200, "sample1": 2.0},
        {"Chr": "chrX", "Start": 200, "End": 220, "sample1": 2.0},
        {"Chr": "chrX", "Start": 220, "End": 240, "sample1": 2.0},
        {"Chr": "chrX", "Start": 240, "End": 260, "sample1": 2.0},
        {"Chr": "chrX", "Start": 260, "End": 280, "sample1": 2.0},
        {"Chr": "chrX", "Start": 280, "End": 300, "sample1": 2.0},
    ])


def _collect_with_highres(monkeypatch, highres_df):
    locus = _FakeLocus()
    filter_max_values = _install_collect_highres_stubs(monkeypatch, highres_df)
    result = preprocess_module.collect_all_locus_bins(
        _lowres_df(),
        _FakeGDTable(locus),
        None,
        min_bins_per_interval=3,
        max_bins_per_interval=3,
        highres_counts_path="highres.tsv.gz",
        column_medians=np.array([1.0]),
        lowres_median_bin_size=100.0,
        filter_params=None,
        min_rebin_coverage=0.5,
    )
    return result, filter_max_values


def test_validate_supported_loci_rejects_chr_y(tmp_path):
    gd_path = tmp_path / "gd_table.tsv"
    gd_path.write_text(
        "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\tcluster\tBP1\tBP2\n"
        "chrY\t100\t200\tGDY\tDEL\tyes\tno\tcluster_y\t1\t2\n"
    )

    gd_table = GDTable(str(gd_path))

    with pytest.raises(ValueError, match="chrY"):
        validate_gd_table_for_preprocess(gd_table)


def test_write_preprocessed_baf_excludes_constant_samples(tmp_path, monkeypatch):
    class FakeHit:
        def __init__(self, data):
            self.data = data

    class FakeIntervalTree:
        def __init__(self):
            self._intervals = []

        def addi(self, start, end, data):
            self._intervals.append((start, end, data))

        def at(self, pos):
            return [
                FakeHit(data)
                for start, end, data in self._intervals
                if start <= pos < end
            ]

    class FakeTabixFile:
        def __init__(self, path):
            self.path = path

        def fetch(self, chrom, start, end):
            if (chrom, start, end) != ("chr1", 0, 100):
                return iter(())
            return iter([
                "chr1\t10\t0.5\tflat_sample",
                "chr1\t20\t0.5\tflat_sample",
                "chr1\t30\t0.25\tgood_sample",
                "chr1\t40\t0.75\tgood_sample",
            ])

        def close(self):
            return None

    monkeypatch.setattr(preprocess_module, "IntervalTree", FakeIntervalTree)
    monkeypatch.setattr(preprocess_module.pysam, "TabixFile", FakeTabixFile)

    baf_path = tmp_path / "input_baf.tsv"
    baf_path.write_text("chr1\t10\t0.5\tflat_sample\n")

    mappings = [
        preprocess_module.LocusBinMapping(
            cluster="cluster1",
            locus=None,
            interval_name="body",
            array_idx=0,
            chrom="chr1",
            start=0,
            end=100,
        )
    ]

    preprocess_module.write_preprocessed_baf(str(baf_path), mappings, str(tmp_path))

    filtered_baf_df = pd.read_csv(
        tmp_path / "preprocessed_baf.tsv.gz",
        sep="\t",
        compression="gzip",
    )
    summary_df = pd.read_csv(
        tmp_path / "preprocessed_baf_summary.tsv.gz",
        sep="\t",
        compression="gzip",
    )

    assert set(filtered_baf_df["Sample"]) == {"good_sample"}
    assert set(summary_df["sample"]) == {"good_sample"}
    assert summary_df["baf_n_sites"].tolist() == [2]
    assert summary_df["minor_baf_median"].tolist() == pytest.approx([0.25])


def test_write_and_load_normalization_metadata_round_trip(tmp_path):
    pd.DataFrame(
        [{"Chr": "chr1", "Start": 100, "End": 200, "sample1": 2.0}]
    ).to_csv(
        tmp_path / "preprocessed_bins.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(
        [{
            "cluster": "cluster1",
            "interval": "A-C",
            "array_idx": 0,
            "chr": "chr1",
            "start": 100,
            "end": 200,
        }]
    ).to_csv(
        tmp_path / "bin_mappings.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    metadata_df = preprocess_module.build_normalization_metadata(
        ["sample1", "sample2"],
        np.asarray([2000.0, 2500.0], dtype=np.float64),
        10000.0,
    )
    preprocess_module.write_normalization_metadata(metadata_df, str(tmp_path))

    combined_df, mappings, baf_summary_df, loaded_metadata_df = preprocess_module.load_preprocessed_data(
        str(tmp_path)
    )

    assert combined_df.shape == (1, 4)
    assert len(mappings) == 1
    assert baf_summary_df is None
    assert loaded_metadata_df is not None
    assert loaded_metadata_df["sample"].tolist() == ["sample1", "sample2"]
    assert loaded_metadata_df["raw_count_median"].tolist() == pytest.approx([2000.0, 2500.0])
    assert loaded_metadata_df["reference_bin_size"].tolist() == pytest.approx([10000.0, 10000.0])


def test_select_highres_interval_replacements_returns_only_improved_intervals():
    undercovered = [("A-C", 5), ("C-D", 8), ("D-E", 3)]
    hr_interval_bins = {
        "A-C": list(range(12)),
        "C-D": list(range(8)),
        "D-E": [0, 1],
    }

    replacements = preprocess_module._select_highres_interval_replacements(
        undercovered,
        hr_interval_bins,
    )

    assert replacements == [("A-C", 5, 12)]


def test_merge_highres_interval_replacements_replaces_only_target_interval():
    locus = _FakeLocus()
    flank_regions = []

    lowres_df = pd.DataFrame([
        {"Chr": "chr1", "Start": 100, "End": 150, "sample1": 2.0},
        {"Chr": "chr1", "Start": 150, "End": 200, "sample1": 2.1},
        {"Chr": "chr1", "Start": 200, "End": 250, "sample1": 2.2},
        {"Chr": "chr1", "Start": 250, "End": 300, "sample1": 2.3},
    ])
    hr_df = pd.DataFrame([
        {"Chr": "chr1", "Start": 100, "End": 125, "sample1": 1.9},
        {"Chr": "chr1", "Start": 125, "End": 150, "sample1": 2.0},
        {"Chr": "chr1", "Start": 150, "End": 175, "sample1": 2.1},
        {"Chr": "chr1", "Start": 175, "End": 200, "sample1": 2.2},
        {"Chr": "chr1", "Start": 200, "End": 250, "sample1": 2.2},
        {"Chr": "chr1", "Start": 250, "End": 300, "sample1": 2.3},
    ])

    interval_bins = preprocess_module.assign_bins_to_intervals(lowres_df, locus, flank_regions)
    interval_bins.pop("breakpoint_ranges", None)
    hr_interval_bins = preprocess_module.assign_bins_to_intervals(hr_df, locus, flank_regions)
    hr_interval_bins.pop("breakpoint_ranges", None)

    merged_df, merged_interval_bins = preprocess_module._merge_highres_interval_replacements(
        lowres_df,
        hr_df,
        interval_bins,
        hr_interval_bins,
        ["A-C"],
        locus,
        flank_regions,
    )

    assert len(merged_interval_bins["A-C"]) == 4
    assert len(merged_interval_bins["C-D"]) == 2
    assert merged_df[["Start", "End"]].values.tolist() == [
        [100, 125],
        [125, 150],
        [150, 175],
        [175, 200],
        [200, 250],
        [250, 300],
    ]


def test_collect_all_locus_bins_excludes_par_from_flanks():
    locus = _ChrXLocus()
    par_mask = _IntervalMask("chrX", 240, 320)

    _, mappings, _ = preprocess_module.collect_all_locus_bins(
        _chr_x_flank_df(),
        _FakeGDTable(locus),
        None,
        par_mask=par_mask,
        min_bins_per_interval=1,
        max_bins_per_interval=0,
        min_flank_bases=100,
        min_flank_bins=1,
        min_flank_coverage=0.0,
    )

    right_flank = [m for m in mappings if m.interval_name == "right_flank"]

    assert [(m.start, m.end) for m in right_flank] == [(200, 220), (220, 240)]
    assert all(not (m.start < 320 and m.end > 240) for m in right_flank)


def test_filter_and_prepare_locus_bins_excludes_par_from_flanks():
    locus = _ChrXLocus()
    par_mask = _IntervalMask("chrX", 240, 320)

    processed_df, interval_bins = preprocess_module._filter_and_prepare_locus_bins(
        _chr_x_flank_df(),
        locus,
        [(0, 100, "left_flank"), (200, 300, "right_flank")],
        0,
        300,
        0,
        par_mask=par_mask,
    )

    retained = list(zip(processed_df["Start"].tolist(), processed_df["End"].tolist()))

    assert (240, 260) not in retained
    assert (260, 280) not in retained
    assert (280, 300) not in retained
    assert len(interval_bins["right_flank"]) == 2


def test_collect_all_locus_bins_retries_highres_with_expanded_rebin(monkeypatch):
    highres_df = _highres_df([
        (100, 110),
        (120, 130),
        (140, 150),
        (160, 170),
        (180, 190),
    ])

    (_, mappings, _), filter_max_values = _collect_with_highres(monkeypatch, highres_df)

    interval_counts = {}
    for mapping in mappings:
        interval_counts[mapping.interval_name] = interval_counts.get(mapping.interval_name, 0) + 1

    assert filter_max_values == [3, 30]
    assert interval_counts == {"A-C": 5, "C-D": 3}


def test_collect_all_locus_bins_errors_when_expanded_highres_rebin_still_undercovered(monkeypatch):
    highres_df = _highres_df([
        (100, 110),
        (180, 190),
    ])

    with pytest.raises(ValueError, match="including high-res fallback"):
        _collect_with_highres(monkeypatch, highres_df)
