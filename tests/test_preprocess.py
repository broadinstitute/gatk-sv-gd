import numpy as np
import pandas as pd
import pytest
import sys
from types import SimpleNamespace
from types import SimpleNamespace

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

    def get_overlap_fraction(self, chrom, start, end):
        if chrom != self.chrom:
            return 0.0
        overlap = max(0, min(end, self.end) - max(start, self.start))
        length = max(1, end - start)
        return overlap / length

    def get_overlap_fractions_batch(self, chrom, starts, ends):
        if chrom != self.chrom:
            return np.zeros(len(starts))
        starts = np.asarray(starts)
        ends = np.asarray(ends)
        overlap = np.maximum(0, np.minimum(ends, self.end) - np.maximum(starts, self.start))
        lengths = np.maximum(1, ends - starts)
        return overlap / lengths


def test_region_parsing_helpers_cover_valid_and_invalid_inputs():
    assert preprocess_module._parse_region("chr1") == ("chr1", None, None)
    assert preprocess_module._parse_region("chr2:1,000-2,500") == ("chr2", 1000, 2500)

    with pytest.raises(ValueError, match="expected chrom:start-end"):
        preprocess_module._parse_region("chr1:100")

    with pytest.raises(ValueError, match="Invalid region coordinates"):
        preprocess_module._parse_region("chr1:start-end")

    assert preprocess_module._is_chr_x(" chrX ") is True
    assert preprocess_module._is_chr_x("X") is True
    assert preprocess_module._is_chr_x("chr1") is False
    assert preprocess_module._is_chr_y(" chrY ") is True
    assert preprocess_module._is_chr_y("Y") is True
    assert preprocess_module._is_chr_y("chr2") is False
    assert preprocess_module._flatten_multi_args([["a", "b"], ["c"], []]) == ["a", "b", "c"]


def test_parse_args_accepts_expected_preprocess_cli_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gatk-sv-gd",
            "--input",
            "depth.tsv.gz",
            "--gd-table",
            "gd.tsv",
            "--output-dir",
            "out",
            "--exclusion-intervals",
            "exc1.bed",
            "exc2.bed",
            "--flank-exclusion-intervals",
            "flank.bed",
            "--par-intervals",
            "par.bed",
            "--hard-inclusion-intervals",
            "hard.bed",
            "--high-res-counts",
            "highres.tsv.gz",
            "--baf-table",
            "baf.tsv.gz",
            "--ref-fasta",
            "ref.fasta.gz",
            "--locus-padding",
            "5000",
            "--exclusion-threshold",
            "0.2",
            "--exclusion-bypass-threshold",
            "0.9",
            "--min-bins-per-interval",
            "5",
            "--max-bins-per-interval",
            "8",
            "--min-rebin-coverage",
            "0.7",
            "--min-flank-bases",
            "1000",
            "--max-flank-bases",
            "5000",
            "--min-flank-bins",
            "4",
            "--min-flank-coverage",
            "0.6",
            "--region",
            "chr1:100-200",
            "--region",
            "chrX",
            "--median-min",
            "0.8",
            "--median-max",
            "3.5",
            "--mad-max",
            "0.3",
            "--verbose",
        ],
    )

    args = preprocess_module.parse_args()

    assert args.input == "depth.tsv.gz"
    assert args.gd_table == "gd.tsv"
    assert args.output_dir == "out"
    assert args.exclusion_intervals == [["exc1.bed", "exc2.bed"]]
    assert args.flank_exclusion_intervals == [["flank.bed"]]
    assert args.par_intervals == [["par.bed"]]
    assert args.hard_inclusion_intervals == [["hard.bed"]]
    assert args.high_res_counts == "highres.tsv.gz"
    assert args.baf_table == "baf.tsv.gz"
    assert args.ref_fasta == "ref.fasta.gz"
    assert args.locus_padding == 5000
    assert args.exclusion_threshold == pytest.approx(0.2)
    assert args.exclusion_bypass_threshold == pytest.approx(0.9)
    assert args.min_bins_per_interval == 5
    assert args.max_bins_per_interval == 8
    assert args.min_rebin_coverage == pytest.approx(0.7)
    assert args.min_flank_bases == 1000
    assert args.max_flank_bases == 5000
    assert args.min_flank_bins == 4
    assert args.min_flank_coverage == pytest.approx(0.6)
    assert args.regions == ["chr1:100-200", "chrX"]
    assert args.median_min == pytest.approx(0.8)
    assert args.median_max == pytest.approx(3.5)
    assert args.mad_max == pytest.approx(0.3)
    assert args.verbose is True


def test_locus_overlaps_regions_matches_whole_chromosome_and_interval_overlap():
    locus = _FakeLocus()

    assert preprocess_module._locus_overlaps_regions(
        locus,
        [("chr1", None, None)],
    ) is True
    assert preprocess_module._locus_overlaps_regions(
        locus,
        [("chr1", 250, 400)],
    ) is True
    assert preprocess_module._locus_overlaps_regions(
        locus,
        [("chr1", 300, 400)],
    ) is False
    assert preprocess_module._locus_overlaps_regions(
        locus,
        [("chr2", 0, 500)],
    ) is False


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


def test_filter_and_prepare_locus_bins_applies_masks_quality_filters_and_breakpoint_drops(monkeypatch):
    locus_df = pd.DataFrame(
        [
            {"Chr": "chrX", "Start": 80, "End": 100, "sample1": 2.0},
            {"Chr": "chrX", "Start": 100, "End": 120, "sample1": 2.1},
            {"Chr": "chrX", "Start": 160, "End": 180, "sample1": 2.2},
            {"Chr": "chrX", "Start": 180, "End": 200, "sample1": 2.3},
            {"Chr": "chrX", "Start": 200, "End": 220, "sample1": 2.4},
        ]
    )
    rebin_calls = []
    assign_calls = []
    quality_calls = []

    def fake_compute_bin_quality_mask(sub_df, **kwargs):
        quality_calls.append((sub_df[["Start", "End"]].copy(), kwargs))
        if len(quality_calls) == 1:
            return np.asarray([True, False]), {"filtered": 1, "par_ignored": 0}
        return np.asarray([True]), {"filtered": 0, "par_ignored": 1}

    def fake_rebin_locus_intervals(df, locus, max_bins_per_interval, flank_regions, min_rebin_coverage):
        rebin_calls.append((df[["Start", "End"]].copy(), max_bins_per_interval, min_rebin_coverage))
        return df.copy()

    def fake_assign_bins_to_intervals(df, locus, flank_regions):
        assign_calls.append(df[["Start", "End"]].copy())
        return {"body": [0], "flank": [1], "breakpoint_ranges": [0]}

    monkeypatch.setattr(preprocess_module, "compute_bin_quality_mask", fake_compute_bin_quality_mask)
    monkeypatch.setattr(preprocess_module, "get_flank_filter_params", lambda params, chrom: {"median_min": 0.5, "median_max": 3.5, "mad_max": 0.4})
    monkeypatch.setattr(preprocess_module, "rebin_locus_intervals", fake_rebin_locus_intervals)
    monkeypatch.setattr(preprocess_module, "assign_bins_to_intervals", fake_assign_bins_to_intervals)

    filtered_df, interval_bins = preprocess_module._filter_and_prepare_locus_bins(
        locus_df,
        _ChrXLocus(),
        flank_regions=[(0, 100), (200, 300)],
        left_bound=75,
        right_bound=225,
        max_bins_per_interval=4,
        exclusion_mask=_IntervalMask("chrX", 90, 175),
        flank_exclusion_mask=_IntervalMask("chrX", 80, 100),
        exclusion_threshold=0.2,
        filter_params={"median_min": 1.0, "median_max": 3.0, "mad_max": 0.5},
        exclusion_bypass_regions=[(95, 125)],
        min_rebin_coverage=0.6,
        ploidy_map={("sample1", "chrX"): 2},
        par_mask=_IntervalMask("chrX", 205, 215),
        hard_inclusion_mask=_IntervalMask("chrX", 80, 100),
    )

    assert filtered_df[["Start", "End"]].values.tolist() == [[100, 120]]
    assert interval_bins == {"body": [0], "flank": [1]}
    assert quality_calls[0][0]["Start"].tolist() == [100, 180]
    assert quality_calls[1][0]["Start"].tolist() == [80]
    assert quality_calls[1][1]["median_min"] == pytest.approx(0.5)
    assert rebin_calls[0][0]["Start"].tolist() == [80, 100]
    assert rebin_calls[0][1:] == (4, 0.6)
    assert assign_calls[0]["Start"].tolist() == [80, 100]


def test_filter_and_prepare_locus_bins_returns_empty_when_masks_remove_everything(monkeypatch):
    monkeypatch.setattr(
        preprocess_module,
        "compute_bin_quality_mask",
        lambda *args, **kwargs: pytest.fail("quality filtering should not run when all bins are removed earlier"),
    )
    monkeypatch.setattr(
        preprocess_module,
        "assign_bins_to_intervals",
        lambda *args, **kwargs: pytest.fail("interval assignment should not run for empty loci"),
    )

    empty_df, interval_bins = preprocess_module._filter_and_prepare_locus_bins(
        pd.DataFrame(
            [
                {"Chr": "chrX", "Start": 90, "End": 110, "sample1": 2.0},
                {"Chr": "chrX", "Start": 200, "End": 220, "sample1": 2.0},
            ]
        ),
        _ChrXLocus(),
        flank_regions=[(0, 100), (200, 300)],
        left_bound=80,
        right_bound=230,
        max_bins_per_interval=0,
        exclusion_mask=_IntervalMask("chrX", 80, 120),
        flank_exclusion_mask=None,
        exclusion_threshold=0.1,
        filter_params=None,
        exclusion_bypass_regions=None,
        min_rebin_coverage=0.5,
        ploidy_map=None,
        par_mask=_IntervalMask("chrX", 190, 230),
        hard_inclusion_mask=None,
    )

    assert empty_df.empty
    assert interval_bins == {}


def test_build_roi_intervals_from_mappings_merges_overlaps_per_chromosome():
    mappings = [
        preprocess_module.LocusBinMapping(
            cluster="c1",
            locus=None,
            interval_name="a",
            array_idx=0,
            chrom="chr1",
            start=100,
            end=150,
        ),
        preprocess_module.LocusBinMapping(
            cluster="c1",
            locus=None,
            interval_name="b",
            array_idx=1,
            chrom="chr1",
            start=140,
            end=200,
        ),
        preprocess_module.LocusBinMapping(
            cluster="c2",
            locus=None,
            interval_name="c",
            array_idx=2,
            chrom="chr1",
            start=250,
            end=300,
        ),
        preprocess_module.LocusBinMapping(
            cluster="c3",
            locus=None,
            interval_name="d",
            array_idx=3,
            chrom="chr2",
            start=50,
            end=75,
        ),
    ]

    assert preprocess_module._build_roi_intervals_from_mappings(mappings) == {
        "chr1": [(100, 200), (250, 300)],
        "chr2": [(50, 75)],
    }


def test_detect_baf_columns_handles_headerless_headered_empty_and_invalid_inputs(tmp_path):
    empty_path = tmp_path / "empty.tsv"
    empty_path.write_text("")
    assert preprocess_module._detect_baf_columns(str(empty_path)) == (None, ["Chr", "Pos", "BAF", "Sample"])

    headerless_path = tmp_path / "headerless.tsv"
    headerless_path.write_text("chr1\t10\t0.25\tsample1\n")
    assert preprocess_module._detect_baf_columns(str(headerless_path)) == (None, ["Chr", "Pos", "BAF", "Sample"])

    headered_path = tmp_path / "headered.tsv.gz"
    headered_path.write_bytes(b"")
    import gzip
    with gzip.open(headered_path, "wt") as handle:
        handle.write("Chr\tPos\tBAF\tSample\textra\nchr1\t10\t0.25\ts1\tfoo\n")
    assert preprocess_module._detect_baf_columns(str(headered_path)) == (0, ["Chr", "Pos", "BAF", "Sample"])

    invalid_path = tmp_path / "invalid.tsv"
    invalid_path.write_text("chr1\t10\t0.25\n")
    with pytest.raises(ValueError, match="at least 4 tab-delimited columns"):
        preprocess_module._detect_baf_columns(str(invalid_path))


def test_iter_roi_baf_records_skips_invalid_rows_and_fetch_errors(monkeypatch):
    class FakeTabixFile:
        def __init__(self, path):
            self.path = path

        def fetch(self, chrom, start, end):
            if chrom == "chr2":
                raise ValueError("missing contig")
            return iter([
                "Chr\tPos\tBAF\tSample",
                "chr1\t110\t0.25\ts1",
                "chr1\t120\tnot_a_float\ts1",
                "chr1\t130\t1.5\ts1",
                "chr1\t140\t0.75\ts2",
                "chr1\t150",
            ])

        def close(self):
            return None

    monkeypatch.setattr(preprocess_module.pysam, "TabixFile", FakeTabixFile)

    records = list(preprocess_module._iter_roi_baf_records(
        "fake.tsv.gz",
        {"chr1": [(100, 200)], "chr2": [(0, 50)]},
        header=0,
        column_names=["Chr", "Pos", "BAF", "Sample"],
    ))

    assert records == [
        ("chr1", 110, "110", 0.25, "0.25", "s1"),
        ("chr1", 140, "140", 0.75, "0.75", "s2"),
    ]


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
    assert summary_df["baf_effective_n_sites"].tolist() == [2]
    assert summary_df["minor_baf_median"].tolist() == pytest.approx([0.25])
    assert summary_df["baf_effective_variance"].tolist() == pytest.approx(
        summary_df["baf_variance"].tolist()
    )


def test_write_preprocessed_baf_computes_effective_site_counts_from_diploid_reference(tmp_path, monkeypatch):
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
            if (chrom, start, end) != ("chrX", 0, 30000):
                return iter(())
            return iter([
                "chrX\t1000\t0.25\tdip_a",
                "chrX\t11000\t0.75\tdip_a",
                "chrX\t21000\t0.30\tdip_a",
                "chrX\t1500\t0.20\tdip_b",
                "chrX\t11500\t0.80\tdip_b",
                "chrX\t21500\t0.35\tdip_b",
                "chrX\t1000\t0.40\thap_x",
                "chrX\t1200\t0.45\thap_x",
                "chrX\t1400\t0.60\thap_x",
            ])

        def close(self):
            return None

    monkeypatch.setattr(preprocess_module, "IntervalTree", FakeIntervalTree)
    monkeypatch.setattr(preprocess_module.pysam, "TabixFile", FakeTabixFile)

    baf_path = tmp_path / "input_baf.tsv"
    baf_path.write_text("chrX\t1000\t0.25\tdip_a\n")

    mappings = [
        preprocess_module.LocusBinMapping(
            cluster="cluster_x",
            locus=None,
            interval_name="body",
            array_idx=0,
            chrom="chrX",
            start=0,
            end=30000,
        )
    ]

    preprocess_module.write_preprocessed_baf(
        str(baf_path),
        mappings,
        str(tmp_path),
        ploidy_map={
            ("dip_a", "chrX"): 2,
            ("dip_b", "chrX"): 2,
            ("hap_x", "chrX"): 1,
        },
    )

    summary_df = pd.read_csv(
        tmp_path / "preprocessed_baf_summary.tsv.gz",
        sep="\t",
        compression="gzip",
    ).set_index("sample")

    assert summary_df.loc["dip_a", "baf_n_sites"] == 3
    assert summary_df.loc["dip_a", "baf_effective_n_sites"] == 3
    assert summary_df.loc["dip_b", "baf_n_sites"] == 3
    assert summary_df.loc["dip_b", "baf_effective_n_sites"] == 3
    assert summary_df.loc["hap_x", "baf_n_sites"] == 3
    assert summary_df.loc["hap_x", "baf_effective_n_sites"] == 1
    assert summary_df.loc["hap_x", "baf_effective_variance"] > summary_df.loc["hap_x", "baf_variance"]


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


def test_write_preprocessed_bins_persists_expected_table(tmp_path):
    combined_df = pd.DataFrame([
        {"Chr": "chr1", "Start": 100, "End": 200, "sample1": 2.0},
        {"Chr": "chr1", "Start": 200, "End": 300, "sample1": 2.1},
    ])

    output_path = preprocess_module.write_preprocessed_bins(combined_df, str(tmp_path))

    assert output_path.endswith("preprocessed_bins.tsv.gz")
    loaded_df = pd.read_csv(output_path, sep="\t", compression="gzip")
    pd.testing.assert_frame_equal(loaded_df, combined_df)


def test_get_locus_interval_bins_groups_only_requested_cluster():
    mappings = [
        preprocess_module.LocusBinMapping(
            cluster="cluster1",
            locus=None,
            interval_name="body",
            array_idx=2,
            chrom="chr1",
            start=100,
            end=150,
        ),
        preprocess_module.LocusBinMapping(
            cluster="cluster2",
            locus=None,
            interval_name="body",
            array_idx=7,
            chrom="chr1",
            start=150,
            end=200,
        ),
        preprocess_module.LocusBinMapping(
            cluster="cluster1",
            locus=None,
            interval_name="left_flank",
            array_idx=1,
            chrom="chr1",
            start=50,
            end=100,
        ),
    ]

    assert preprocess_module.get_locus_interval_bins(mappings, "cluster1") == {
        "body": [2],
        "left_flank": [1],
    }


def test_build_normalization_metadata_validates_inputs():
    with pytest.raises(ValueError, match="same length"):
        preprocess_module.build_normalization_metadata(
            ["sample1"],
            np.asarray([10.0, 20.0], dtype=np.float64),
            1000.0,
        )

    with pytest.raises(ValueError, match="must be positive"):
        preprocess_module.build_normalization_metadata(
            ["sample1"],
            np.asarray([10.0], dtype=np.float64),
            0.0,
        )

    with pytest.raises(ValueError, match="must all be positive"):
        preprocess_module.build_normalization_metadata(
            ["sample1", "sample2"],
            np.asarray([10.0, 0.0], dtype=np.float64),
            1000.0,
        )


def test_load_preprocessed_data_reads_optional_baf_summary(tmp_path):
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
    pd.DataFrame(
        [{"sample": "sample1", "baf_n_sites": 2, "baf_effective_n_sites": 2}]
    ).to_csv(
        tmp_path / "preprocessed_baf_summary.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    combined_df, mappings, baf_summary_df, normalization_metadata_df = preprocess_module.load_preprocessed_data(
        str(tmp_path)
    )

    assert combined_df.shape == (1, 4)
    assert len(mappings) == 1
    assert baf_summary_df is not None
    assert baf_summary_df["sample"].tolist() == ["sample1"]
    assert normalization_metadata_df is None


def test_main_runs_happy_path_and_writes_filtered_gd_table(tmp_path, monkeypatch):
    class FakeTableLocus:
        def __init__(self, breakpoints):
            self.breakpoints = breakpoints
            self.n_breakpoints = len(breakpoints)

    class FakeGDTable:
        def __init__(self, path):
            self.path = path
            self.loci = {
                "standalone-key": FakeTableLocus([(100, 100), (200, 200)]),
                "cluster_keep": FakeTableLocus([(300, 300), (400, 400), (500, 500)]),
            }
            self.df = pd.DataFrame([
                {
                    "cluster": np.nan,
                    "chr": "chr1",
                    "start_GRCh38": 100,
                    "end_GRCh38": 200,
                    "GD_ID": "GD_STANDALONE",
                    "svtype": "DEL",
                    "NAHR": "yes",
                    "terminal": "no",
                    "BP1": "1",
                    "BP2": "2",
                },
                {
                    "cluster": "cluster_keep",
                    "chr": "chr1",
                    "start_GRCh38": 300,
                    "end_GRCh38": 400,
                    "GD_ID": "GD_KEEP",
                    "svtype": "DUP",
                    "NAHR": "yes",
                    "terminal": "no",
                    "BP1": "2",
                    "BP2": "3",
                },
                {
                    "cluster": "cluster_drop",
                    "chr": "chr2",
                    "start_GRCh38": 500,
                    "end_GRCh38": 600,
                    "GD_ID": "GD_DROP",
                    "svtype": "DEL",
                    "NAHR": "yes",
                    "terminal": "no",
                    "BP1": "1",
                    "BP2": "2",
                },
            ])

        @staticmethod
        def _standalone_locus_key(row):
            return "standalone-key"

    args = SimpleNamespace(
        exclusion_intervals=[["exc1.bed"], ["exc2.bed"]],
        flank_exclusion_intervals=[["flank.bed"]],
        par_intervals=[["par.bed"]],
        hard_inclusion_intervals=[["hard.bed"]],
        verbose=True,
        output_dir=str(tmp_path),
        gd_table="gd_table.tsv",
        input="input.tsv.gz",
        median_min=1.0,
        median_max=3.0,
        mad_max=0.2,
        high_res_counts="highres.tsv.gz",
        regions=["chr1:100-300"],
        exclusion_threshold=0.25,
        locus_padding=50,
        min_bins_per_interval=3,
        max_bins_per_interval=5,
        exclusion_bypass_threshold=0.9,
        min_rebin_coverage=0.6,
        min_flank_bases=100,
        max_flank_bases=1000,
        min_flank_bins=2,
        min_flank_coverage=0.2,
        baf_table="baf.tsv.gz",
        ref_fasta=None,
    )

    mask_calls = []
    baf_calls = []
    metadata_calls = []

    monkeypatch.setattr(preprocess_module, "parse_args", lambda: args)
    monkeypatch.setattr(preprocess_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(preprocess_module, "GDTable", FakeGDTable)
    monkeypatch.setattr(preprocess_module, "validate_gd_table_for_preprocess", lambda gd_table: None)
    monkeypatch.setattr(
        preprocess_module,
        "ExclusionMask",
        lambda paths, label: mask_calls.append((tuple(paths), label)) or {"paths": tuple(paths), "label": label},
    )

    input_df = pd.DataFrame(
        {
            "Chr": ["chr1", "chr1"],
            "Start": [100, 200],
            "End": [150, 250],
            "sample1": [10.0, 30.0],
            "sample2": [20.0, 40.0],
        }
    )
    monkeypatch.setattr(preprocess_module, "read_data", lambda path: input_df.copy())
    monkeypatch.setattr(preprocess_module, "get_sample_columns", lambda df: ["sample1", "sample2"])
    monkeypatch.setattr(preprocess_module, "estimate_ploidy", lambda df, output_dir: pd.DataFrame([{"sample": "sample1"}]))
    monkeypatch.setattr(
        preprocess_module,
        "build_ploidy_map",
        lambda ploidy_df: {("sample1", "chr1"): 2, ("sample2", "chr1"): 2},
    )
    monkeypatch.setattr(preprocess_module, "filter_low_quality_bins", lambda df, **kwargs: df.copy())

    combined_df = pd.DataFrame(
        [{"Chr": "chr1", "Start": 100, "End": 150, "sample1": 1.0, "sample2": 2.0}]
    )
    mappings = [
        preprocess_module.LocusBinMapping(
            cluster="standalone-key",
            locus=None,
            interval_name="body",
            array_idx=0,
            chrom="chr1",
            start=100,
            end=150,
        )
    ]
    monkeypatch.setattr(
        preprocess_module,
        "collect_all_locus_bins",
        lambda *args, **kwargs: (combined_df.copy(), mappings, {"standalone-key": object(), "cluster_keep": object()}),
    )
    monkeypatch.setattr(preprocess_module, "compute_gc_fractions", lambda df, path: np.zeros(len(df), dtype=np.float32))
    monkeypatch.setattr(
        preprocess_module,
        "build_normalization_metadata",
        lambda sample_ids, column_medians, reference_bin_size: metadata_calls.append(
            (list(sample_ids), column_medians.tolist(), reference_bin_size)
        ) or pd.DataFrame([{"sample": "sample1", "raw_count_median": 20.0, "reference_bin_size": 50.0}]),
    )
    monkeypatch.setattr(preprocess_module, "write_preprocessed_bins", lambda *args, **kwargs: str(tmp_path / "preprocessed_bins.tsv.gz"))
    monkeypatch.setattr(preprocess_module, "write_normalization_metadata", lambda *args, **kwargs: str(tmp_path / "normalization_metadata.tsv"))
    monkeypatch.setattr(preprocess_module, "write_locus_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        preprocess_module,
        "write_preprocessed_baf",
        lambda baf_path, mappings_arg, output_dir, ploidy_map=None: baf_calls.append(
            (baf_path, len(mappings_arg), output_dir, dict(ploidy_map))
        ) or str(tmp_path / "preprocessed_baf.tsv.gz"),
    )

    preprocess_module.main()

    assert preprocess_module._util.VERBOSE is True
    assert mask_calls == [
        (("exc1.bed", "exc2.bed"), "exclusion regions"),
        (("flank.bed",), "flank exclusion regions"),
        (("par.bed",), "pseudoautosomal intervals"),
        (("hard.bed",), "hard inclusion regions"),
    ]
    assert metadata_calls == [(["sample1", "sample2"], [20.0, 30.0], 50.0)]
    assert baf_calls == [(
        "baf.tsv.gz",
        1,
        str(tmp_path),
        {("sample1", "chr1"): 2, ("sample2", "chr1"): 2},
    )]

    filtered_gd_df = pd.read_csv(tmp_path / "gd_table_filtered.tsv", sep="\t")
    assert filtered_gd_df["GD_ID"].tolist() == ["GD_STANDALONE", "GD_KEEP"]


def test_main_requires_par_intervals_when_chr_x_bins_are_present(tmp_path, monkeypatch):
    class FakeGDTable:
        def __init__(self, path):
            self.loci = {"cluster1": SimpleNamespace(breakpoints=[(1, 1)], n_breakpoints=1)}

    args = SimpleNamespace(
        exclusion_intervals=[],
        flank_exclusion_intervals=[],
        par_intervals=[],
        hard_inclusion_intervals=[],
        verbose=False,
        output_dir=str(tmp_path),
        gd_table="gd_table.tsv",
        input="input.tsv.gz",
        median_min=1.0,
        median_max=3.0,
        mad_max=0.2,
        high_res_counts=None,
        regions=None,
        exclusion_threshold=0.25,
        locus_padding=0,
        min_bins_per_interval=1,
        max_bins_per_interval=0,
        exclusion_bypass_threshold=1.0,
        min_rebin_coverage=0.5,
        min_flank_bases=10,
        max_flank_bases=100,
        min_flank_bins=1,
        min_flank_coverage=0.0,
        baf_table=None,
        ref_fasta=None,
    )

    monkeypatch.setattr(preprocess_module, "parse_args", lambda: args)
    monkeypatch.setattr(preprocess_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(preprocess_module, "GDTable", FakeGDTable)
    monkeypatch.setattr(preprocess_module, "validate_gd_table_for_preprocess", lambda gd_table: None)
    monkeypatch.setattr(
        preprocess_module,
        "read_data",
        lambda path: pd.DataFrame(
            {"Chr": ["chrX"], "Start": [100], "End": [150], "sample1": [10.0]}
        ),
    )
    monkeypatch.setattr(preprocess_module, "get_sample_columns", lambda df: ["sample1"])

    with pytest.raises(ValueError, match="--par-intervals"):
        preprocess_module.main()


def test_main_rejects_empty_preprocessed_output(tmp_path, monkeypatch):
    class FakeGDTable:
        def __init__(self, path):
            self.loci = {"cluster1": SimpleNamespace(breakpoints=[(1, 1)], n_breakpoints=1)}

    args = SimpleNamespace(
        exclusion_intervals=[],
        flank_exclusion_intervals=[],
        par_intervals=[["par.bed"]],
        hard_inclusion_intervals=[],
        verbose=False,
        output_dir=str(tmp_path),
        gd_table="gd_table.tsv",
        input="input.tsv.gz",
        median_min=1.0,
        median_max=3.0,
        mad_max=0.2,
        high_res_counts=None,
        regions=None,
        exclusion_threshold=0.25,
        locus_padding=0,
        min_bins_per_interval=1,
        max_bins_per_interval=0,
        exclusion_bypass_threshold=1.0,
        min_rebin_coverage=0.5,
        min_flank_bases=10,
        max_flank_bases=100,
        min_flank_bins=1,
        min_flank_coverage=0.0,
        baf_table=None,
        ref_fasta=None,
    )

    monkeypatch.setattr(preprocess_module, "parse_args", lambda: args)
    monkeypatch.setattr(preprocess_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(preprocess_module, "GDTable", FakeGDTable)
    monkeypatch.setattr(preprocess_module, "validate_gd_table_for_preprocess", lambda gd_table: None)
    monkeypatch.setattr(preprocess_module, "ExclusionMask", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        preprocess_module,
        "read_data",
        lambda path: pd.DataFrame(
            {"Chr": ["chr1"], "Start": [100], "End": [150], "sample1": [10.0]}
        ),
    )
    monkeypatch.setattr(preprocess_module, "get_sample_columns", lambda df: ["sample1"])
    monkeypatch.setattr(preprocess_module, "estimate_ploidy", lambda df, output_dir: pd.DataFrame([{"sample": "sample1"}]))
    monkeypatch.setattr(preprocess_module, "build_ploidy_map", lambda ploidy_df: {("sample1", "chr1"): 2})
    monkeypatch.setattr(preprocess_module, "filter_low_quality_bins", lambda df, **kwargs: df.copy())
    monkeypatch.setattr(
        preprocess_module,
        "collect_all_locus_bins",
        lambda *args, **kwargs: (pd.DataFrame(columns=["Chr", "Start", "End", "sample1"]), [], {}),
    )

    with pytest.raises(RuntimeError, match="No loci survived preprocessing"):
        preprocess_module.main()


def test_main_without_baf_table_skips_baf_outputs_and_reports_seven_tables(tmp_path, monkeypatch, capsys):
    class FakeGDTable:
        def __init__(self, path):
            self.loci = {"cluster_keep": SimpleNamespace(breakpoints=[(1, 1)], n_breakpoints=1)}
            self.df = pd.DataFrame([
                {
                    "cluster": "cluster_keep",
                    "chr": "chr1",
                    "start_GRCh38": 100,
                    "end_GRCh38": 200,
                    "GD_ID": "GD_KEEP",
                    "svtype": "DEL",
                    "NAHR": "yes",
                    "terminal": "no",
                    "BP1": "1",
                    "BP2": "2",
                },
            ])

        @staticmethod
        def _standalone_locus_key(row):
            return "unused"

    args = SimpleNamespace(
        exclusion_intervals=[],
        flank_exclusion_intervals=[],
        par_intervals=[["par.bed"]],
        hard_inclusion_intervals=[],
        verbose=False,
        output_dir=str(tmp_path),
        gd_table="gd_table.tsv",
        input="input.tsv.gz",
        median_min=1.0,
        median_max=3.0,
        mad_max=0.2,
        high_res_counts=None,
        regions=None,
        exclusion_threshold=0.25,
        locus_padding=0,
        min_bins_per_interval=1,
        max_bins_per_interval=0,
        exclusion_bypass_threshold=1.0,
        min_rebin_coverage=0.5,
        min_flank_bases=10,
        max_flank_bases=100,
        min_flank_bins=1,
        min_flank_coverage=0.0,
        baf_table=None,
        ref_fasta=None,
    )

    monkeypatch.setattr(preprocess_module, "parse_args", lambda: args)
    monkeypatch.setattr(preprocess_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(preprocess_module, "GDTable", FakeGDTable)
    monkeypatch.setattr(preprocess_module, "validate_gd_table_for_preprocess", lambda gd_table: None)
    monkeypatch.setattr(preprocess_module, "ExclusionMask", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        preprocess_module,
        "read_data",
        lambda path: pd.DataFrame({"Chr": ["chr1"], "Start": [100], "End": [150], "sample1": [10.0]}),
    )
    monkeypatch.setattr(preprocess_module, "get_sample_columns", lambda df: ["sample1"])
    monkeypatch.setattr(preprocess_module, "estimate_ploidy", lambda df, output_dir: pd.DataFrame([{"sample": "sample1"}]))
    monkeypatch.setattr(preprocess_module, "build_ploidy_map", lambda ploidy_df: {("sample1", "chr1"): 2})
    monkeypatch.setattr(preprocess_module, "filter_low_quality_bins", lambda df, **kwargs: df.copy())
    monkeypatch.setattr(
        preprocess_module,
        "collect_all_locus_bins",
        lambda *args, **kwargs: (
            pd.DataFrame([{"Chr": "chr1", "Start": 100, "End": 150, "sample1": 2.0}]),
            [
                preprocess_module.LocusBinMapping(
                    cluster="cluster_keep",
                    locus=None,
                    interval_name="body",
                    array_idx=0,
                    chrom="chr1",
                    start=100,
                    end=150,
                )
            ],
            {"cluster_keep": object()},
        ),
    )
    monkeypatch.setattr(preprocess_module, "compute_gc_fractions", lambda df, path: np.zeros(len(df), dtype=np.float32))
    monkeypatch.setattr(
        preprocess_module,
        "build_normalization_metadata",
        lambda *args, **kwargs: pd.DataFrame([{"sample": "sample1", "raw_count_median": 10.0, "reference_bin_size": 50.0}]),
    )
    monkeypatch.setattr(preprocess_module, "write_preprocessed_bins", lambda *args, **kwargs: str(tmp_path / "preprocessed_bins.tsv.gz"))
    monkeypatch.setattr(preprocess_module, "write_normalization_metadata", lambda *args, **kwargs: str(tmp_path / "normalization_metadata.tsv"))
    monkeypatch.setattr(preprocess_module, "write_locus_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        preprocess_module,
        "write_preprocessed_baf",
        lambda *args, **kwargs: pytest.fail("write_preprocessed_baf should not run when baf_table is absent"),
    )

    preprocess_module.main()

    stdout = capsys.readouterr().out
    assert "Output tables written: 7" in stdout
    assert "BAF outputs included" not in stdout


def test_collect_all_locus_bins_verbose_rebin_and_breakpoint_masking(monkeypatch, capsys):
    locus = _FakeLocus()
    locus.n_breakpoints = len(locus.breakpoints)

    monkeypatch.setattr(preprocess_module._util, "VERBOSE", True)
    monkeypatch.setattr(preprocess_module, "compute_flank_regions_from_bins", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        preprocess_module,
        "extract_locus_bins",
        lambda *args, **kwargs: pd.DataFrame([
            {"Chr": "chr1", "Start": 100, "End": 140, "sample1": 2.0},
            {"Chr": "chr1", "Start": 140, "End": 200, "sample1": 2.2},
            {"Chr": "chr1", "Start": 200, "End": 260, "sample1": 2.4},
        ]),
    )
    monkeypatch.setattr(
        preprocess_module,
        "rebin_locus_intervals",
        lambda df, locus_arg, max_bins_per_interval, flank_regions, min_rebin_coverage: pd.DataFrame([
            {"Chr": "chr1", "Start": 100, "End": 200, "sample1": 2.1},
            {"Chr": "chr1", "Start": 200, "End": 260, "sample1": 2.4},
        ]),
    )
    monkeypatch.setattr(
        preprocess_module,
        "assign_bins_to_intervals",
        lambda locus_df, locus_arg, flank_regions: {
            "A-C": [0],
            "C-D": [1],
            "breakpoint_ranges": [1],
        },
    )

    combined_df, mappings, included_loci = preprocess_module.collect_all_locus_bins(
        pd.DataFrame([
            {"Chr": "chr1", "Start": 100, "End": 140, "sample1": 2.0},
            {"Chr": "chr1", "Start": 140, "End": 200, "sample1": 2.2},
            {"Chr": "chr1", "Start": 200, "End": 260, "sample1": 2.4},
        ]),
        _FakeGDTable(locus),
        None,
        min_bins_per_interval=0,
        max_bins_per_interval=2,
        min_rebin_coverage=0.5,
    )

    stdout = capsys.readouterr().out
    assert "Bins after rebinning: 2 (reduced from 3)" in stdout
    assert "Masking 1 breakpoint-range bin(s)" in stdout
    assert list(zip(combined_df["Start"], combined_df["End"])) == [(100, 200)]
    assert [mapping.interval_name for mapping in mappings] == ["A-C"]
    assert set(included_loci) == {"cluster1"}


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


def test_filter_low_quality_bins_keeps_hard_included_bin():
    df = pd.DataFrame([
        {"Chr": "chr1", "Start": 0, "End": 100, "sample1": 0.2},
        {"Chr": "chr1", "Start": 100, "End": 200, "sample1": 2.0},
    ])

    filtered_df = preprocess_module.filter_low_quality_bins(
        df,
        median_min=1.0,
        median_max=3.0,
        mad_max=0.5,
        hard_inclusion_mask=_IntervalMask("chr1", 0, 100),
    )

    assert filtered_df[["Start", "End"]].values.tolist() == [[0, 100], [100, 200]]


def test_filter_and_prepare_locus_bins_keeps_hard_included_body_bin():
    locus = _FakeLocus()
    locus_df = pd.DataFrame([
        {"Chr": "chr1", "Start": 100, "End": 150, "sample1": 0.2},
        {"Chr": "chr1", "Start": 150, "End": 200, "sample1": 2.0},
        {"Chr": "chr1", "Start": 200, "End": 250, "sample1": 2.0},
        {"Chr": "chr1", "Start": 250, "End": 300, "sample1": 2.0},
    ])

    processed_df, interval_bins = preprocess_module._filter_and_prepare_locus_bins(
        locus_df,
        locus,
        [],
        100,
        300,
        0,
        exclusion_mask=_IntervalMask("chr1", 100, 150),
        exclusion_threshold=0.1,
        filter_params={"median_min": 1.0, "median_max": 3.0, "mad_max": 0.5},
        hard_inclusion_mask=_IntervalMask("chr1", 100, 150),
    )

    retained = list(zip(processed_df["Start"].tolist(), processed_df["End"].tolist()))

    assert (100, 150) in retained
    assert len(interval_bins["A-C"]) == 2


def test_collect_all_locus_bins_keeps_hard_included_par_flank_bin():
    locus = _ChrXLocus()
    par_mask = _IntervalMask("chrX", 240, 320)
    hard_inclusion_mask = _IntervalMask("chrX", 240, 260)

    _, mappings, _ = preprocess_module.collect_all_locus_bins(
        _chr_x_flank_df(),
        _FakeGDTable(locus),
        None,
        par_mask=par_mask,
        hard_inclusion_mask=hard_inclusion_mask,
        min_bins_per_interval=1,
        max_bins_per_interval=0,
        min_flank_bases=100,
        min_flank_bins=1,
        min_flank_coverage=0.0,
    )

    right_flank = [m for m in mappings if m.interval_name == "right_flank"]

    assert (240, 260) in [(m.start, m.end) for m in right_flank]
    assert (260, 280) not in [(m.start, m.end) for m in right_flank]


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


def test_collect_all_locus_bins_errors_when_highres_query_returns_no_bins(monkeypatch, capsys):
    locus = _FakeLocus()

    monkeypatch.setattr(preprocess_module, "compute_flank_regions_from_bins", lambda *args, **kwargs: [])
    monkeypatch.setattr(preprocess_module, "query_highres_bins", lambda *args, **kwargs: pd.DataFrame(columns=_lowres_df().columns))

    with pytest.raises(ValueError, match="fewer than --min-bins-per-interval=3 bins"):
        preprocess_module.collect_all_locus_bins(
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

    assert "[highres] no bins returned from tabix query" in capsys.readouterr().out


def test_collect_all_locus_bins_errors_when_highres_fallback_is_unavailable(monkeypatch, capsys):
    locus = _FakeLocus()
    highres_df = _highres_df([
        (100, 110),
        (180, 190),
    ])

    monkeypatch.setattr(preprocess_module, "compute_flank_regions_from_bins", lambda *args, **kwargs: [])
    monkeypatch.setattr(preprocess_module, "query_highres_bins", lambda *args, **kwargs: highres_df.copy())
    monkeypatch.setattr(preprocess_module, "normalize_highres_bins", lambda raw_df, *args, **kwargs: raw_df.copy())

    with pytest.raises(ValueError, match="including high-res fallback"):
        preprocess_module.collect_all_locus_bins(
            _lowres_df(),
            _FakeGDTable(locus),
            None,
            min_bins_per_interval=3,
            max_bins_per_interval=0,
            highres_counts_path="highres.tsv.gz",
            column_medians=np.array([1.0]),
            lowres_median_bin_size=100.0,
            filter_params=None,
            min_rebin_coverage=0.5,
        )

    assert "fallback rebin is not available because --max-bins-per-interval=0" in capsys.readouterr().out


def test_collect_all_locus_bins_uses_filtered_cache_and_skips_unassigned_bins(monkeypatch, capsys):
    locus = _FakeLocus()
    observed = {}

    input_df = pd.DataFrame([
        {"Chr": "chr1", "Start": 0, "End": 50, "sample1": 0.2},
        {"Chr": "chr1", "Start": 50, "End": 100, "sample1": 2.0},
        {"Chr": "chr1", "Start": 300, "End": 325, "sample1": 2.0},
        {"Chr": "chr1", "Start": 325, "End": 350, "sample1": 2.0},
        {"Chr": "chr1", "Start": 350, "End": 400, "sample1": 2.0},
    ])

    def fake_compute_bin_quality_mask(sub_df, **kwargs):
        observed["quality_positions"] = sub_df["Start"].tolist()
        keep = np.ones(len(sub_df), dtype=bool)
        keep[0] = False
        return keep, {"filtered": 1, "par_ignored": 0}

    def fake_compute_flank_regions_from_bins(chrom_bins, locus_arg, locus_size, **kwargs):
        observed["chrom_bins"] = chrom_bins[["Start", "End"]].values.tolist()
        observed["flank_filter_params"] = kwargs["filter_params"]
        return [(0, 100, "left_flank"), (300, 400, "right_flank")]

    def fake_extract_locus_bins(df, locus_arg, exclusion_mask, **kwargs):
        observed["exclusion_bypass_regions"] = kwargs["exclusion_bypass_regions"]
        return pd.DataFrame([
            {"Chr": "chr1", "Start": 100, "End": 150, "sample1": 2.0},
            {"Chr": "chr1", "Start": 200, "End": 250, "sample1": 2.0},
            {"Chr": "chr1", "Start": 500, "End": 520, "sample1": 2.0},
        ])

    def fake_assign_bins_to_intervals(locus_df, locus_arg, flank_regions):
        observed["combined_bins"] = locus_df[["Start", "End"]].values.tolist()
        return {
            "left_flank": [0],
            "A-C": [1],
            "C-D": [2],
            "right_flank": [3],
            "breakpoint_ranges": [],
        }

    gd_table = SimpleNamespace(
        get_all_loci=lambda: {
            "outside": SimpleNamespace(chrom="chr2", start=100, end=200, is_nahr=True),
            "skip_non_nahr": SimpleNamespace(chrom="chr1", start=110, end=290, is_nahr=False),
            "cluster1": locus,
        }
    )

    monkeypatch.setattr(preprocess_module, "compute_bin_quality_mask", fake_compute_bin_quality_mask)
    monkeypatch.setattr(preprocess_module, "compute_flank_regions_from_bins", fake_compute_flank_regions_from_bins)
    monkeypatch.setattr(preprocess_module, "extract_locus_bins", fake_extract_locus_bins)
    monkeypatch.setattr(preprocess_module, "assign_bins_to_intervals", fake_assign_bins_to_intervals)

    combined_df, mappings, included_loci = preprocess_module.collect_all_locus_bins(
        input_df,
        gd_table,
        regions=[("chr1", 0, 400)],
        exclusion_mask=_IntervalMask("chr1", 50, 200),
        flank_exclusion_mask=_IntervalMask("chr1", 350, 400),
        exclusion_threshold=0.1,
        filter_params={"median_min": 1.0, "median_max": 3.0, "mad_max": 0.5},
        exclusion_bypass_threshold=0.5,
        min_bins_per_interval=1,
        max_bins_per_interval=0,
        min_flank_bases=50,
        min_flank_bins=1,
        min_flank_coverage=0.0,
        ploidy_map={("sample1", "chr1"): 2},
        par_mask=_IntervalMask("chr1", 300, 350),
        hard_inclusion_mask=_IntervalMask("chr1", 50, 100),
    )

    captured = capsys.readouterr()
    assert observed["quality_positions"] == [0, 50, 350]
    assert observed["chrom_bins"] == [[50, 100], [350, 400]]
    assert observed["flank_filter_params"] == {"median_min": 1.5, "median_max": 2.5, "mad_max": 0.3}
    assert observed["exclusion_bypass_regions"] == [(100, 200)]
    assert observed["combined_bins"] == [[50, 100], [100, 150], [200, 250], [500, 520]]
    assert "WARNING: skipped one unassigned bin outside modeled intervals" in captured.out
    assert "Filtered cache built: 2/5 bins retained across 1 contigs" in captured.out
    assert "processed_NAHR=1, skipped_non_NAHR=1" in captured.out
    assert list(zip(combined_df["Start"], combined_df["End"])) == [(50, 100), (100, 150), (200, 250), (500, 520)]
    assert [mapping.interval_name for mapping in mappings] == ["left_flank", "A-C", "C-D"]
    assert set(included_loci) == {"cluster1"}


def test_compute_gc_fractions_computes_per_bin_gc(tmp_path, monkeypatch):
    """Test that compute_gc_fractions fetches sequences and computes GC fractions."""
    import pysam
    import subprocess

    # Create a temporary FASTA file
    fasta_path = str(tmp_path / "test.fa")
    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        f.write("AAAAAACCCCCCGGGGGG\n")
        f.write(">chr2\n")
        f.write("TTTTTTTT\n")

    # Index the FASTA (conftest stub is no-op, use samtools directly)
    pysam.faidx(fasta_path)
    subprocess.run(["samtools", "faidx", fasta_path], check=True, capture_output=True)

    # Replace stub FastaFile with samtools-backed implementation
    class SamtoolsFastaFile:
        """FastaFile stub that uses samtools faidx (1-based) for pysam (0-based)."""
        def __init__(self, path):
            self._path = path

        def fetch(self, chrom, start=None, end=None):
            # pysam uses 0-based, samtools uses 1-based
            if start is not None:
                region = f"{chrom}:{start + 1}-{end}"
            else:
                region = chrom
            result = subprocess.run(
                ["samtools", "faidx", self._path, region],
                capture_output=True, text=True, check=True,
            )
            lines = result.stdout.strip().split("\n")
            return lines[1] if len(lines) > 1 else lines[0]

        def close(self):
            pass

    monkeypatch.setattr(pysam, "FastaFile", SamtoolsFastaFile)

    combined_df = pd.DataFrame({
        "Chr": ["chr1", "chr1", "chr2"],
        "Start": [0, 6, 0],
        "End": [6, 12, 8],
    })

    gc_fractions = preprocess_module.compute_gc_fractions(combined_df, fasta_path)

    # chr1:0-6 = AAAAAA, GC = 0/6 = 0.0
    # chr1:6-12 = CCCCCC, GC = 6/6 = 1.0
    # chr2:0-8 = TTTTTTTT, GC = 0/8 = 0.0
    assert len(gc_fractions) == 3
    assert gc_fractions[0] == 0.0
    assert gc_fractions[1] == 1.0
    assert gc_fractions[2] == 0.0


def test_compute_gc_fractions_handles_missing_sequence(tmp_path, monkeypatch):
    """Test that compute_gc_fractions returns NaN for missing sequences."""
    import pysam
    import subprocess

    # Create a temporary FASTA file with only chr1
    fasta_path = str(tmp_path / "test.fa")
    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        f.write("AAAAAA\n")

    pysam.faidx(fasta_path)
    subprocess.run(["samtools", "faidx", fasta_path], check=True, capture_output=True)

    class SamtoolsFastaFile:
        """FastaFile stub that uses samtools faidx (1-based) for pysam (0-based)."""
        def __init__(self, path):
            self._path = path

        def fetch(self, chrom, start=None, end=None):
            # pysam uses 0-based, samtools uses 1-based
            if start is not None:
                region = f"{chrom}:{start + 1}-{end}"
            else:
                region = chrom
            result = subprocess.run(
                ["samtools", "faidx", self._path, region],
                capture_output=True, text=True, check=True,
            )
            lines = result.stdout.strip().split("\n")
            return lines[1] if len(lines) > 1 else lines[0]

        def close(self):
            pass

    monkeypatch.setattr(pysam, "FastaFile", SamtoolsFastaFile)

    combined_df = pd.DataFrame({
        "Chr": ["chr1", "chr_missing"],
        "Start": [0, 0],
        "End": [6, 6],
    })

    gc_fractions = preprocess_module.compute_gc_fractions(combined_df, fasta_path)

    assert len(gc_fractions) == 2
    assert gc_fractions[0] == 0.0  # AAAAAA
    assert np.isnan(gc_fractions[1])  # chr_missing not in FASTA


def test_compute_gc_fractions_prints_summary(capsys, monkeypatch, tmp_path):
    """Test that compute_gc_fractions prints a summary message."""
    import pysam
    import subprocess

    fasta_path = str(tmp_path / "test.fa")
    with open(fasta_path, "w") as f:
        f.write(">chr1\n")
        f.write("AAAAAA\n")

    pysam.faidx(fasta_path)
    subprocess.run(["samtools", "faidx", fasta_path], check=True, capture_output=True)

    class SamtoolsFastaFile:
        """FastaFile stub that uses samtools faidx (1-based) for pysam (0-based)."""
        def __init__(self, path):
            self._path = path

        def fetch(self, chrom, start=None, end=None):
            # pysam uses 0-based, samtools uses 1-based
            if start is not None:
                region = f"{chrom}:{start + 1}-{end}"
            else:
                region = chrom
            result = subprocess.run(
                ["samtools", "faidx", self._path, region],
                capture_output=True, text=True, check=True,
            )
            lines = result.stdout.strip().split("\n")
            return lines[1] if len(lines) > 1 else lines[0]

        def close(self):
            pass

    monkeypatch.setattr(pysam, "FastaFile", SamtoolsFastaFile)

    combined_df = pd.DataFrame({
        "Chr": ["chr1"],
        "Start": [0],
        "End": [6],
    })

    gc_fractions = preprocess_module.compute_gc_fractions(combined_df, fasta_path)
    captured = capsys.readouterr()
    assert "GC fractions computed for 1/1 bins" in captured.out
    assert len(gc_fractions) == 1
