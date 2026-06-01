import csv
from types import SimpleNamespace

import numpy as np
import pytest

from gatk_sv_gd import synthesize as synthesize_module
from gatk_sv_gd.models import GDLocus
from gatk_sv_gd.synthesize import (
    _BGZF_EOF,
    _background_conflicts_with_primary,
    _build_assignment_events,
    _build_baf_interval_map,
    _build_gd_lookup,
    _build_interval_map_from_events,
    _build_spike_index,
    _concat_bgzf_parts,
    _detect_baf_columns,
    _discover_sample_ids,
    _ensure_baf_tabix_index,
    _ensure_tabix_index,
    _gd_entry_overlaps_regions,
    _infer_chrom_name,
    _intervals_overlap,
    _load_ploidy_lookup,
    _load_truth_assignments,
    _make_synth_event,
    _matches_canonical_breakpoint_pair,
    _matches_canonical_gd_interval,
    _parse_region,
    _process_baf_contig_group,
    _process_contig_group,
    _read_baf_sample_ids,
    _read_count_sample_ids,
    _rewrite_baf_file,
    _rewrite_counts_file,
    _resolve_intervals,
    _resolve_event_multiplier,
    _resolve_sample_contig_ploidy,
    _select_sample_subset,
    _spike_baf_value,
    _write_background_event_table,
    _write_truth_table,
    assign_gd_to_samples,
    generate_salted_flank_bleed_events,
    generate_viable_trisomy_events,
)


class _ScriptedRng:
    def __init__(self, *, choices=None, integers=None, uniforms=None, randoms=None):
        self._choices = list(choices or [])
        self._integers = list(integers or [])
        self._uniforms = list(uniforms or [])
        self._randoms = list(randoms or [])

    def choice(self, values, size, replace=False):
        if self._choices:
            return np.array(self._choices.pop(0), dtype=object)
        return np.array(list(values)[:size], dtype=object)

    def integers(self, low, high=None, size=None):
        return self._integers.pop(0)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._uniforms.pop(0)

    def random(self, size=None):
        return self._randoms.pop(0)

    def shuffle(self, values):
        return None


class _StubTabixFile:
    def __init__(self, rows_by_contig, *, header=None, contigs=None, fail_contigs=None):
        self._rows_by_contig = rows_by_contig
        self.header = list(header or [])
        self.contigs = list(contigs or rows_by_contig)
        self._fail_contigs = set(fail_contigs or [])

    def fetch(self, contig):
        if contig in self._fail_contigs:
            raise ValueError("missing contig")
        return iter(self._rows_by_contig.get(contig, []))

    def close(self):
        return None


class _StubBGZFile:
    def __init__(self, path, mode):
        self._handle = open(path, "wb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._handle.close()

    def write(self, data):
        self._handle.write(data)


def _make_locus():
    return GDLocus(
        cluster="cluster1",
        chrom="chr1",
        breakpoints=[(100, 110), (200, 210), (300, 310)],
        breakpoint_names=["A", "B", "C"],
        gd_entries=[
            {
                "GD_ID": "GD1",
                "start_GRCh38": 110,
                "end_GRCh38": 300,
                "svtype": "DEL",
                "BP1": "A",
                "BP2": "C",
                "cluster": "cluster1",
            },
            {
                "GD_ID": "GD2",
                "start_GRCh38": 110,
                "end_GRCh38": 210,
                "svtype": "DUP",
                "BP1": "A",
                "BP2": "B",
                "cluster": "cluster1",
            },
        ],
        is_nahr=True,
        is_terminal=False,
    )


def test_parse_region_and_interval_helpers():
    assert _parse_region("chr1") == ("chr1", None, None)
    assert _parse_region("chr1:1,000-2,000") == ("chr1", 1000, 2000)

    with pytest.raises(ValueError, match="Invalid region format"):
        _parse_region("chr1:1000")

    with pytest.raises(ValueError, match="Invalid region coordinates"):
        _parse_region("chr1:start-end")

    entry = {"start_GRCh38": 150, "end_GRCh38": 250}
    assert _gd_entry_overlaps_regions(entry, "chr1", [("chr1", None, None)]) is True
    assert _gd_entry_overlaps_regions(entry, "chr1", [("chr1", 200, 300)]) is True
    assert _gd_entry_overlaps_regions(entry, "chr1", [("chr1", 250, 300)]) is False
    assert _gd_entry_overlaps_regions(entry, "chr1", [("chr2", None, None)]) is False

    assert _intervals_overlap(10, 20, 15, 25) is True
    assert _intervals_overlap(10, 20, 20, 30) is False


def test_sample_ploidy_and_event_multiplier_resolution():
    lookup = {
        ("S1", "chr1"): 3,
        ("S2", "1"): 1,
        ("S3", "chr2"): 0,
    }

    assert _resolve_sample_contig_ploidy("S1", "chr1", lookup) == 3
    assert _resolve_sample_contig_ploidy("S2", "chr1", lookup) == 1
    assert _resolve_sample_contig_ploidy("S3", "2", lookup) == 1
    assert _resolve_sample_contig_ploidy("S9", "chr9", lookup, default_ploidy=2) == 2
    assert _resolve_sample_contig_ploidy("S0", "chr1", None, default_ploidy=4) == 4

    assert _resolve_event_multiplier("DEL", 2, 0.4, 1.6) == 0.5
    assert _resolve_event_multiplier("DUP", 2, 0.4, 1.6) == 1.5
    assert _resolve_event_multiplier("DEL", 0, 0.4, 1.6) == 0.4
    assert _resolve_event_multiplier("DUP", -1, 0.4, 1.6) == 1.6
    assert _resolve_event_multiplier("CNV", 2, 0.4, 1.6) == 1.0


def test_canonical_matching_helpers():
    locus = _make_locus()

    assert _matches_canonical_gd_interval(locus, 110, 300) is True
    assert _matches_canonical_gd_interval(locus, 120, 300) is False
    assert _matches_canonical_breakpoint_pair(locus, "A", "C") is True
    assert _matches_canonical_breakpoint_pair(locus, "B", "C") is False


def test_build_gd_lookup_rejects_duplicate_ids():
    locus = _make_locus()
    gd_table = SimpleNamespace(get_all_loci=lambda: {"cluster1": locus})

    lookup = _build_gd_lookup(gd_table)

    assert lookup["GD1"][0] == "chr1"
    assert lookup["GD2"][1]["svtype"] == "DUP"

    dup_locus = GDLocus(
        cluster="cluster2",
        chrom="chr2",
        breakpoints=[(1, 2), (3, 4)],
        breakpoint_names=["1", "2"],
        gd_entries=[{"GD_ID": "GD1", "start_GRCh38": 2, "end_GRCh38": 3, "svtype": "DEL", "BP1": "1", "BP2": "2"}],
        is_nahr=True,
        is_terminal=False,
    )
    dup_table = SimpleNamespace(get_all_loci=lambda: {"cluster1": locus, "cluster2": dup_locus})

    with pytest.raises(ValueError, match="Duplicate GD_ID"):
        _build_gd_lookup(dup_table)


def test_truth_and_input_table_loaders(tmp_path):
    gd_lookup = {
        "GD1": ("chr1", {"GD_ID": "GD1", "start_GRCh38": 10, "end_GRCh38": 20}),
        "GD2": ("chr2", {"GD_ID": "GD2", "start_GRCh38": 30, "end_GRCh38": 40}),
    }
    truth_path = tmp_path / "truth.tsv"
    truth_path.write_text(
        "sample_id\tGD_ID\n"
        "S1\tGD1\n"
        "S1\tGD1\n"
        "S2\tGD2\n"
        "\tGD2\n"
    )

    assignments = _load_truth_assignments(str(truth_path), gd_lookup)
    assert assignments["S1"][1]["GD_ID"] == "GD1"
    assert assignments["S2"][0] == "chr2"

    bad_truth_path = tmp_path / "bad_truth.tsv"
    bad_truth_path.write_text("sample_id\tGD_ID\nS1\tGD1\nS1\tGD2\n")
    with pytest.raises(ValueError, match="multiple GD assignments"):
        _load_truth_assignments(str(bad_truth_path), gd_lookup)

    unknown_truth_path = tmp_path / "unknown_truth.tsv"
    unknown_truth_path.write_text("sample_id\tGD_ID\nS1\tGD9\n")
    with pytest.raises(ValueError, match="not found"):
        _load_truth_assignments(str(unknown_truth_path), gd_lookup)

    count_path = tmp_path / "counts.tsv"
    count_path.write_text("#Chr\tStart\tEnd\tBin\tS1\tS2\nchr1\t1\t10\t1\t5\t6\n")
    assert _read_count_sample_ids(str(count_path)) == ["S1", "S2"]
    assert _discover_sample_ids(str(count_path), None, None) == ["S1", "S2"]

    baf_with_header = tmp_path / "baf_header.tsv"
    baf_with_header.write_text("Chr\tPos\tBAF\tSample\nchr1\t10\t0.4\tS1\nchr1\t20\t0.6\tS2\n")
    assert _detect_baf_columns(str(baf_with_header)) == (0, ["Chr", "Pos", "BAF", "Sample"])
    assert _read_baf_sample_ids(str(baf_with_header)) == ["S1", "S2"]
    assert _discover_sample_ids(None, None, str(baf_with_header)) == ["S1", "S2"]

    baf_no_header = tmp_path / "baf_no_header.tsv"
    baf_no_header.write_text("chr1\t10\t0.4\tS3\nchr1\t20\t0.5\tS4\n")
    assert _detect_baf_columns(str(baf_no_header)) == (None, ["Chr", "Pos", "BAF", "Sample"])

    short_baf = tmp_path / "baf_short.tsv"
    short_baf.write_text("chr1\t10\t0.4\n")
    with pytest.raises(ValueError, match="at least 4"):
        _detect_baf_columns(str(short_baf))

    ploidy_path = tmp_path / "ploidy.tsv"
    ploidy_path.write_text(
        "sample\tcontig\tploidy\n"
        "S1\tchr1\t2\n"
        "S2\tchr2\t3.0\n"
        "S3\tchr3\t\n"
    )
    assert _load_ploidy_lookup(str(ploidy_path)) == {("S1", "chr1"): 2, ("S2", "chr2"): 3}


def test_assignment_and_event_helpers_are_deterministic():
    rng = np.random.default_rng(123)
    eligible_entries = [
        ("chr1", {"GD_ID": "GD1", "start_GRCh38": 100, "end_GRCh38": 200, "svtype": "DEL", "cluster": "c1"}),
        ("chr2", {"GD_ID": "GD2", "start_GRCh38": 300, "end_GRCh38": 500, "svtype": "DUP", "cluster": "c2"}),
    ]

    none_assigned = assign_gd_to_samples(["S1", "S2"], eligible_entries, np.random.default_rng(1), 0.0)
    assert none_assigned == {}

    assigned = assign_gd_to_samples(["S1", "S2", "S3"], eligible_entries, rng, 1.0)
    assert set(assigned) == {"S1", "S2", "S3"}
    assert {value[1]["GD_ID"] for value in assigned.values()} <= {"GD1", "GD2"}

    chosen = _select_sample_subset(["S1", "S2", "S3", "S4"], 0.5, np.random.default_rng(4), minimum=1)
    assert len(chosen) == 2
    assert set(chosen) <= {"S1", "S2", "S3", "S4"}
    assert _select_sample_subset(["S1"], 0.0, np.random.default_rng(4)) == []

    event = _make_synth_event(
        sample_id="S1",
        chrom="chr1",
        start=100,
        end=200,
        svtype="DEL",
        multiplier=0.5,
        baseline_ploidy=2,
        event_id="event1",
        source="gd",
        cluster="c1",
        gd_id="GD1",
        extra={"note": "x"},
    )
    assert event["sample_id"] == "S1"
    assert event["cluster"] == "c1"
    assert event["note"] == "x"

    built = _build_assignment_events(
        {
            "S1": ("chr1", eligible_entries[0][1]),
            "S2": ("chr2", eligible_entries[1][1]),
        },
        del_multiplier=0.4,
        dup_multiplier=1.6,
        ploidy_lookup={("S1", "chr1"): 4, ("S2", "chr2"): 2},
    )
    built_by_sample = {row["sample_id"]: row for row in built}
    assert built_by_sample["S1"]["baseline_ploidy"] == 4
    assert built_by_sample["S1"]["multiplier"] == 0.75
    assert built_by_sample["S2"]["baseline_ploidy"] == 2
    assert built_by_sample["S2"]["multiplier"] == 1.5


def test_background_and_salted_flank_event_generation():
    assignments = {
        "S1": ("chr1", {"start_GRCh38": 110, "end_GRCh38": 300}),
    }
    assert _background_conflicts_with_primary("S1", "chr1", 200, 320, assignments) is True
    assert _background_conflicts_with_primary("S1", "chr2", 200, 320, assignments) is False
    assert _background_conflicts_with_primary("S9", "chr1", 200, 320, assignments) is False

    rng = _ScriptedRng(
        choices=[["S1"]],
        integers=[0, 1, 2, 2],
        uniforms=[0.5, 0.75],
        randoms=[0.25],
    )
    events = generate_salted_flank_bleed_events(
        sample_ids=["S1"],
        eligible_loci=[_make_locus()],
        rng=rng,
        probability=1.0,
        del_multiplier=0.4,
        dup_multiplier=1.6,
        ploidy_lookup={("S1", "chr1"): 4},
    )

    assert len(events) == 1
    event = events[0]
    assert event["sample_id"] == "S1"
    assert event["chrom"] == "chr1"
    assert event["start"] == 165
    assert event["end"] == 368
    assert event["svtype"] == "DEL"
    assert event["multiplier"] == 0.75
    assert event["source"] == "salt"
    assert event["event_id"] == "salt_del_0001"
    assert event["cluster"] == "cluster1"
    assert event["body_start"] == 210
    assert event["body_end"] == 300
    assert event["flank_mode"] == "both"
    assert event["body_bp1"] == "B"
    assert event["body_bp2"] == "C"


def test_viable_trisomy_and_interval_resolution_helpers():
    assert _infer_chrom_name(["chr1", "chr2"], "21") == "chr21"
    assert _infer_chrom_name(["1", "2"], "21") == "21"

    rng = _ScriptedRng(choices=[["S1", "S2"]], integers=[0, 0])
    events = generate_viable_trisomy_events(
        sample_ids=["S1", "S2"],
        chrom_examples=["chr1", "chr21"],
        rng=rng,
        probability=1.0,
        dup_multiplier=1.6,
        ploidy_lookup={("S1", "chr18"): 2, ("S2", "chr21"): 1},
        assignments={"S1": ("chr21", {"GD_ID": "GD1"})},
    )

    events_by_sample = {event["sample_id"]: event for event in events}
    assert events_by_sample["S1"]["chrom"] == "chr18"
    assert events_by_sample["S1"]["source"] == "aneuploidy"
    assert events_by_sample["S1"]["svtype"] == "DUP"
    assert events_by_sample["S1"]["start"] == 0
    assert events_by_sample["S1"]["end"] == 2_000_000_000
    assert events_by_sample["S1"]["baseline_ploidy"] == 2
    assert events_by_sample["S1"]["multiplier"] == 1.5
    assert events_by_sample["S2"]["chrom"] == "chr21"
    assert events_by_sample["S2"]["baseline_ploidy"] == 1
    assert events_by_sample["S2"]["multiplier"] == 2.0

    interval_map = _build_interval_map_from_events(
        [
            {"chrom": "chr1", "start": 10, "end": 20, "sample_id": "S1", "multiplier": 0.5},
            {"chrom": "chr1", "start": 10, "end": 20, "sample_id": "S2", "multiplier": 1.5},
            {"chrom": "chr2", "start": 5, "end": 15, "sample_id": "S3", "multiplier": 2.0},
        ]
    )
    assert interval_map[("chr1", 10, 20)] == [("S1", 0.5), ("S2", 1.5)]

    resolved = _resolve_intervals(interval_map, {"S1": 0, "S2": 1})
    assert resolved["chr1"] == [(10, 20, [(0, 0.5), (1, 1.5)])]
    assert "chr2" not in resolved


def test_tabix_index_helpers_use_expected_index_parameters(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(synthesize_module.os.path, "exists", lambda path: False)
    monkeypatch.setattr(
        synthesize_module.pysam,
        "tabix_index",
        lambda path, **kwargs: calls.append((path, kwargs)),
    )

    baf_path = tmp_path / "sample.baf.tsv"
    baf_path.write_text("Chr\tPos\tBAF\tSample\nchr1\t10\t0.4\tS1\n")

    _ensure_tabix_index("counts.bgz")
    _ensure_baf_tabix_index(str(baf_path))

    assert calls[0] == (
        "counts.bgz",
        {
            "seq_col": 0,
            "start_col": 1,
            "end_col": 2,
            "meta_char": "#",
            "zerobased": True,
            "force": True,
        },
    )
    assert calls[1] == (
        str(baf_path),
        {
            "seq_col": 0,
            "start_col": 1,
            "end_col": 1,
            "meta_char": "#",
            "zerobased": False,
            "line_skip": 1,
            "force": True,
        },
    )


def test_process_helpers_rewrite_counts_and_baf_rows(monkeypatch, tmp_path):
    counts_rows = {
        "chr1": [
            "chr1\t100\t150\t10\t20",
            "chr1\t300\t350\t30\t40",
        ],
        "chr2": ["chr2\t50\t80\t5\t6"],
    }
    monkeypatch.setattr(
        synthesize_module.pysam,
        "TabixFile",
        lambda path: _StubTabixFile(counts_rows, fail_contigs={"chr_missing"}),
    )
    monkeypatch.setattr(synthesize_module.pysam, "BGZFile", _StubBGZFile, raising=False)

    count_part_1 = tmp_path / "chr1.bgzf"
    count_part_2 = tmp_path / "chr2.bgzf"
    total_rows, total_modified = _process_contig_group(
        "counts.bgz",
        [
            ("chr1", str(count_part_1)),
            ("chr2", str(count_part_2)),
            ("chr_missing", str(tmp_path / "missing.bgzf")),
        ],
        {"chr1": [(90, 200, [(3, 0.5)])]},
        start_col=1,
        end_col=2,
    )

    assert (total_rows, total_modified) == (3, 1)
    assert count_part_1.read_text() == "chr1\t100\t150\t5\t20\nchr1\t300\t350\t30\t40\n"
    assert count_part_2.read_text() == "chr2\t50\t80\t5\t6\n"

    baf_rows = {
        "chr1": [
            "chr1\t120\t0.6\tS1",
            "chr1\t125\t0.2\tS2",
            "chr1\t180\tnan\tS1",
        ]
    }
    monkeypatch.setattr(
        synthesize_module.pysam,
        "TabixFile",
        lambda path: _StubTabixFile(baf_rows),
    )

    baf_part = tmp_path / "baf.bgzf"
    baf_total_rows, baf_total_modified = _process_baf_contig_group(
        "baf.bgz",
        [("chr1", str(baf_part))],
        {"chr1": [(100, 150, {"S1": ("DUP", 2)})]},
        pos_col=1,
        baf_col=2,
        sample_col=3,
    )

    assert (baf_total_rows, baf_total_modified) == (3, 1)
    assert baf_part.read_text() == "chr1\t120\t0.666667\tS1\nchr1\t125\t0.2\tS2\nchr1\t180\tnan\tS1\n"
    assert _spike_baf_value(0.8, "DEL", 2) == 1.0
    assert _spike_baf_value(0.2, "DUP", 2) == pytest.approx(1.0 / 3.0)
    assert _spike_baf_value(0.8, "DUP", 1) == 1.0

    baf_map = _build_baf_interval_map(
        [
            {"chrom": "chr1", "start": 100, "end": 150, "sample_id": "S1", "svtype": "DEL", "baseline_ploidy": 2},
            {"chrom": "chr1", "start": 100, "end": 150, "sample_id": "S2", "svtype": "DUP", "baseline_ploidy": 3},
        ]
    )
    assert baf_map == {"chr1": [(100, 150, {"S1": ("DEL", 2), "S2": ("DUP", 3)})]}

    spike_index = _build_spike_index(
        {
            "S1": ("chr1", {"start_GRCh38": 100, "end_GRCh38": 150, "svtype": "DEL"}),
            "S2": ("chr2", {"start_GRCh38": 200, "end_GRCh38": 260, "svtype": "DUP"}),
        }
    )
    assert spike_index == {
        ("chr1", 100, 150): [("S1", 0.5)],
        ("chr2", 200, 260): [("S2", 1.5)],
    }


def test_concat_and_manifest_writers_emit_expected_files(tmp_path):
    part1 = tmp_path / "part1.bgzf"
    part2 = tmp_path / "part2.bgzf"
    output = tmp_path / "merged.bgzf"
    part1.write_bytes(b"header\n" + _BGZF_EOF)
    part2.write_bytes(b"body\n" + _BGZF_EOF)

    _concat_bgzf_parts(str(output), [str(part1), str(part2)], label="test")

    assert output.read_bytes() == b"header\nbody\n" + _BGZF_EOF

    truth_path = tmp_path / "truth.tsv"
    _write_truth_table(
        {
            "S2": ("chr2", {"GD_ID": "GD2"}),
            "S1": ("chr1", {"GD_ID": "GD1"}),
        },
        str(truth_path),
    )
    assert truth_path.read_text() == "sample_id\tGD_ID\nS1\tGD1\nS2\tGD2\n"

    events_path = tmp_path / "events.tsv"
    _write_background_event_table(
        [
            {
                "sample_id": "S1",
                "event_id": "salt_del_0001",
                "source": "salt",
                "chrom": "chr1",
                "start": 100,
                "end": 200,
                "svtype": "DEL",
                "multiplier": 0.5,
                "baseline_ploidy": 2,
                "cluster": "cluster1",
                "GD_ID": "",
                "body_start": 120,
                "body_end": 180,
                "flank_mode": "both",
                "body_bp1": "A",
                "body_bp2": "B",
            }
        ],
        str(events_path),
    )

    with open(events_path, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows[0]["sample_id"] == "S1"
    assert rows[0]["event_id"] == "salt_del_0001"
    assert rows[0]["flank_mode"] == "both"


def test_main_exits_when_no_input_tables(monkeypatch, tmp_path):
    monkeypatch.setattr(
        synthesize_module,
        "parse_args",
        lambda: SimpleNamespace(
            lo_res_counts=None,
            hi_res_counts=None,
            baf_table=None,
            ploidy_table="ploidy.tsv",
            gd_table="gd.tsv",
            output_dir=str(tmp_path / "out"),
            gd_probability=0.5,
            salted_event_probability=0.2,
            viable_trisomy_probability=0.1,
            seed=7,
            truth_table=None,
            regions=None,
            del_multiplier=0.5,
            dup_multiplier=1.5,
            threads=1,
        ),
    )
    monkeypatch.setattr(synthesize_module, "setup_logging", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as excinfo:
        synthesize_module.main()

    assert excinfo.value.code == 1


def test_main_reuses_truth_table_and_rewrites_requested_outputs(monkeypatch, tmp_path, capsys):
    output_dir = tmp_path / "out"
    yes_entry = {
        "GD_ID": "GD1",
        "NAHR": "yes",
        "svtype": "DEL",
        "start_GRCh38": 110,
        "end_GRCh38": 300,
        "cluster": "cluster1",
    }
    no_entry = {
        "GD_ID": "GD2",
        "NAHR": "no",
        "svtype": "DUP",
        "start_GRCh38": 500,
        "end_GRCh38": 700,
        "cluster": "cluster2",
    }
    gd_table_stub = SimpleNamespace(
        loci={"cluster1": _make_locus()},
        get_all_loci=lambda: {"cluster1": _make_locus()},
    )

    monkeypatch.setattr(
        synthesize_module,
        "parse_args",
        lambda: SimpleNamespace(
            lo_res_counts="lo.rd.txt.gz",
            hi_res_counts="hi.rd.txt.gz",
            baf_table="all.baf.txt.gz",
            ploidy_table="ploidy.tsv",
            gd_table="gd.tsv",
            output_dir=str(output_dir),
            gd_probability=0.5,
            salted_event_probability=0.2,
            viable_trisomy_probability=0.1,
            seed=11,
            truth_table="truth.tsv",
            regions=["chr1:1-100"],
            del_multiplier=0.5,
            dup_multiplier=1.5,
            threads=3,
        ),
    )
    monkeypatch.setattr(synthesize_module, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(synthesize_module, "GDTable", lambda path: gd_table_stub)
    monkeypatch.setattr(
        synthesize_module,
        "_build_gd_lookup",
        lambda gd_table: {"GD1": ("chr1", yes_entry), "GD2": ("chr2", no_entry)},
    )
    monkeypatch.setattr(synthesize_module, "_discover_sample_ids", lambda *args: ["S1", "S2"])
    monkeypatch.setattr(synthesize_module, "_load_ploidy_lookup", lambda path: {("S1", "chr1"): 2})
    monkeypatch.setattr(
        synthesize_module,
        "_load_truth_assignments",
        lambda path, lookup: {
            "S1": ("chr1", yes_entry),
            "S8": ("chr1", yes_entry),
            "S9": ("chr2", no_entry),
        },
    )
    monkeypatch.setattr(
        synthesize_module,
        "_build_assignment_events",
        lambda assignments, **kwargs: [{"sample_id": "S1", "chrom": "chr1", "start": 110, "end": 300, "multiplier": 0.5}],
    )
    monkeypatch.setattr(
        synthesize_module,
        "generate_salted_flank_bleed_events",
        lambda *args, **kwargs: [{"sample_id": "S1", "chrom": "chr1", "start": 50, "end": 350, "multiplier": 1.5}],
    )
    monkeypatch.setattr(
        synthesize_module,
        "generate_viable_trisomy_events",
        lambda *args, **kwargs: [{"sample_id": "S2", "chrom": "chr21", "start": 0, "end": 2_000_000_000, "multiplier": 1.5}],
    )
    monkeypatch.setattr(
        synthesize_module,
        "_build_interval_map_from_events",
        lambda events: {("chr1", 110, 300): [("S1", 0.5)]},
    )

    counts_calls = []
    baf_calls = []
    truth_writes = []
    bg_writes = []
    monkeypatch.setattr(
        synthesize_module,
        "_rewrite_counts_file",
        lambda input_path, output_path, interval_map, **kwargs: counts_calls.append((input_path, output_path, interval_map, kwargs)) or 1,
    )
    monkeypatch.setattr(
        synthesize_module,
        "_rewrite_baf_file",
        lambda input_path, output_path, events, **kwargs: baf_calls.append((input_path, output_path, events, kwargs)) or 1,
    )
    monkeypatch.setattr(
        synthesize_module,
        "_write_truth_table",
        lambda assignments, output_path: truth_writes.append((assignments, output_path)),
    )
    monkeypatch.setattr(
        synthesize_module,
        "_write_background_event_table",
        lambda events, output_path: bg_writes.append((events, output_path)),
    )

    synthesize_module.main()

    stdout = capsys.readouterr().out
    assert "NOTE: --region is ignored when --truth-table is provided" in stdout
    assert "Restricting synthesis to 1 NAHR GD entries; skipping 1 non-NAHR entries" in stdout
    assert "WARNING: dropping 1 non-NAHR carrier assignments" in stdout
    assert "WARNING: dropping 1 truth-table carriers absent from input tables" in stdout
    assert "Reused 1 carrier assignments from truth table" in stdout
    assert "Added 1 salted flank-bleed event(s)" in stdout
    assert "Added 1 viable trisomy/YY event(s)" in stdout
    assert "SYNTHESIS COMPLETE" in stdout

    assert counts_calls == [
        (
            "lo.rd.txt.gz",
            str(output_dir / "lo_res_counts.synthesized.rd.txt.gz"),
            {("chr1", 110, 300): [("S1", 0.5)]},
            {"label": "lo-res", "n_workers": 3},
        ),
        (
            "hi.rd.txt.gz",
            str(output_dir / "hi_res_counts.synthesized.rd.txt.gz"),
            {("chr1", 110, 300): [("S1", 0.5)]},
            {"label": "hi-res", "n_workers": 3},
        ),
    ]
    assert baf_calls == [
        (
            "all.baf.txt.gz",
            str(output_dir / "all_samples.synthesized.baf.txt.gz"),
            [
                {"sample_id": "S1", "chrom": "chr1", "start": 110, "end": 300, "multiplier": 0.5},
                {"sample_id": "S1", "chrom": "chr1", "start": 50, "end": 350, "multiplier": 1.5},
                {"sample_id": "S2", "chrom": "chr21", "start": 0, "end": 2_000_000_000, "multiplier": 1.5},
            ],
            {"label": "baf", "n_workers": 3},
        )
    ]
    assert truth_writes == [({"S1": ("chr1", yes_entry)}, str(output_dir / "truth_table.tsv"))]
    assert bg_writes == [(
        [
            {"sample_id": "S1", "chrom": "chr1", "start": 50, "end": 350, "multiplier": 1.5},
            {"sample_id": "S2", "chrom": "chr21", "start": 0, "end": 2_000_000_000, "multiplier": 1.5},
        ],
        str(output_dir / "background_events.tsv"),
    )]


def test_rewrite_counts_file_orchestrates_header_workers_and_indexing(monkeypatch, tmp_path):
    process_calls = []
    concat_calls = []
    tabix_calls = []
    monkeypatch.setattr(synthesize_module, "_ensure_tabix_index", lambda path: None)
    monkeypatch.setattr(synthesize_module.os.path, "getsize", lambda path: 100)
    monkeypatch.setattr(
        synthesize_module.pysam,
        "TabixFile",
        lambda path: _StubTabixFile({}, header=["#Chr\tStart\tEnd\tS1"], contigs=["chr1", "chr2"]),
    )
    monkeypatch.setattr(synthesize_module.pysam, "BGZFile", _StubBGZFile, raising=False)
    monkeypatch.setattr(
        synthesize_module,
        "_process_contig_group",
        lambda input_path, group, resolved, start_col, end_col: process_calls.append(
            (input_path, group, resolved, start_col, end_col)
        ) or (5, 2),
    )
    monkeypatch.setattr(
        synthesize_module,
        "_concat_bgzf_parts",
        lambda output_path, part_paths, label="": concat_calls.append((output_path, part_paths, label)),
    )
    monkeypatch.setattr(
        synthesize_module.pysam,
        "tabix_index",
        lambda path, **kwargs: tabix_calls.append((path, kwargs)),
    )

    modified = _rewrite_counts_file(
        "counts.bgz",
        str(tmp_path / "counts.out.bgz"),
        {("chr1", 100, 200): [("S1", 0.5)]},
        label="lo-res",
        n_workers=1,
    )

    assert modified == 2
    assert len(process_calls) == 1
    assert process_calls[0][2] == {"chr1": [(100, 200, [(3, 0.5)])]}
    assert process_calls[0][3:] == (1, 2)
    assert concat_calls[0][0] == str(tmp_path / "counts.out.bgz")
    assert concat_calls[0][2] == "lo-res"
    assert len(concat_calls[0][1]) == 3
    assert tabix_calls == [(
        str(tmp_path / "counts.out.bgz"),
        {
            "seq_col": 0,
            "start_col": 1,
            "end_col": 2,
            "meta_char": "#",
            "zerobased": True,
            "force": True,
        },
    )]


def test_rewrite_baf_file_orchestrates_header_workers_and_indexing(monkeypatch, tmp_path):
    input_path = tmp_path / "input.baf.tsv"
    input_path.write_text("Chr\tPos\tBAF\tSample\nchr1\t100\t0.5\tS1\n")

    process_calls = []
    concat_calls = []
    tabix_calls = []
    monkeypatch.setattr(synthesize_module, "_ensure_baf_tabix_index", lambda path: None)
    monkeypatch.setattr(synthesize_module.os.path, "getsize", lambda path: 100)
    monkeypatch.setattr(
        synthesize_module.pysam,
        "TabixFile",
        lambda path: _StubTabixFile({}, header=["#meta"], contigs=["chr1"]),
    )
    monkeypatch.setattr(synthesize_module.pysam, "BGZFile", _StubBGZFile, raising=False)
    monkeypatch.setattr(
        synthesize_module,
        "_process_baf_contig_group",
        lambda input_path, group, resolved, pos_col, baf_col, sample_col: process_calls.append(
            (input_path, group, resolved, pos_col, baf_col, sample_col)
        ) or (3, 1),
    )
    monkeypatch.setattr(
        synthesize_module,
        "_concat_bgzf_parts",
        lambda output_path, part_paths, label="": concat_calls.append((output_path, part_paths, label)),
    )
    monkeypatch.setattr(
        synthesize_module.pysam,
        "tabix_index",
        lambda path, **kwargs: tabix_calls.append((path, kwargs)),
    )

    modified = _rewrite_baf_file(
        str(input_path),
        str(tmp_path / "output.baf.bgz"),
        [{"chrom": "chr1", "start": 100, "end": 200, "sample_id": "S1", "svtype": "DEL", "baseline_ploidy": 2}],
        label="baf",
        n_workers=1,
    )

    assert modified == 1
    assert len(process_calls) == 1
    assert process_calls[0][2] == {"chr1": [(100, 200, {"S1": ("DEL", 2)})]}
    assert process_calls[0][3:] == (1, 2, 3)
    assert concat_calls[0][0] == str(tmp_path / "output.baf.bgz")
    assert concat_calls[0][2] == "baf"
    assert len(concat_calls[0][1]) == 2
    assert tabix_calls == [(
        str(tmp_path / "output.baf.bgz"),
        {
            "seq_col": 0,
            "start_col": 1,
            "end_col": 1,
            "meta_char": "#",
            "zerobased": False,
            "line_skip": 1,
            "force": True,
        },
    )]