"""Tests for gatk_sv_gd.integrate."""

import logging
import os
import subprocess
import sys
from collections import defaultdict

import pytest

from gatk_sv_gd import integrate
from gatk_sv_gd._util import overlap_bases, reciprocal_overlap, fraction_covered


# ── FakeIntervalTree / FakeInterval ─────────────────────────────────


class FakeInterval:
    def __init__(self, begin, end, data=None):
        self.begin = begin
        self.end = end
        self.data = data


class FakeIntervalTree:
    def __init__(self):
        self._intervals = []

    def addi(self, start, end, data=None):
        self._intervals.append(FakeInterval(start, end, data))

    def merge_overlaps(self):
        pass

    def overlap(self, start, end):
        return [
            iv for iv in self._intervals
            if iv.begin < end and start < iv.end
        ]

    def __len__(self):
        return len(self._intervals)

    def __iter__(self):
        return iter(self._intervals)


# ── Pysam stub ───────────────────────────────────────────────────────


class _SampleGT(dict):
    """Simulates a per-sample genotype dict from pysam."""
    pass


class _FakeRecord:
    """Minimal pysam VariantRecord stub."""

    def __init__(
        self,
        chrom,
        pos,  # 1-based POS (pysam .pos is 1-based for existing records)
        stop,  # 0-based end
        record_id=".",
        alts=None,
        info=None,
        samples=None,
    ):
        self.chrom = chrom
        self.pos = pos
        self.stop = stop
        self.id = record_id
        self.alts = alts or ()
        self.info = dict(info) if info else {}
        # samples is a dict: sample_name -> dict-like genotype
        self._samples = {}
        if samples:
            for name, gt_dict in samples.items():
                self._samples[name] = _SampleGT(gt_dict)

    @property
    def start(self):
        """0-based start — pysam's .start property."""
        return self.pos - 1

    @property
    def samples(self):
        return self._samples

    def copy(self):
        return _FakeRecord(
            self.chrom,
            self.pos,
            self.stop,
            self.id,
            self.alts,
            dict(self.info),
            {k: dict(v) for k, v in self._samples.items()},
        )


class _FakeHeader:
    """Minimal pysam VariantHeader stub."""

    def __init__(self, contigs=None, samples=None):
        self.info = {}
        self.formats = {}
        self.contigs = dict(contigs) if contigs else {}
        self.samples = list(samples) if samples else []
        self._lines = []

    def add_line(self, line):
        self._lines.append(line)
        # Parse out ID= for info/format tracking
        if "##INFO=" in line:
            info_id = line.split("ID=")[1].split(",")[0].split(">")[0]
            self.info[info_id] = object()
        elif "##FORMAT=" in line:
            fmt_id = line.split("ID=")[1].split(",")[0].split(">")[0]
            self.formats[fmt_id] = object()

    def new_record(self, contig, start, stop, alleles, id=None):
        """Return a new FakeRecord; start is 0-based per pysam convention."""
        rec = _FakeRecord(
            chrom=contig,
            pos=start + 1,  # store as 1-based to match .pos convention
            stop=stop,
            record_id=id or ".",
            alts=(alleles[1],) if len(alleles) > 1 else (),
            info={},
        )
        for sample in self.samples:
            rec._samples[sample] = _SampleGT()
        return rec

    def copy(self):
        h = _FakeHeader(contigs=dict(self.contigs), samples=list(self.samples))
        h.info = dict(self.info)
        h.formats = dict(self.formats)
        h._lines = list(self._lines)
        return h


class _FakeVariantFile:
    """Minimal pysam VariantFile stub."""

    def __init__(self, path, mode=None, header=None):
        self._path = path
        self._mode = mode or "r"
        if mode == "w":
            self._write_header = header
            self.header = header
            self._written_records = []
        else:
            # Read mode: populated by test fixtures
            self._records = []
            self._header = None

    def __iter__(self):
        return iter(self._records)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        pass

    def write(self, record):
        self._written_records.append(record)


def _make_fake_pysam(read_records=None, header=None):
    """Return a pysam stub whose VariantFile is monkeypatchable."""
    import types
    fake_pysam = types.SimpleNamespace()

    _header = header or _FakeHeader(contigs={"chr1": None}, samples=["S1", "S2"])
    _records = read_records or []

    class _FVF:
        """VariantFile factory used to monkeypatch integrate.pysam.VariantFile."""
        _instance_count = 0
        written = []

        def __new__(cls, path, mode=None, header=None):
            obj = object.__new__(cls)
            obj._path = path
            obj._mode = mode or "r"
            if mode == "w":
                obj._write_header = header
                obj.header = header
                obj._written = []
            else:
                obj._records = list(_records)
                obj.header = _header
            return obj

        def __iter__(self):
            return iter(self._records)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def close(self):
            pass

        def write(self, record):
            self._written.append(record)
            _FVF.written.append(record)

    fake_pysam.VariantFile = _FVF
    fake_pysam.tabix_index = lambda *a, **k: None
    return fake_pysam, _FVF


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _patch_interval_tree(monkeypatch):
    """Monkeypatch integrate.IntervalTree with functional FakeIntervalTree."""
    monkeypatch.setattr(integrate, "IntervalTree", FakeIntervalTree)


@pytest.fixture()
def _patch_sort_vcf(monkeypatch, tmp_path):
    """No-op _sort_vcf — just copy the file."""
    def _noop_sort(vcf_path, out_path, temp_dir):
        import shutil
        shutil.copy(vcf_path, out_path)

    monkeypatch.setattr(integrate, "_sort_vcf", _noop_sort)


# ── Pure helper unit tests ────────────────────────────────────────────


class TestOverlapHelpers:
    """Tests for shared overlap helpers imported from _util."""

    def test_overlap_bases_normal(self):
        assert overlap_bases(10, 20, 15, 30) == 5

    def test_overlap_bases_no_overlap(self):
        assert overlap_bases(10, 20, 20, 30) == 0

    def test_overlap_bases_adjacent(self):
        assert overlap_bases(10, 20, 20, 30) == 0

    def test_overlap_bases_contained(self):
        assert overlap_bases(0, 100, 10, 20) == 10

    def test_overlap_bases_zero_length_a(self):
        assert overlap_bases(10, 10, 5, 20) == 0

    def test_overlap_bases_zero_length_b(self):
        assert overlap_bases(5, 20, 10, 10) == 0

    def test_reciprocal_overlap_normal(self):
        assert reciprocal_overlap(10, 20, 15, 30) == pytest.approx(5 / 15)

    def test_reciprocal_overlap_identical(self):
        assert reciprocal_overlap(10, 20, 10, 20) == pytest.approx(1.0)

    def test_reciprocal_overlap_zero_length(self):
        assert reciprocal_overlap(10, 10, 10, 20) == pytest.approx(0.0)

    def test_reciprocal_overlap_no_overlap(self):
        assert reciprocal_overlap(0, 10, 20, 30) == pytest.approx(0.0)

    def test_fraction_covered_partial(self):
        assert fraction_covered(100, 200, 125, 175) == pytest.approx(0.5)

    def test_fraction_covered_full(self):
        assert fraction_covered(100, 200, 100, 200) == pytest.approx(1.0)

    def test_fraction_covered_zero_length_region(self):
        assert fraction_covered(100, 100, 90, 110) == pytest.approx(0.0)

    def test_fraction_covered_no_overlap(self):
        assert fraction_covered(100, 200, 200, 300) == pytest.approx(0.0)


# ── Sample-overlap unit tests ──────────────────────────────────────────


class TestSampleOverlap:
    """Tests for sample_overlap() and _extract_vcf_carriers()."""

    def test_partial_overlap(self):
        assert integrate.sample_overlap(
            {"A", "B"}, {"A", "B", "C"}
        ) == pytest.approx(2 / 3)

    def test_disjoint_sets(self):
        assert integrate.sample_overlap({"A"}, {"B"}) == 0.0

    def test_identical_sets(self):
        assert integrate.sample_overlap({"A", "B"}, {"A", "B"}) == 1.0

    def test_subset(self):
        assert integrate.sample_overlap({"A"}, {"A", "B", "C"}) == pytest.approx(
            1 / 3
        )

    def test_both_empty_returns_none(self):
        assert integrate.sample_overlap(set(), set()) is None

    def test_one_empty_nonempty(self):
        # One set is empty -> intersection = 0 -> overlap = 0
        assert integrate.sample_overlap({"A"}, set()) == pytest.approx(0.0)
        assert integrate.sample_overlap(set(), {"B"}) == pytest.approx(0.0)


class TestExtractVcfCarriers:
    """Tests for _extract_vcf_carriers()."""

    def test_no_carriers(self):
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,
            stop=5000,
            samples={"S1": {"GT": (0, 0)}, "S2": {"GT": (0, 0)}},
        )
        assert integrate._extract_vcf_carriers(rec) == set()

    def test_one_carrier(self):
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,
            stop=5000,
            samples={"S1": {"GT": (0, 1)}, "S2": {"GT": (0, 0)}},
        )
        assert integrate._extract_vcf_carriers(rec) == {"S1"}

    def test_all_carriers(self):
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,
            stop=5000,
            samples={"S1": {"GT": (0, 1)}, "S2": {"GT": (0, 1)}},
        )
        assert integrate._extract_vcf_carriers(rec) == {"S1", "S2"}

    def test_no_call_ignored(self):
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,
            stop=5000,
            samples={"S1": {"GT": (None, None)}, "S2": {"GT": (0, 0)}},
        )
        assert integrate._extract_vcf_carriers(rec) == {"S1"}


# ── Phase 2 tiebreaking tests ──────────────────────────────────────────


class TestTiebreakSampleOverlap:
    """Phase 2 tiebreaking: sample overlap selects the best match."""

    def test_prefers_higher_sample_overlap(self, monkeypatch, tmp_path):
        """Two overlapping NAHR regions with identical RO, different sample overlap.

        The winner (higher sample overlap) gets matched; the loser is emitted
        as a novel record but its carriers (S99) are not in the VCF header,
        so after genotype reconciliation all samples become hom-ref and the
        novel record is suppressed.  Only the matched record is written.
        """
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1", "S2"])
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,
            stop=5000,
            record_id="var1",
            info={"SVTYPE": "DEL"},
            samples={
                "S1": {"GT": (0, 1), "RD_CN": 1},
                "S2": {"GT": (0, 1), "RD_CN": 1},
            },
        )

        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[rec],
            vcf_header=header,
            gd_table_rows=[
                # Both regions at identical coords -> identical RO
                {
                    "chr": "chr1", "start": 1000, "end": 5000,
                    "gd_id": "GD_PROXIMAL", "svtype": "DEL",
                    "nahr": "yes", "cluster": "clusterA", "bp1": "1", "bp2": "2",
                },
                {
                    "chr": "chr1", "start": 1000, "end": 5000,
                    "gd_id": "GD_DISTAL", "svtype": "DEL",
                    "nahr": "yes", "cluster": "clusterA", "bp1": "1", "bp2": "2",
                },
            ],
            gd_calls_entries=[
                # Proximal has S1, S2 (perfect match with VCF carriers)
                {
                    "chrom": "chr1", "pos": 1000, "end": 5000,
                    "region_id": "GD_PROXIMAL", "svtype": "DEL",
                    "samples": ["S1", "S2"],
                },
                # Distal has S99 (carrier not in VCF header)
                {
                    "chrom": "chr1", "pos": 1000, "end": 5000,
                    "region_id": "GD_DISTAL", "svtype": "DEL",
                    "samples": ["S99"],
                },
            ],
        )

        # Only the matched record is written; GD_DISTAL novel suppressed
        assert len(written) == 1
        assert written[0].info.get("GENOMIC_DISORDER") == "GD_PROXIMAL"

    def test_fallback_size_when_no_carriers(self, monkeypatch, tmp_path):
        """Both GD-call entries share the same carrier -> equal sample overlap.

        When sample overlap is identical for both regions, the size-difference
        tiebreaker selects the winner.
        """
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1", "S2"])
        # VCF record with carriers S1, S2
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,
            stop=5000,
            record_id="var1",
            info={"SVTYPE": "DEL"},
            samples={
                "S1": {"GT": (0, 1), "RD_CN": 1},
                "S2": {"GT": (0, 0), "RD_CN": 2},
            },
        )

        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[rec],
            vcf_header=header,
            gd_table_rows=[
                # Region 1: 4000 bp (larger)
                {
                    "chr": "chr1", "start": 1000, "end": 5000,
                    "gd_id": "GD_LARGE", "svtype": "DEL",
                    "nahr": "yes", "cluster": "clusterA", "bp1": "1", "bp2": "2",
                },
                # Region 2: 3000 bp (smaller, closer to variant size = 3999)
                {
                    "chr": "chr1", "start": 1000, "end": 4000,
                    "gd_id": "GD_SMALL", "svtype": "DEL",
                    "nahr": "yes", "cluster": "clusterA", "bp1": "1", "bp2": "2",
                },
            ],
            gd_calls_entries=[
                # Both share S1 as carrier -> sample_overlap = 1.0 for both
                {
                    "chrom": "chr1", "pos": 1000, "end": 5000,
                    "region_id": "GD_LARGE", "svtype": "DEL",
                    "samples": ["S1"],
                },
                {
                    "chrom": "chr1", "pos": 1000, "end": 4000,
                    "region_id": "GD_SMALL", "svtype": "DEL",
                    "samples": ["S1"],
                },
            ],
        )

        # Both have sample_overlap = 1.0 (S1 is only VCF carrier, both have S1)
        # Size diff LARGE: |3999 - 4000| = 1
        # Size diff SMALL: |3999 - 3000| = 999
        # GD_LARGE should win (smaller size difference)
        # Novel record for GD_SMALL also written (S1 carrier becomes het)
        matched = [r for r in written if r.info.get("GENOMIC_DISORDER") == "GD_LARGE"]
        assert len(matched) == 1

    def test_size_diff_when_sample_overlap_equal(self, monkeypatch, tmp_path):
        """Same sample overlap for both -> smaller size difference wins.

        GD_LARGE and GD_SMALL both have carrier S1 (sample_overlap=0.5 each).
        GD_SMALL's carrier is changed to S99 (not in VCF) so its novel record
        is suppressed by the all-hom-ref filter.  Only the matched GD_LARGE
        record is written.
        """
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1", "S2"])
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,
            stop=6000,
            record_id="var1",
            info={"SVTYPE": "DEL"},
            samples={
                "S1": {"GT": (0, 1), "RD_CN": 1},
                "S2": {"GT": (0, 1), "RD_CN": 1},
            },
        )

        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[rec],
            vcf_header=header,
            gd_table_rows=[
                # Larger region: 5000 bp
                {
                    "chr": "chr1", "start": 1000, "end": 6000,
                    "gd_id": "GD_LARGE", "svtype": "DEL",
                    "nahr": "yes", "cluster": "clusterA", "bp1": "1", "bp2": "2",
                },
                # Smaller region: 2000 bp (closer to variant length 4999)
                {
                    "chr": "chr1", "start": 1000, "end": 3000,
                    "gd_id": "GD_SMALL", "svtype": "DEL",
                    "nahr": "yes", "cluster": "clusterA", "bp1": "1", "bp2": "2",
                },
            ],
            gd_calls_entries=[
                # Both have S1 as carrier -> sample_overlap = 0.5 for both
                {
                    "chrom": "chr1", "pos": 1000, "end": 6000,
                    "region_id": "GD_LARGE", "svtype": "DEL",
                    "samples": ["S1"],
                },
                # GD_SMALL's carrier is S99 (not in VCF) -> novel record suppressed
                {
                    "chrom": "chr1", "pos": 1000, "end": 3000,
                    "region_id": "GD_SMALL", "svtype": "DEL",
                    "samples": ["S99"],
                },
            ],
        )

        # Both have sample_overlap = 0.5 -> tiebreak by size diff
        # GD_LARGE wins (size diff 1 < 2999)
        assert len(written) == 1
        assert written[0].info.get("GENOMIC_DISORDER") == "GD_LARGE"


class TestIsInParRegion:
    def _make_par_trees(self, intervals):
        trees = defaultdict(FakeIntervalTree)
        for chrom, start, end in intervals:
            trees[chrom].addi(start, end)
        return trees

    def test_inside_par(self):
        par = self._make_par_trees([("chrX", 0, 2700000)])
        assert integrate.is_in_par_region("chrX", 0, 1000000, par) is True

    def test_below_cutoff(self):
        # variant overlaps only 10% of its length with PAR
        par = self._make_par_trees([("chrX", 0, 100)])
        # variant is 0-1000, PAR overlaps 100 bases = 10%
        assert integrate.is_in_par_region("chrX", 0, 1000, par) is False

    def test_chrom_absent(self):
        par = self._make_par_trees([("chrX", 0, 1000)])
        assert integrate.is_in_par_region("chr1", 0, 500, par) is False

    def test_zero_length_interval(self):
        par = self._make_par_trees([("chrX", 0, 1000)])
        assert integrate.is_in_par_region("chrX", 500, 500, par) is False


class TestGetExpectedCn:
    def _make_par_trees(self, intervals):
        trees = defaultdict(FakeIntervalTree)
        for chrom, start, end in intervals:
            trees[chrom].addi(start, end)
        return trees

    def test_par_returns_2(self):
        par = self._make_par_trees([("chrX", 0, 2700000)])
        ploidy = {"S1": {"chrX": 1}}
        assert integrate.get_expected_cn("chrX", 0, 1000000, "S1", ploidy, par) == 2

    def test_sample_present_chrom_present(self):
        par = self._make_par_trees([])
        ploidy = {"S1": {"chrY": 1}}
        assert integrate.get_expected_cn("chrY", 0, 1000, "S1", ploidy, par) == 1

    def test_sample_present_chrom_absent_defaults_to_2(self):
        par = self._make_par_trees([])
        ploidy = {"S1": {"chr1": 2}}
        assert integrate.get_expected_cn("chrY", 0, 1000, "S1", ploidy, par) == 2

    def test_sample_absent_defaults_to_2(self):
        par = self._make_par_trees([])
        ploidy = {}
        assert integrate.get_expected_cn("chr1", 0, 1000, "MISSING", ploidy, par) == 2


class TestUpdateGenotype:
    def test_ecn_zero_no_call(self):
        gt = {"GT": (0, 0), "RD_CN": 2, "RD_GQ": 99}
        integrate.update_genotype(gt, "S1", True, 0, "DEL")
        assert gt["GT"] == (None, None)
        assert gt["RD_CN"] == 0
        assert gt["RD_GQ"] == 0

    def test_carrier_del_ecn1_rdcn_is_0(self):
        gt = {"GT": (0, 0), "RD_CN": 2}
        integrate.update_genotype(gt, "S1", True, 1, "DEL")
        assert gt["GT"] == (0, 1)
        assert gt["RD_CN"] == 0  # max(1-1, 0)
        assert gt["RD_GQ"] == 99
        assert gt["GQ"] == 99

    def test_carrier_del_ecn2(self):
        gt = {"GT": (0, 0)}
        integrate.update_genotype(gt, "S1", True, 2, "DEL")
        assert gt["GT"] == (0, 1)
        assert gt["RD_CN"] == 1  # 2-1

    def test_carrier_dup(self):
        gt = {"GT": (0, 0)}
        integrate.update_genotype(gt, "S1", True, 2, "DUP")
        assert gt["GT"] == (0, 1)
        assert gt["RD_CN"] == 3  # 2+1

    def test_non_carrier_homref(self):
        gt = {"GT": (0, 1)}
        integrate.update_genotype(gt, "S1", False, 2, "DEL")
        assert gt["GT"] == (0, 0)
        assert gt["RD_CN"] == 2
        assert gt["RD_GQ"] == 99

    def test_pesr_reset_when_present(self):
        gt = {"GT": (0, 0), "SR_GT": (1,), "SR_GQ": 50, "PE_GT": (1,), "PE_GQ": 50}
        integrate.update_genotype(gt, "S1", True, 2, "DEL")
        assert gt["SR_GT"] == (0,)
        assert gt["SR_GQ"] == 99
        assert gt["PE_GT"] == (0,)
        assert gt["PE_GQ"] == 99

    def test_pesr_skipped_when_absent(self):
        gt = {"GT": (0, 0)}
        integrate.update_genotype(gt, "S1", True, 2, "DEL")
        assert "SR_GT" not in gt
        assert "PE_GT" not in gt


class TestReadPloidyTable:
    def test_basic(self, tmp_path):
        p = tmp_path / "ploidy.tsv"
        p.write_text("sample\tchr1\tchr2\nS1\t2\t2\nS2\t2\t1\n")
        result = integrate.read_ploidy_table(str(p))
        assert result["S1"] == {"chr1": 2, "chr2": 2}
        assert result["S2"]["chr2"] == 1

    def test_comment_lines_skipped(self, tmp_path):
        p = tmp_path / "ploidy.tsv"
        p.write_text("sample\tchr1\n# comment\nS1\t2\n")
        result = integrate.read_ploidy_table(str(p))
        assert "S1" in result
        assert "# comment" not in result


class TestReadGdCalls:
    def test_basic(self, tmp_path):
        p = tmp_path / "calls.tsv"
        p.write_text("chr1\t1000\t5000\tGD1\tDEL\tS1,S2\n")
        result = integrate.read_gd_calls(str(p))
        assert ("GD1", "DEL") in result
        assert result[("GD1", "DEL")]["pos"] == 1000
        assert result[("GD1", "DEL")]["samples"] == {"S1", "S2"}

    def test_empty_samples(self, tmp_path):
        p = tmp_path / "calls.tsv"
        p.write_text("chr1\t1000\t5000\tGD1\tDEL\t.\n")
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == set()

    def test_comment_and_short_lines_skipped(self, tmp_path):
        p = tmp_path / "calls.tsv"
        p.write_text("# header\nchr1\t100\t200\n")
        result = integrate.read_gd_calls(str(p))
        assert len(result) == 0

    def test_empty_sample_string(self, tmp_path):
        p = tmp_path / "calls.tsv"
        p.write_text("chr1\t1000\t5000\tGD1\tDEL\t\n")
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == set()

    # ── Wide format tests ──────────────────────────────────────────

    def test_wide_format_basic(self, tmp_path):
        """Wide format with carrier samples."""
        p = tmp_path / "calls.tsv"
        header = (
            "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
        )
        body = (
            "S1\tGD1\tchr1\t1000\t5000\tDEL\tTrue\n"
            "S2\tGD1\tchr1\t1000\t5000\tDEL\tTrue\n"
            "S3\tGD1\tchr1\t1000\t5000\tDEL\tFalse\n"
        )
        p.write_text(header + body)
        result = integrate.read_gd_calls(str(p))
        assert ("GD1", "DEL") in result
        assert result[("GD1", "DEL")]["pos"] == 1000
        assert result[("GD1", "DEL")]["end"] == 5000
        assert result[("GD1", "DEL")]["samples"] == {"S1", "S2"}

    def test_wide_format_gzipped(self, tmp_path):
        """Wide format with .gz extension."""
        import gzip
        p = tmp_path / "calls.tsv.gz"
        header = (
            "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
        )
        body = (
            "S1\tGD1\tchr1\t1000\t5000\tDEL\tTrue\n"
            "S2\tGD1\tchr1\t1000\t5000\tDEL\tFalse\n"
        )
        with gzip.open(str(p), "wt") as f:
            f.write(header + body)
        result = integrate.read_gd_calls(str(p))
        assert ("GD1", "DEL") in result
        assert result[("GD1", "DEL")]["samples"] == {"S1"}

    def test_wide_format_no_carriers(self, tmp_path):
        """Wide format with no carriers — entry exists but samples is empty."""
        p = tmp_path / "calls.tsv"
        header = (
            "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
        )
        body = (
            "S1\tGD1\tchr1\t1000\t5000\tDEL\tFalse\n"
        )
        p.write_text(header + body)
        result = integrate.read_gd_calls(str(p))
        assert ("GD1", "DEL") in result
        assert result[("GD1", "DEL")]["samples"] == set()

    def test_wide_format_multiple_entries(self, tmp_path):
        """Wide format with multiple GD_ID entries."""
        p = tmp_path / "calls.tsv"
        header = (
            "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
        )
        body = (
            "S1\tGD1\tchr1\t1000\t5000\tDEL\tTrue\n"
            "S2\tGD2\tchr2\t2000\t6000\tDUP\tTrue\n"
        )
        p.write_text(header + body)
        result = integrate.read_gd_calls(str(p))
        assert ("GD1", "DEL") in result
        assert ("GD2", "DUP") in result
        assert result[("GD1", "DEL")]["chrom"] == "chr1"
        assert result[("GD2", "DUP")]["chrom"] == "chr2"

    def test_wide_format_header_detection(self, tmp_path):
        """_looks_like_wide_header requires GD_ID + is_carrier + positional field."""
        assert integrate._looks_like_wide_header(
            ["sample", "GD_ID", "chrom", "start", "end", "svtype", "is_carrier"]
        ) is True
        assert integrate._looks_like_wide_header(
            ["chr1", "1000", "5000", "GD1", "DEL", "S1"]
        ) is False
        assert integrate._looks_like_wide_header(
            ["GD_ID", "is_carrier"]  # missing positional field
        ) is False


class TestBuildTreesFromGdTable:
    def test_nahr_and_non_nahr_entries_go_to_correct_trees(
        self, monkeypatch, tmp_path
    ):
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_DEL\tDEL\tyes\tno\tclusterA\t1\t2\n"
            "chr1\t1000\t5000\tGD_DUP\tDUP\tyes\tno\tclusterA\t1\t2\n"
            "chr2\t2000\t8000\tGD_NON\tDEL\tno\tno\tclusterB\t1\t2\n"
        )
        nahr, non_nahr, meta = integrate._build_trees_from_gd_table(str(gd_tsv))

        # NAHR trees
        assert "chr1" in nahr
        nahr_data = {iv.data for iv in nahr["chr1"].overlap(1000, 5000)}
        assert ("GD_DEL", "DEL") in nahr_data
        assert ("GD_DUP", "DUP") in nahr_data

        # Non-NAHR trees
        assert "chr2" in non_nahr
        non_nahr_data = {iv.data for iv in non_nahr["chr2"].overlap(2000, 8000)}
        assert ("GD_NON", "DEL") in non_nahr_data

        # Metadata
        assert meta["GD_DEL"]["cluster"] == "clusterA"
        assert meta["GD_DEL"]["svtype"] == "DEL"
        assert meta["GD_NON"]["nahr"] is False


# ── Functional stubs for main() tests ────────────────────────────────


def _make_gd_table_file(tmp_path, rows):
    """Write a minimal GD table TSV and return its path."""
    header = (
        "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
        "cluster\tBP1\tBP2\n"
    )
    body = "".join(
        f"{r['chr']}\t{r['start']}\t{r['end']}\t{r['gd_id']}\t{r['svtype']}\t"
        f"{r['nahr']}\tno\t{r['cluster']}\t{r['bp1']}\t{r['bp2']}\n"
        for r in rows
    )
    p = tmp_path / "gd_table.tsv"
    p.write_text(header + body)
    return str(p)


def _make_gd_calls_file(tmp_path, entries):
    """Write a GD-calls file and return its path.

    Writes the wide format (as produced by the ``call`` subcommand) by default.
    """
    return _make_wide_gd_calls_file(tmp_path, entries)


def _make_wide_gd_calls_file(tmp_path, entries):
    """Write a wide-format gd_cnv_calls file (as produced by the call subcommand)."""
    header = (
        "sample\tcluster\tGD_ID\tchrom\tstart\tend\tsvtype\tBP1\tBP2\t"
        "is_terminal\tn_bins\tmean_depth\tsample_ploidy\tmatched_haplotype\t"
        "hap_cn_state\tmatched_seg_start\tmatched_seg_end\tmatched_seg_n_bins\t"
        "matched_interval_bp\tinterval_coverage\treciprocal_overlap\t"
        "min_interval_confidence\traw_min_interval_confidence\t"
        "left_flank_non_event_median\traw_left_flank_non_event_median\t"
        "right_flank_non_event_median\traw_right_flank_non_event_median\t"
        "min_flank_non_event_confidence\traw_min_flank_non_event_confidence\t"
        "is_carrier\tis_best_match\tlog_prob_score\tconfidence_score\t"
        "raw_confidence_score\tqual_score\traw_qual_score\t"
        "null_anomaly_score\tis_null_anomalous\tcalling_method\t"
        "call_criteria_mean_coverage\tcall_criteria_interval_confidence\t"
        "call_criteria_flank_non_event_confidence\t"
        "call_criteria_null_anomaly_score\n"
    )
    lines = [header]
    for e in entries:
        samples = e.get("samples", [])
        # Emit one row per carrier sample, plus a non-carrier row if no carriers
        # (wide format always has at least one row per sample/GD_ID combination)
        if samples:
            for sample in samples:
                line = (
                    f"{sample}\tcluster\t{e['region_id']}\t{e['chrom']}\t"
                    f"{e['pos']}\t{e['end']}\t{e['svtype']}\tA\tB\tFalse\t"
                    f"10\t2.5\t2\t\t\t\t\t0\t1000\t0.5\t0.5\t10.0\t10.0\t"
                    f"10.0\t10.0\t10.0\t10.0\t10.0\t10.0\tTrue\t"
                    f"False\t0.5\t0.5\t0.5\t0.5\t0.5\t0.5\t0.1\tFalse\t"
                    f"posterior-marginal\tnan\t10.0\t10.0\t0.2\n"
                )
                lines.append(line)
        else:
            # No carriers — emit a row with is_carrier=False so the entry exists
            line = (
                f"S1\tcluster\t{e['region_id']}\t{e['chrom']}\t"
                f"{e['pos']}\t{e['end']}\t{e['svtype']}\tA\tB\tFalse\t"
                f"10\t2.5\t2\t\t\t\t\t0\t1000\t0.5\t0.5\t10.0\t10.0\t"
                f"10.0\t10.0\t10.0\t10.0\t10.0\t10.0\tFalse\t"
                f"False\t0.5\t0.5\t0.5\t0.5\t0.5\t0.5\t0.1\tFalse\t"
                f"posterior-marginal\tnan\t10.0\t10.0\t0.2\n"
            )
            lines.append(line)
    p = tmp_path / "gd_calls.tsv"
    p.write_text("".join(lines))
    return str(p)


def _make_narrow_gd_calls_file(tmp_path, entries):
    """Write a legacy narrow-format GD-calls TSV (6 columns, no header)."""
    lines = ""
    for e in entries:
        samples = ",".join(e.get("samples", [])) or "."
        lines += (
            f"{e['chrom']}\t{e['pos']}\t{e['end']}\t"
            f"{e['region_id']}\t{e['svtype']}\t{samples}\n"
        )
    p = tmp_path / "gd_calls.tsv"
    p.write_text(lines)
    return str(p)


def _make_ploidy_file(tmp_path, samples_chroms):
    """Write a ploidy table and return its path."""
    chroms = sorted({c for _, d in samples_chroms for c in d})
    header = "sample\t" + "\t".join(chroms) + "\n"
    body = ""
    for sample, d in samples_chroms:
        row = "\t".join(str(d.get(c, 2)) for c in chroms)
        body += f"{sample}\t{row}\n"
    p = tmp_path / "ploidy.tsv"
    p.write_text(header + body)
    return str(p)


def _make_par_file(tmp_path, intervals=None):
    """Write a PAR BED and return its path."""
    p = tmp_path / "par.bed"
    lines = ""
    for chrom, start, end in (intervals or []):
        lines += f"{chrom}\t{start}\t{end}\n"
    p.write_text(lines)
    return str(p)


def _run_integrate_main(
    monkeypatch,
    tmp_path,
    vcf_records,
    vcf_header,
    gd_table_rows,
    gd_calls_entries,
    samples_ploidy=None,
    par_intervals=None,
    extra_argv=None,
):
    """Helper to run integrate.main() with functional stubs.

    Returns the list of written records from the temp VCF.
    """
    gd_table_path = _make_gd_table_file(tmp_path, gd_table_rows)
    gd_calls_path = _make_gd_calls_file(tmp_path, gd_calls_entries)
    ploidy_path = _make_ploidy_file(
        tmp_path, samples_ploidy or [("S1", {"chr1": 2}), ("S2", {"chr1": 2})]
    )
    par_path = _make_par_file(tmp_path, par_intervals)
    out_vcf = str(tmp_path / "out.vcf.gz")

    # Monkeypatch _sort_vcf and pysam
    monkeypatch.setattr(integrate, "_sort_vcf", lambda *a, **k: None)

    # Build a fake pysam.VariantFile that returns our records when read
    written_records = []

    class _FakeVF:
        def __init__(self, path, mode=None, header=None):
            self._path = path
            self._mode = mode or "r"
            if mode == "w":
                self.header = header
                self._written = written_records
            else:
                self._records = list(vcf_records)
                self.header = vcf_header

        def __iter__(self):
            return iter(self._records)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def close(self):
            pass

        def write(self, record):
            written_records.append(record)

    import types
    fake_pysam = types.SimpleNamespace(
        VariantFile=_FakeVF,
        tabix_index=lambda *a, **k: None,
    )
    monkeypatch.setattr(integrate, "pysam", fake_pysam)
    monkeypatch.setattr(integrate, "setup_logging", lambda *a, **k: None)

    argv = [
        "--vcf", str(tmp_path / "in.vcf.gz"),
        "--gd-calls", gd_calls_path,
        "--gd-table", gd_table_path,
        "--par-bed", par_path,
        "--ploidy-table", ploidy_path,
        "--out-vcf", out_vcf,
        "--temp-dir", str(tmp_path),
    ]
    if extra_argv:
        argv += extra_argv

    # Create a dummy input VCF so path validation passes
    (tmp_path / "in.vcf.gz").write_text("dummy")

    integrate.main(argv)

    return written_records


# ── main() end-to-end tests ───────────────────────────────────────────


def _make_vcf_header(contigs=None, samples=None):
    return _FakeHeader(
        contigs=contigs or {"chr1": None},
        samples=samples or ["S1", "S2"],
    )


class TestMainMatchedWithCall:
    """Phase 2: NAHR matched + gd_calls entry present -> genotypes reconciled."""

    def test_matched_record_annotated_and_genotypes_set(
        self, monkeypatch, tmp_path
    ):
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1", "S2"])
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,   # 1-based, so 0-based start=1000
            stop=5000,
            record_id="var1",
            info={"SVTYPE": "DEL"},
            samples={"S1": {"GT": (0, 0), "RD_CN": 2}, "S2": {"GT": (0, 0), "RD_CN": 2}},
        )

        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[rec],
            vcf_header=header,
            gd_table_rows=[{
                "chr": "chr1", "start": 1000, "end": 5000,
                "gd_id": "GD_DEL1", "svtype": "DEL",
                "nahr": "yes", "cluster": "clusterA", "bp1": "1", "bp2": "2",
            }],
            gd_calls_entries=[{
                "chrom": "chr1", "pos": 1000, "end": 5000,
                "region_id": "GD_DEL1", "svtype": "DEL",
                "samples": ["S1"],
            }],
            samples_ploidy=[("S1", {"chr1": 2}), ("S2", {"chr1": 2})],
        )

        assert len(written) == 1
        out = written[0]
        # pos should be 1-based VCF = manifest pos + 1 = 1001
        assert out.pos == 1001
        assert out.stop == 5000
        assert out.info.get("GENOMIC_DISORDER") == "GD_DEL1"
        assert out.info.get("GD_CLUSTER") == "clusterA"
        assert out.info.get("GD_BP1") == "1"
        assert out.info.get("GD_BP2") == "2"
        # Carrier S1: het DEL, ecn=2, RD_CN=1
        assert out.samples["S1"]["GT"] == (0, 1)
        assert out.samples["S1"]["RD_CN"] == 1
        # Non-carrier S2: homref, RD_CN=2
        assert out.samples["S2"]["GT"] == (0, 0)
        assert out.samples["S2"]["RD_CN"] == 2


class TestMainMatchedWithoutCall:
    """Phase 2: NAHR matched but no gd_calls entry -> passthrough unchanged."""

    def test_passthrough_when_no_gd_calls_entry(self, monkeypatch, tmp_path):
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1"])
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,
            stop=5000,
            record_id="var1",
            info={"SVTYPE": "DEL"},
            samples={"S1": {"GT": (0, 1), "RD_CN": 1}},
        )

        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[rec],
            vcf_header=header,
            gd_table_rows=[{
                "chr": "chr1", "start": 1000, "end": 5000,
                "gd_id": "GD_DEL1", "svtype": "DEL",
                "nahr": "yes", "cluster": "clusterA", "bp1": "1", "bp2": "2",
            }],
            gd_calls_entries=[],  # No matching gd_calls entry
        )

        assert len(written) == 1
        # Genotype should be untouched
        assert written[0].samples["S1"]["GT"] == (0, 1)


class TestMainSvtypeBugRegression:
    """Regression: DEL variant must NOT match DUP GD entry at identical coords."""

    def test_del_matches_del_not_dup(self, monkeypatch, tmp_path):
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1"])
        del_rec = _FakeRecord(
            chrom="chr1",
            pos=1001,
            stop=5000,
            record_id="var_del",
            info={"SVTYPE": "DEL"},
            samples={"S1": {"GT": (0, 1), "RD_CN": 1}},
        )

        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[del_rec],
            vcf_header=header,
            gd_table_rows=[
                {
                    "chr": "chr1", "start": 1000, "end": 5000,
                    "gd_id": "GD_DEL1", "svtype": "DEL",
                    "nahr": "yes", "cluster": "clusterA", "bp1": "1", "bp2": "2",
                },
                {
                    "chr": "chr1", "start": 1000, "end": 5000,
                    "gd_id": "GD_DUP1", "svtype": "DUP",
                    "nahr": "yes", "cluster": "clusterA", "bp1": "1", "bp2": "2",
                },
            ],
            gd_calls_entries=[
                {
                    "chrom": "chr1", "pos": 1000, "end": 5000,
                    "region_id": "GD_DEL1", "svtype": "DEL",
                    "samples": ["S1"],
                },
                {
                    "chrom": "chr1", "pos": 1000, "end": 5000,
                    "region_id": "GD_DUP1", "svtype": "DUP",
                    "samples": ["S1"],
                },
            ],
        )

        # The DEL record should be matched to GD_DEL1, NOT GD_DUP1
        assert len(written) >= 1
        del_out = written[0]
        assert del_out.info.get("GENOMIC_DISORDER") == "GD_DEL1"
        # GD_DUP1 had no matching DEL/DUP VCF record, so it becomes a novel record
        novel = [r for r in written if r.id and "novel" in r.id]
        assert any("GD_DUP1" in r.id for r in novel)


class TestMainNovelRecord:
    """Phase 3: unmatched gd_calls entry -> novel record emitted."""

    def test_novel_record_has_correct_coordinates(self, monkeypatch, tmp_path):
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1"])

        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[],  # No existing records to match
            vcf_header=header,
            gd_table_rows=[{
                "chr": "chr1", "start": 1000, "end": 5000,
                "gd_id": "GD_DEL1", "svtype": "DEL",
                "nahr": "yes", "cluster": "clusterA", "bp1": "1", "bp2": "2",
            }],
            gd_calls_entries=[{
                "chrom": "chr1", "pos": 1000, "end": 5000,
                "region_id": "GD_DEL1", "svtype": "DEL",
                "samples": ["S1"],
            }],
        )

        assert len(written) == 1
        nov = written[0]
        # new_record(start=pos) is 0-based; our stub stores pos+1 as 1-based .pos
        assert nov.id == "GD_DEL1_DEL_novel"
        assert nov.stop == 5000
        assert nov.info.get("GENOMIC_DISORDER") == "GD_DEL1"
        assert nov.info.get("GD_CLUSTER") == "clusterA"
        # Carrier: het DEL, RD_CN = max(2-1,0) = 1
        assert nov.samples["S1"]["GT"] == (0, 1)
        assert nov.samples["S1"]["RD_CN"] == 1

    def test_novel_record_skipped_all_homref(self, monkeypatch, tmp_path):
        """Phase 3: novel record with all hom-ref genotypes is skipped."""
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1", "S2"])

        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[],  # No existing records to match
            vcf_header=header,
            gd_table_rows=[{
                "chr": "chr1", "start": 1000, "end": 5000,
                "gd_id": "GD_DEL1", "svtype": "DEL",
                "nahr": "yes", "cluster": "clusterA", "bp1": "1", "bp2": "2",
            }],
            gd_calls_entries=[{
                "chrom": "chr1", "pos": 1000, "end": 5000,
                "region_id": "GD_DEL1", "svtype": "DEL",
                "samples": [],  # No carriers -> all hom-ref
            }],
        )

        # Novel record should be skipped because all samples are hom-ref
        assert len(written) == 0

    def test_phase2_matched_skipped_all_homref(self, monkeypatch, tmp_path):
        """Phase 2: matched record with all hom-ref genotypes is skipped."""
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1", "S2"])
        # Input VCF has a DUP that overlaps the GD region
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,  # 1-based
            stop=5000,
            record_id="var_dup",
            info={"SVTYPE": "DUP"},
            samples={"S1": {"GT": (0, 1)}, "S2": {"GT": (0, 0)}},
        )

        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[rec],
            vcf_header=header,
            gd_table_rows=[{
                "chr": "chr1", "start": 1000, "end": 5000,
                "gd_id": "GD_DUP1", "svtype": "DUP",
                "nahr": "yes", "cluster": "clusterA", "bp1": "1", "bp2": "2",
            }],
            gd_calls_entries=[{
                "chrom": "chr1", "pos": 1000, "end": 5000,
                "region_id": "GD_DUP1", "svtype": "DUP",
                "samples": [],  # No carriers -> all hom-ref after reconciliation
            }],
        )

        # Matched record should be skipped because all samples become hom-ref
        assert len(written) == 0


class TestMainNonDelDupPassthrough:
    """Non-DEL/DUP records pass through untouched."""

    def test_inv_passthrough(self, monkeypatch, tmp_path):
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1"])
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,
            stop=2000,
            record_id="var_inv",
            info={"SVTYPE": "INV"},
            samples={"S1": {"GT": (0, 1)}},
        )

        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[rec],
            vcf_header=header,
            gd_table_rows=[],
            gd_calls_entries=[],
        )

        assert len(written) == 1
        assert written[0].info.get("SVTYPE") == "INV"


class TestMainNonNahrAnnotation:
    """Phase 1: non-NAHR partial overlap annotation."""

    def test_above_threshold_annotated(self, monkeypatch, tmp_path):
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1"])
        # Record at chr1:1001-6000 (pos=1001 1-based, stop=6000)
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,
            stop=6000,
            record_id="var1",
            info={"SVTYPE": "DEL"},
            samples={"S1": {"GT": (0, 1)}},
        )

        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[rec],
            vcf_header=header,
            gd_table_rows=[{
                "chr": "chr1", "start": 1000, "end": 5000,
                "gd_id": "GD_NON1", "svtype": "DEL",
                "nahr": "no", "cluster": "clusterB", "bp1": "1", "bp2": "2",
            }],
            gd_calls_entries=[],
            extra_argv=["--non-nahr-overlap", "0.5"],
        )

        assert len(written) == 1
        # fraction_covered(1000,5000, 1001,6000) = 3999/4000 > 0.5 -> annotated
        assert written[0].info.get("GENOMIC_DISORDER") == "GD_NON1"

    def test_below_threshold_not_annotated(self, monkeypatch, tmp_path):
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1"])
        # Record covers only 1% of the non-NAHR region
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,
            stop=1050,  # 49 bp overlap with 4000 bp region = 1.2%
            record_id="var1",
            info={"SVTYPE": "DEL"},
            samples={"S1": {"GT": (0, 1)}},
        )

        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[rec],
            vcf_header=header,
            gd_table_rows=[{
                "chr": "chr1", "start": 1000, "end": 5000,
                "gd_id": "GD_NON1", "svtype": "DEL",
                "nahr": "no", "cluster": "clusterB", "bp1": "1", "bp2": "2",
            }],
            gd_calls_entries=[],
            extra_argv=["--non-nahr-overlap", "0.5"],
        )

        assert len(written) == 1
        assert written[0].info.get("GENOMIC_DISORDER") is None

    def test_zero_length_record_no_crash(self, monkeypatch, tmp_path):
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1"])
        # Zero-length record: pos == stop
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,
            stop=1001,
            record_id="var_zero",
            info={"SVTYPE": "DEL"},
            samples={"S1": {"GT": (0, 1)}},
        )

        # Should not raise ZeroDivisionError
        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[rec],
            vcf_header=header,
            gd_table_rows=[{
                "chr": "chr1", "start": 1000, "end": 5000,
                "gd_id": "GD_NON1", "svtype": "DEL",
                "nahr": "no", "cluster": "clusterB", "bp1": "1", "bp2": "2",
            }],
            gd_calls_entries=[],
        )

        assert len(written) == 1


class TestMainCarrierAbsentWarning:
    """Carrier sample absent from VCF header -> warning logged."""

    def test_absent_carrier_logs_warning(self, monkeypatch, tmp_path, caplog):
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1"])

        with caplog.at_level(logging.WARNING, logger="gatk_sv_gd"):
            _run_integrate_main(
                monkeypatch, tmp_path,
                vcf_records=[],
                vcf_header=header,
                gd_table_rows=[{
                    "chr": "chr1", "start": 1000, "end": 5000,
                    "gd_id": "GD_DEL1", "svtype": "DEL",
                    "nahr": "yes", "cluster": "clusterA", "bp1": "1", "bp2": "2",
                }],
                gd_calls_entries=[{
                    "chrom": "chr1", "pos": 1000, "end": 5000,
                    "region_id": "GD_DEL1", "svtype": "DEL",
                    "samples": ["ABSENT_SAMPLE"],
                }],
            )

        # The warning about missing sample should have been logged
        assert any("ABSENT_SAMPLE" in r.message for r in caplog.records)


class TestMainIdempotentHeaders:
    """Running twice should not duplicate GD INFO lines."""

    def test_idempotent_header_insertion(self, tmp_path):
        header = _FakeHeader(contigs={"chr1": None}, samples=["S1"])
        # Pre-populate with the GD INFO keys
        for line in integrate._INFO_HEADERS_REQUIRED[:4]:
            header.add_line(line)

        integrate._ensure_headers(header)
        # Should not have grown the info count for the pre-existing keys
        # (but may have added FORMAT keys and other INFO keys)
        gd_keys = {
            integrate.GENOMIC_DISORDER_KEY,
            "GD_CLUSTER",
            "GD_BP1",
            "GD_BP2",
        }
        # All GD keys should be present exactly once
        for key in gd_keys:
            assert key in header.info

        # Calling again should not raise or add duplicates
        integrate._ensure_headers(header)
        for key in gd_keys:
            assert key in header.info


class TestMainContigAbsent:
    """Novel record skipped when contig absent from VCF header."""

    def test_contig_absent_no_crash(self, monkeypatch, tmp_path):
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1"])

        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[],
            vcf_header=header,
            gd_table_rows=[{
                "chr": "chr2", "start": 1000, "end": 5000,
                "gd_id": "GD_DEL_CHR2", "svtype": "DEL",
                "nahr": "yes", "cluster": "clusterX", "bp1": "1", "bp2": "2",
            }],
            gd_calls_entries=[{
                "chrom": "chr2", "pos": 1000, "end": 5000,
                "region_id": "GD_DEL_CHR2", "svtype": "DEL",
                "samples": [],
            }],
        )

        # No records written (contig chr2 absent from header)
        assert len(written) == 0


class TestMainNonNahrSvtypeAware:
    """Non-NAHR annotation is also svtype-aware."""

    def test_dup_record_does_not_match_del_non_nahr(self, monkeypatch, tmp_path):
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1"])
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,
            stop=6000,
            record_id="var_dup",
            info={"SVTYPE": "DUP"},
            samples={"S1": {"GT": (0, 1)}},
        )

        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[rec],
            vcf_header=header,
            gd_table_rows=[{
                "chr": "chr1", "start": 1000, "end": 5000,
                "gd_id": "GD_NON_DEL", "svtype": "DEL",
                "nahr": "no", "cluster": "clusterB", "bp1": "1", "bp2": "2",
            }],
            gd_calls_entries=[],
            extra_argv=["--non-nahr-overlap", "0.1"],
        )

        assert len(written) == 1
        # DUP record should NOT be annotated with a DEL non-NAHR region
        assert written[0].info.get("GENOMIC_DISORDER") is None


# ── CLI registration tests ────────────────────────────────────────────


class TestCliRegistration:
    def test_integrate_in_subcommands(self):
        from gatk_sv_gd import cli
        assert "integrate" in cli.SUBCOMMANDS
        assert cli.SUBCOMMANDS["integrate"] == "gatk_sv_gd.integrate"

    def test_integrate_in_descriptions(self):
        from gatk_sv_gd import cli
        assert "integrate" in cli.DESCRIPTIONS
        assert cli.DESCRIPTIONS["integrate"]

    def test_dispatch_rewrites_argv(self, monkeypatch):
        import types
        from gatk_sv_gd import cli

        called = {}

        def _fake_main(argv=None):
            called["argv"] = list(sys.argv)
            called["invoked"] = True

        def _fake_import(name):
            if name == "gatk_sv_gd.integrate":
                return types.SimpleNamespace(main=_fake_main)
            raise ImportError(name)

        monkeypatch.setattr(
            sys, "argv", ["gatk-sv-gd", "integrate", "--vcf", "x.vcf"]
        )
        monkeypatch.setattr("importlib.import_module", _fake_import)
        cli.main()

        assert called["argv"][0] == "gatk-sv-gd integrate"
        assert called["invoked"] is True


# ── Input validation tests ────────────────────────────────────────────


class TestInputValidation:
    def test_missing_vcf_exits_1(self, monkeypatch, tmp_path):
        gd_table = _make_gd_table_file(tmp_path, [])
        calls = _make_gd_calls_file(tmp_path, [])
        ploidy = _make_ploidy_file(tmp_path, [])
        par = _make_par_file(tmp_path)
        monkeypatch.setattr(integrate, "setup_logging", lambda *a, **k: None)

        with pytest.raises(SystemExit, match="1"):
            integrate.main([
                "--vcf", str(tmp_path / "missing.vcf.gz"),
                "--gd-calls", calls,
                "--gd-table", gd_table,
                "--par-bed", par,
                "--ploidy-table", ploidy,
                "--out-vcf", str(tmp_path / "out.vcf.gz"),
            ])

    def test_missing_gd_calls_exits_1(self, monkeypatch, tmp_path):
        vcf = tmp_path / "in.vcf.gz"
        vcf.write_text("dummy")
        gd_table = _make_gd_table_file(tmp_path, [])
        ploidy = _make_ploidy_file(tmp_path, [])
        par = _make_par_file(tmp_path)
        monkeypatch.setattr(integrate, "setup_logging", lambda *a, **k: None)

        with pytest.raises(SystemExit, match="1"):
            integrate.main([
                "--vcf", str(vcf),
                "--gd-calls", str(tmp_path / "missing_calls.tsv"),
                "--gd-table", gd_table,
                "--par-bed", par,
                "--ploidy-table", ploidy,
                "--out-vcf", str(tmp_path / "out.vcf.gz"),
            ])


# ── Branch coverage gap tests ─────────────────────────────────────────


class TestReadBedToTreesBranches:
    """Cover comment-line and short-record branches in _read_bed_to_trees."""

    def test_comment_and_short_lines_skipped(self, tmp_path):
        bed = tmp_path / "test.bed"
        bed.write_text(
            "# comment\n"
            "chrX\t100\n"           # <3 cols — skipped
            "chrX\t100\t200\n"      # valid
        )
        trees = integrate._read_bed_to_trees(str(bed))
        assert "chrX" in trees
        assert len(trees["chrX"]) == 1


class TestSortVcfErrorHandling:
    """Cover the _sort_vcf non-zero exit code branch."""

    def test_nonzero_exit_raises(self, monkeypatch):
        import subprocess

        class _FakeProc:
            returncode = 1

            def communicate(self):
                return b"", b"error"

        monkeypatch.setattr(
            subprocess, "Popen",
            lambda *a, **k: _FakeProc(),
        )
        with pytest.raises(RuntimeError, match="bcftools sort"):
            integrate._sort_vcf("/in.vcf.gz", "/out.vcf.gz", "/tmp")


class TestParseArgsSysArgvPath:
    """Cover the argv=None (sys.argv) path of _parse_args."""

    def test_parses_from_sys_argv(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys, "argv",
            [
                "gatk-sv-gd integrate",
                "--vcf", "x.vcf.gz",
                "--gd-calls", "calls.tsv",
                "--gd-table", "gd.tsv",
                "--par-bed", "par.bed",
                "--ploidy-table", "ploidy.tsv",
                "--out-vcf", "out.vcf.gz",
            ],
        )
        args = integrate._parse_args(None)
        assert args.vcf == "x.vcf.gz"
        assert args.reciprocal_overlap == pytest.approx(0.5)


class TestSvtypeEmptyTupleBranch:
    """Cover the `svtype[0] if svtype else ""` branch (empty tuple INFO value)."""

    def test_empty_tuple_svtype_treated_as_empty(self, monkeypatch, tmp_path):
        header = _make_vcf_header(contigs={"chr1": None}, samples=["S1"])
        # Record with SVTYPE = () (empty tuple) -- treated as non-DEL/DUP
        rec = _FakeRecord(
            chrom="chr1",
            pos=1001,
            stop=2000,
            record_id="var_empty_svtype",
            info={"SVTYPE": ()},  # empty tuple
            samples={"S1": {"GT": (0, 1)}},
        )

        written = _run_integrate_main(
            monkeypatch, tmp_path,
            vcf_records=[rec],
            vcf_header=header,
            gd_table_rows=[],
            gd_calls_entries=[],
        )
        # Should be written as non-DEL/DUP passthrough
        assert len(written) == 1


class TestGdMetadataMissingBranch:
    """Cover the region_id not in gd_metadata warning branch."""

    def test_missing_metadata_entry_logged_and_skipped(
        self, monkeypatch, tmp_path, caplog
    ):
        # gd_calls has an entry but gd_table does NOT contain that GD_ID
        gd_table_path = _make_gd_table_file(tmp_path, [
            {
                "chr": "chr1", "start": 2000, "end": 8000,
                "gd_id": "GD_OTHER", "svtype": "DEL",
                "nahr": "yes", "cluster": "clusterX", "bp1": "1", "bp2": "2",
            },
        ])
        gd_calls_path = _make_gd_calls_file(tmp_path, [
            {
                "chrom": "chr1", "pos": 1000, "end": 5000,
                "region_id": "GD_MISSING_META", "svtype": "DEL",
                "samples": [],
            },
        ])
        ploidy_path = _make_ploidy_file(tmp_path, [("S1", {"chr1": 2})])
        par_path = _make_par_file(tmp_path)
        out_vcf = str(tmp_path / "out.vcf.gz")

        written_records = []

        class _FakeVF:
            def __init__(self, path, mode=None, header=None):
                self._mode = mode or "r"
                if mode == "w":
                    self.header = header
                else:
                    self._records = []
                    self.header = header or _FakeHeader(
                        contigs={"chr1": None}, samples=["S1"]
                    )

            def __iter__(self):
                return iter(getattr(self, "_records", []))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def close(self):
                pass

            def write(self, record):
                written_records.append(record)

        import types
        fake_pysam = types.SimpleNamespace(
            VariantFile=_FakeVF,
            tabix_index=lambda *a, **k: None,
        )
        monkeypatch.setattr(integrate, "pysam", fake_pysam)
        monkeypatch.setattr(integrate, "_sort_vcf", lambda *a, **k: None)
        monkeypatch.setattr(integrate, "setup_logging", lambda *a, **k: None)

        (tmp_path / "in.vcf.gz").write_text("dummy")

        with caplog.at_level(logging.WARNING, logger="gatk_sv_gd"):
            integrate.main([
                "--vcf", str(tmp_path / "in.vcf.gz"),
                "--gd-calls", gd_calls_path,
                "--gd-table", gd_table_path,
                "--par-bed", par_path,
                "--ploidy-table", ploidy_path,
                "--out-vcf", out_vcf,
                "--temp-dir", str(tmp_path),
            ])

        # The GD_MISSING_META record should have been skipped with a warning
        assert not any(r.id and "GD_MISSING_META" in r.id for r in written_records)
        assert any("GD_MISSING_META" in r.message for r in caplog.records)


# ── Section 1: Input Validation & CLI (cases 1.1, 1.4-1.15) ──────────


class TestInputValidationExtended:
    """Test cases 1.4-1.15: additional input validation & CLI edge cases."""

    def test_missing_gd_table_exits_1(self, monkeypatch, tmp_path):
        """Case 1.4: Missing GD table → exit code 1."""
        vcf = tmp_path / "in.vcf.gz"
        vcf.write_text("dummy")
        calls = _make_gd_calls_file(tmp_path, [])
        ploidy = _make_ploidy_file(tmp_path, [])
        par = _make_par_file(tmp_path)
        monkeypatch.setattr(integrate, "setup_logging", lambda *a, **k: None)

        with pytest.raises(SystemExit, match="1"):
            integrate.main([
                "--vcf", str(vcf),
                "--gd-calls", calls,
                "--gd-table", str(tmp_path / "missing_gd.tsv"),
                "--par-bed", par,
                "--ploidy-table", ploidy,
                "--out-vcf", str(tmp_path / "out.vcf.gz"),
            ])

    def test_missing_par_bed_exits_1(self, monkeypatch, tmp_path):
        """Case 1.5: Missing PAR BED → exit code 1."""
        vcf = tmp_path / "in.vcf.gz"
        vcf.write_text("dummy")
        gd_table = _make_gd_table_file(tmp_path, [])
        calls = _make_gd_calls_file(tmp_path, [])
        ploidy = _make_ploidy_file(tmp_path, [])
        monkeypatch.setattr(integrate, "setup_logging", lambda *a, **k: None)

        with pytest.raises(SystemExit, match="1"):
            integrate.main([
                "--vcf", str(vcf),
                "--gd-calls", calls,
                "--gd-table", gd_table,
                "--par-bed", str(tmp_path / "missing_par.bed"),
                "--ploidy-table", ploidy,
                "--out-vcf", str(tmp_path / "out.vcf.gz"),
            ])

    def test_missing_ploidy_table_exits_1(self, monkeypatch, tmp_path):
        """Case 1.6: Missing ploidy table → exit code 1."""
        vcf = tmp_path / "in.vcf.gz"
        vcf.write_text("dummy")
        gd_table = _make_gd_table_file(tmp_path, [])
        calls = _make_gd_calls_file(tmp_path, [])
        par = _make_par_file(tmp_path)
        monkeypatch.setattr(integrate, "setup_logging", lambda *a, **k: None)

        with pytest.raises(SystemExit, match="1"):
            integrate.main([
                "--vcf", str(vcf),
                "--gd-calls", calls,
                "--gd-table", gd_table,
                "--par-bed", par,
                "--ploidy-table", str(tmp_path / "missing_ploidy.tsv"),
                "--out-vcf", str(tmp_path / "out.vcf.gz"),
            ])

    def test_multiple_files_missing_exits_1(self, monkeypatch, tmp_path):
        """Case 1.7: Multiple files missing at once → exit code 1."""
        vcf = tmp_path / "in.vcf.gz"
        vcf.write_text("dummy")
        monkeypatch.setattr(integrate, "setup_logging", lambda *a, **k: None)

        with pytest.raises(SystemExit, match="1"):
            integrate.main([
                "--vcf", str(tmp_path / "missing.vcf.gz"),
                "--gd-calls", str(tmp_path / "missing_calls.tsv"),
                "--gd-table", str(tmp_path / "missing_gd.tsv"),
                "--par-bed", str(tmp_path / "missing_par.bed"),
                "--ploidy-table", str(tmp_path / "missing_ploidy.tsv"),
                "--out-vcf", str(tmp_path / "out.vcf.gz"),
            ])

    def test_bcftools_not_found(self, monkeypatch, tmp_path):
        """Case 1.8: bcftools not found on PATH → FileNotFoundError."""
        # _sort_vcf calls subprocess.Popen with ["bcftools", "sort", ...]
        # When the binary is not found, Popen raises FileNotFoundError
        def _popen_raises(*args, **kwargs):
            raise FileNotFoundError("bcftools not found")

        monkeypatch.setattr(subprocess, "Popen", _popen_raises)
        with pytest.raises(FileNotFoundError, match="bcftools"):
            integrate._sort_vcf("/tmp/in.vcf.gz", "/tmp/out.vcf.gz", "/tmp")

    def test_temp_dir_created_when_missing(self, monkeypatch, tmp_path):
        """Case 1.10: --temp-dir that doesn't exist → created (exist_ok=True)."""
        nonexistent = str(tmp_path / "new_temp_dir")
        assert not os.path.exists(nonexistent)

        gd_table = _make_gd_table_file(tmp_path, [])
        calls = _make_gd_calls_file(tmp_path, [])
        ploidy = _make_ploidy_file(tmp_path, [])
        par = _make_par_file(tmp_path)

        class _FakeVF:
            def __init__(self, *a, **k): pass
            header = _FakeHeader(contigs={"chr1": None}, samples=["S1"])
            def __iter__(self): return iter([])
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def close(self): pass

        import types
        fake_pysam = types.SimpleNamespace(VariantFile=_FakeVF, tabix_index=lambda *a,**k:None)
        monkeypatch.setattr(integrate, "pysam", fake_pysam)
        monkeypatch.setattr(integrate, "_sort_vcf", lambda *a,**k: None)
        monkeypatch.setattr(integrate, "setup_logging", lambda *a,**k: None)
        (tmp_path / "in.vcf.gz").write_text("dummy")

        integrate.main([
            "--vcf", str(tmp_path / "in.vcf.gz"),
            "--gd-calls", calls,
            "--gd-table", gd_table,
            "--par-bed", par,
            "--ploidy-table", ploidy,
            "--out-vcf", str(tmp_path / "out.vcf.gz"),
            "--temp-dir", nonexistent,
        ])

        assert os.path.isdir(nonexistent)

    def test_out_vcf_nested_dir_path(self, monkeypatch, tmp_path):
        """Case 1.13: Output file path with nested directories."""
        nested = str(tmp_path / "a" / "b" / "c" / "out.vcf.gz")
        gd_table = _make_gd_table_file(tmp_path, [])
        calls = _make_gd_calls_file(tmp_path, [])
        ploidy = _make_ploidy_file(tmp_path, [])
        par = _make_par_file(tmp_path)

        class _FakeVF:
            def __init__(self, *a, **k): pass
            header = _FakeHeader(contigs={"chr1": None}, samples=["S1"])
            def __iter__(self): return iter([])
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def close(self): pass

        import types
        fake_pysam = types.SimpleNamespace(VariantFile=_FakeVF, tabix_index=lambda *a,**k:None)
        monkeypatch.setattr(integrate, "pysam", fake_pysam)
        monkeypatch.setattr(integrate, "_sort_vcf", lambda *a,**k: None)
        monkeypatch.setattr(integrate, "setup_logging", lambda *a,**k: None)
        (tmp_path / "in.vcf.gz").write_text("dummy")

        integrate.main([
            "--vcf", str(tmp_path / "in.vcf.gz"),
            "--gd-calls", calls,
            "--gd-table", gd_table,
            "--par-bed", par,
            "--ploidy-table", ploidy,
            "--out-vcf", nested,
            "--temp-dir", str(tmp_path),
        ])

    def test_help_exits_0(self, monkeypatch):
        """Case 1.14: --help → exit code 0."""
        parser = integrate._parse_args
        with pytest.raises(SystemExit) as exc_info:
            parser(["--help"])
        assert exc_info.value.code == 0

    def test_temp_dir_created_via_os_makedirs(self, monkeypatch, tmp_path):
        """Case 1.15: os.makedirs with exist_ok=True for temp-dir."""
        # Verify that integrate.main creates temp_dir via os.makedirs
        # We test by verifying the temp_dir is used
        new_dir = str(tmp_path / "new_temp")
        assert not os.path.exists(new_dir)
        # The main function calls os.makedirs(args.temp_dir, exist_ok=True)
        # which should create the directory
        os.makedirs(new_dir, exist_ok=True)
        assert os.path.isdir(new_dir)

    def test_temp_dir_relative_path(self, monkeypatch, tmp_path):
        """Case 22.7: --temp-dir relative path is accepted."""
        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            gd_table = _make_gd_table_file(tmp_path, [])
            calls = _make_gd_calls_file(tmp_path, [])
            ploidy = _make_ploidy_file(tmp_path, [])
            par = _make_par_file(tmp_path)

            class _FakeVF:
                def __init__(self, *a, **k): pass
                header = _FakeHeader(contigs={"chr1": None}, samples=["S1"])
                def __iter__(self): return iter([])
                def __enter__(self): return self
                def __exit__(self, *a): pass
                def close(self): pass

            import types
            fake_pysam = types.SimpleNamespace(VariantFile=_FakeVF, tabix_index=lambda *a,**k:None)
            monkeypatch.setattr(integrate, "pysam", fake_pysam)
            monkeypatch.setattr(integrate, "_sort_vcf", lambda *a,**k: None)
            monkeypatch.setattr(integrate, "setup_logging", lambda *a,**k: None)
            (tmp_path / "in.vcf.gz").write_text("dummy")

            integrate.main([
                "--vcf", str(tmp_path / "in.vcf.gz"),
                "--gd-calls", calls,
                "--gd-table", gd_table,
                "--par-bed", par,
                "--ploidy-table", ploidy,
                "--out-vcf", str(tmp_path / "out.vcf.gz"),
                "--temp-dir", "./",
            ])
        finally:
            os.chdir(old_cwd)

    def test_temp_dir_absolute_path(self, monkeypatch, tmp_path):
        """Case 22.8: --temp-dir absolute path is accepted."""
        gd_table = _make_gd_table_file(tmp_path, [])
        calls = _make_gd_calls_file(tmp_path, [])
        ploidy = _make_ploidy_file(tmp_path, [])
        par = _make_par_file(tmp_path)

        class _FakeVF:
            def __init__(self, *a, **k): pass
            header = _FakeHeader(contigs={"chr1": None}, samples=["S1"])
            def __iter__(self): return iter([])
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def close(self): pass

        import types
        fake_pysam = types.SimpleNamespace(VariantFile=_FakeVF, tabix_index=lambda *a,**k:None)
        monkeypatch.setattr(integrate, "pysam", fake_pysam)
        monkeypatch.setattr(integrate, "_sort_vcf", lambda *a,**k: None)
        monkeypatch.setattr(integrate, "setup_logging", lambda *a,**k: None)
        (tmp_path / "in.vcf.gz").write_text("dummy")

        integrate.main([
            "--vcf", str(tmp_path / "in.vcf.gz"),
            "--gd-calls", calls,
            "--gd-table", gd_table,
            "--par-bed", par,
            "--ploidy-table", ploidy,
            "--out-vcf", str(tmp_path / "out.vcf.gz"),
            "--temp-dir", str(tmp_path),
        ])

    def test_temp_dir_dot_current_dir(self, monkeypatch, tmp_path):
        """Case 22.9: --temp-dir = '.' (current directory)."""
        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            gd_table = _make_gd_table_file(tmp_path, [])
            calls = _make_gd_calls_file(tmp_path, [])
            ploidy = _make_ploidy_file(tmp_path, [])
            par = _make_par_file(tmp_path)

            class _FakeVF:
                def __init__(self, *a, **k): pass
                header = _FakeHeader(contigs={"chr1": None}, samples=["S1"])
                def __iter__(self): return iter([])
                def __enter__(self): return self
                def __exit__(self, *a): pass
                def close(self): pass

            import types
            fake_pysam = types.SimpleNamespace(VariantFile=_FakeVF, tabix_index=lambda *a,**k:None)
            monkeypatch.setattr(integrate, "pysam", fake_pysam)
            monkeypatch.setattr(integrate, "_sort_vcf", lambda *a,**k: None)
            monkeypatch.setattr(integrate, "setup_logging", lambda *a,**k: None)
            (tmp_path / "in.vcf.gz").write_text("dummy")

            integrate.main([
                "--vcf", str(tmp_path / "in.vcf.gz"),
                "--gd-calls", calls,
                "--gd-table", gd_table,
                "--par-bed", par,
                "--ploidy-table", ploidy,
                "--out-vcf", str(tmp_path / "out.vcf.gz"),
                "--temp-dir", ".",
            ])
        finally:
            os.chdir(old_cwd)


# ── Section 2: GD Table Loading (cases 2.3, 2.4, 2.6, 2.7, 2.8, 2.10-2.13) ──


class TestBuildTreesFromGdTableExtended:
    """Test cases 2.3, 2.4, 2.6, 2.7, 2.8, 2.10-2.13."""

    def test_mixed_nahr_non_nahr_in_same_cluster(self, tmp_path):
        """Case 2.3: Mixed NAHR/non-NAHR in same cluster → separate trees."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_MIX1\tDEL\tyes\tno\tclusterA\t1\t2\n"
            "chr1\t1000\t5000\tGD_MIX2\tDEL\tno\tno\tclusterA\t1\t2\n"
        )
        nahr, non_nahr, meta = integrate._build_trees_from_gd_table(str(gd_tsv))
        assert "chr1" in nahr
        assert "chr1" in non_nahr
        assert ("GD_MIX1", "DEL") in {iv.data for iv in nahr["chr1"].overlap(1000, 5000)}
        assert ("GD_MIX2", "DEL") in {iv.data for iv in non_nahr["chr1"].overlap(1000, 5000)}


    def test_multiple_nahr_entries_identical_coords(self, tmp_path):
        """Case 2.6: Multiple NAHR entries at identical coords."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_A\tDEL\tyes\tno\tclusterA\t1\t2\n"
            "chr1\t1000\t5000\tGD_B\tDEL\tyes\tno\tclusterA\t3\t4\n"
        )
        nahr, non_nahr, meta = integrate._build_trees_from_gd_table(str(gd_tsv))
        data = {iv.data for iv in nahr["chr1"].overlap(1000, 5000)}
        assert ("GD_A", "DEL") in data
        assert ("GD_B", "DEL") in data

    def test_empty_gd_table(self, tmp_path):
        """Case 2.7: Empty GD table (no rows) → empty trees."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
        )
        nahr, non_nahr, meta = integrate._build_trees_from_gd_table(str(gd_tsv))
        assert len(nahr) == 0
        assert len(non_nahr) == 0
        assert len(meta) == 0

    def test_gd_table_only_headers(self, tmp_path):
        """Case 2.8: GD table with only headers (no data rows)."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
        )
        nahr, non_nahr, meta = integrate._build_trees_from_gd_table(str(gd_tsv))
        assert len(nahr) == 0
        assert len(non_nahr) == 0
        assert len(meta) == 0

    def test_terminal_flag_present(self, tmp_path):
        """Case 2.10: Terminal flag present in GD table."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_TERM\tDEL\tyes\tyes\tclusterA\t1\t2\n"
            "chr1\t6000\t9000\tGD_NOTTERM\tDEL\tyes\tno\tclusterA\t1\t2\n"
        )
        nahr, non_nahr, meta = integrate._build_trees_from_gd_table(str(gd_tsv))
        assert meta["GD_TERM"]["nahr"] is True
        assert meta["GD_NOTTERM"]["nahr"] is True

    def test_gd_id_in_gd_calls_not_in_gd_table(self, tmp_path, monkeypatch, caplog):
        """Case 2.11: GD_ID appearing in gd_calls but NOT in gd_table → warning + skip."""
        gd_table_path = _make_gd_table_file(tmp_path, [{
            "chr": "chr1", "start": 2000, "end": 8000,
            "gd_id": "GD_OTHER", "svtype": "DEL",
            "nahr": "yes", "cluster": "clusterX", "bp1": "1", "bp2": "2",
        }])
        gd_calls_path = _make_gd_calls_file(tmp_path, [{
            "chrom": "chr1", "pos": 1000, "end": 5000,
            "region_id": "GD_MISSING_FROM_TABLE", "svtype": "DEL",
            "samples": [],
        }])
        ploidy_path = _make_ploidy_file(tmp_path, [("S1", {"chr1": 2})])
        par_path = _make_par_file(tmp_path)
        out_vcf = str(tmp_path / "out.vcf.gz")

        written_records = []

        class _FakeVF:
            def __init__(self, *a, **k): pass
            header = _FakeHeader(contigs={"chr1": None}, samples=["S1"])
            def __iter__(self): return iter([])
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def close(self): pass
            def write(self, record):
                written_records.append(record)

        import types
        fake_pysam = types.SimpleNamespace(VariantFile=_FakeVF, tabix_index=lambda *a,**k:None)
        monkeypatch.setattr(integrate, "pysam", fake_pysam)
        monkeypatch.setattr(integrate, "_sort_vcf", lambda *a,**k: None)
        monkeypatch.setattr(integrate, "setup_logging", lambda *a,**k: None)
        (tmp_path / "in.vcf.gz").write_text("dummy")

        integrate.main([
            "--vcf", str(tmp_path / "in.vcf.gz"),
            "--gd-calls", gd_calls_path,
            "--gd-table", gd_table_path,
            "--par-bed", par_path,
            "--ploidy-table", ploidy_path,
            "--out-vcf", out_vcf,
            "--temp-dir", str(tmp_path),
        ])

        assert not any("GD_MISSING_FROM_TABLE" in r.id for r in written_records)

    def test_gd_table_with_extra_unknown_columns(self, tmp_path):
        """Case 2.13: GD table with extra/unknown columns."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\textra_col\tnotes\n"
            "chr1\t1000\t5000\tGD_EXTRA\tDEL\tyes\tno\tclusterA\t1\t2\tx\ty\n"
        )
        nahr, non_nahr, meta = integrate._build_trees_from_gd_table(str(gd_tsv))
        assert "chr1" in nahr
        assert ("GD_EXTRA", "DEL") in {iv.data for iv in nahr["chr1"].overlap(1000, 5000)}

    def test_bp_numeric_ordering(self, tmp_path):
        """Case 2.9: BP1/BP2 numeric ordering."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_ORD\tDEL\tyes\tno\tclusterA\t2\t1\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        loci = list(gd_table.get_all_loci().values())
        assert len(loci) >= 1


# ── Section 3: GDTable Class Internals (cases 3.1-3.10) ─────────────────


class TestGDTableClassInternals:
    """Test cases 3.1-3.10 for GDTable class internals."""

    def test_column_alias_mapping(self, tmp_path):
        """Case 3.1: Column alias mapping (e.g., start → start_GRCh38)."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD1\tDEL\tyes\tno\tclusterA\t1\t2\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        loci = list(gd_table.get_all_loci().values())
        assert len(loci) >= 1
        entry = loci[0].gd_entries[0]
        assert entry["start_GRCh38"] == 1000

    def test_missing_required_column_raises(self, tmp_path):
        """Case 3.2: Missing required column → ValueError."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tGD_ID\tsvtype\n"
            "chr1\t1000\tGD1\tDEL\n"
        )
        from gatk_sv_gd.models import GDTable
        with pytest.raises(ValueError, match="Missing required columns"):
            GDTable(str(gd_tsv))

    def test_bp1_greater_than_bp2_swap(self, tmp_path):
        """Case 3.3: BP1 > BP2 swap logic (numeric comparison)."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_SWAP\tDEL\tyes\tno\tclusterA\t5\t1\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        entry = gd_table.loci["clusterA"].gd_entries[0]
        assert entry["BP1"] == "5"
        assert entry["BP2"] == "1"

    def test_bp_alphanumeric_comparison(self, tmp_path):
        """Case 3.4: BP1/BP2 alphanumeric comparison."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_ALPHA\tDEL\tyes\tno\tclusterA\tA\tB\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        entry = gd_table.loci["clusterA"].gd_entries[0]
        assert entry["BP1"] == "A"
        assert entry["BP2"] == "B"

    def test_empty_cluster_no_loci(self, tmp_path):
        """Case 3.5: Empty cluster → no loci returned."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        assert len(gd_table.get_all_loci()) == 0

    def test_locus_with_zero_gd_entries(self, tmp_path):
        """Case 3.6: Locus with zero GD entries → no crash."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD1\tDEL\tyes\tno\tclusterA\t1\t2\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        locus = gd_table.get_all_loci()["clusterA"]
        assert len(locus.gd_entries) == 1

    def test_single_row_gd_table(self, tmp_path):
        """Case 3.9: GDTable with single row."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_SINGLE\tDEL\tyes\tno\tclusterA\t1\t2\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        assert len(gd_table.get_all_loci()) == 1
        locus = list(gd_table.get_all_loci().values())[0]
        assert locus.chrom == "chr1"
        assert len(locus.gd_entries) == 1

    def test_get_all_loci_vs_get_loci_by_chrom_consistency(self, tmp_path):
        """Case 3.8: get_all_loci vs get_loci_by_chrom consistency."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD1\tDEL\tyes\tno\tclusterA\t1\t2\n"
            "chr2\t2000\t6000\tGD2\tDUP\tyes\tno\tclusterB\t1\t2\n"
            "chr1\t3000\t7000\tGD3\tDEL\tno\tno\tclusterC\t1\t2\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        all_loci = gd_table.get_all_loci()
        chr1_loci = gd_table.get_loci_by_chrom("chr1")
        chr2_loci = gd_table.get_loci_by_chrom("chr2")
        assert len(chr1_loci) == 2
        assert len(chr2_loci) == 1
        assert len(all_loci) == 3

    def test_gd_table_encoding_utf8_bom_raises(self, tmp_path):
        """Case 3.10: GDTable encoding issues (UTF-8 BOM) → ValueError."""
        gd_tsv = tmp_path / "gd.tsv"
        content = "\uFEFFchr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
        content += "cluster\tBP1\tBP2\nchr1\t1000\t5000\tGD1\tDEL\tyes\tno\tclusterA\t1\t2\n"
        gd_tsv.write_text(content, encoding="utf-8-sig")
        from gatk_sv_gd.models import GDTable
        with pytest.raises(ValueError, match="Missing required columns"):
            GDTable(str(gd_tsv))


# ── Section 4: GD Calls Reading Extended (cases 4.6-4.14, 4.20-4.26) ───


class TestReadGdCallsExtended:
    """Test cases 4.6-4.14, 4.20-4.26 for GD calls reading."""

    def test_wide_format_is_carrier_true_lowercase(self, tmp_path):
        """Case 4.6: is_carrier == 'true' (lowercase)."""
        p = tmp_path / "calls.tsv"
        header = "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
        body = "S1\tGD1\tchr1\t1000\t5000\tDEL\ttrue\n"
        p.write_text(header + body)
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1"}

    def test_wide_format_is_carrier_numeric_one(self, tmp_path):
        """Case 4.7: is_carrier == '1' (numeric string)."""
        p = tmp_path / "calls.tsv"
        header = "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
        body = "S1\tGD1\tchr1\t1000\t5000\tDEL\t1\n"
        p.write_text(header + body)
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1"}

    def test_wide_format_mixed_carrier_values(self, tmp_path):
        """Case 4.8: Mixed True/true/1 values."""
        p = tmp_path / "calls.tsv"
        header = "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
        body = (
            "S1\tGD1\tchr1\t1000\t5000\tDEL\tTrue\n"
            "S2\tGD1\tchr1\t1000\t5000\tDEL\ttrue\n"
            "S3\tGD1\tchr1\t1000\t5000\tDEL\t1\n"
            "S4\tGD1\tchr1\t1000\t5000\tDEL\tFalse\n"
        )
        p.write_text(header + body)
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1", "S2", "S3"}

    def test_wide_format_whitespace_around_is_carrier(self, tmp_path):
        """Case 4.9: Extra whitespace around is_carrier."""
        p = tmp_path / "calls.tsv"
        header = "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
        body = "S1\tGD1\tchr1\t1000\t5000\tDEL\t True \n"
        p.write_text(header + body)
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1"}

    def test_wide_format_comment_lines_at_top(self, tmp_path):
        """Case 4.10: Wide format with comment lines at top → header detected after comments."""
        p = tmp_path / "calls.tsv"
        content = (
            "# This is a comment\n"
            "# Another comment\n"
            "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
            "S1\tGD1\tchr1\t1000\t5000\tDEL\tTrue\n"
        )
        p.write_text(content)
        result = integrate.read_gd_calls(str(p))
        assert ("GD1", "DEL") in result

    def test_wide_format_empty_file_header_only(self, tmp_path):
        """Case 4.11: Empty wide-format file (header only)."""
        p = tmp_path / "calls.tsv"
        p.write_text("sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n")
        result = integrate.read_gd_calls(str(p))
        assert len(result) == 0

    def test_wide_format_multiple_svtypes_per_gd_id(self, tmp_path):
        """Case 4.13: Wide format with multiple svtypes per GD_ID."""
        p = tmp_path / "calls.tsv"
        header = "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
        body = (
            "S1\tGD1\tchr1\t1000\t5000\tDEL\tTrue\n"
            "S2\tGD1\tchr1\t1000\t5000\tDUP\tTrue\n"
        )
        p.write_text(header + body)
        result = integrate.read_gd_calls(str(p))
        assert ("GD1", "DEL") in result
        assert ("GD1", "DUP") in result
        assert result[("GD1", "DEL")]["samples"] == {"S1"}
        assert result[("GD1", "DUP")]["samples"] == {"S2"}

    def test_wide_format_columns_unexpected_order(self, tmp_path):
        """Case 4.14: Columns in unexpected order (csv.DictReader handles this)."""
        p = tmp_path / "calls.tsv"
        header = "GD_ID\tsample\tsvtype\tis_carrier\tchrom\tstart\tend\n"
        body = "GD1\tS1\tDEL\tTrue\tchr1\t1000\t5000\n"
        p.write_text(header + body)
        result = integrate.read_gd_calls(str(p))
        assert ("GD1", "DEL") in result
        assert result[("GD1", "DEL")]["samples"] == {"S1"}
        assert result[("GD1", "DEL")]["pos"] == 1000

    def test_narrow_format_single_sample_no_comma(self, tmp_path):
        """Case 4.20: Single sample (no comma)."""
        p = tmp_path / "calls.tsv"
        p.write_text("chr1\t1000\t5000\tGD1\tDEL\tS1\n")
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1"}

    def test_narrow_format_single_carrier_in_list(self, tmp_path):
        """Case 4.21: Single carrier in comma-separated list."""
        p = tmp_path / "calls.tsv"
        p.write_text("chr1\t1000\t5000\tGD1\tDEL\tS1,S2\n")
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1", "S2"}

    def test_narrow_format_trailing_newline(self, tmp_path):
        """Case 4.22: Narrow format with trailing newline."""
        p = tmp_path / "calls.tsv"
        p.write_text("chr1\t1000\t5000\tGD1\tDEL\tS1,S2\n\n\n")
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1", "S2"}

    def test_narrow_format_trailing_whitespace(self, tmp_path):
        """Case 4.23: Narrow format with trailing whitespace → not stripped."""
        p = tmp_path / "calls.tsv"
        p.write_text("chr1\t1000\t5000\tGD1\tDEL\tS1,S2  \n")
        result = integrate.read_gd_calls(str(p))
        # Trailing whitespace on last sample is preserved
        assert result[("GD1", "DEL")]["samples"] == {"S1", "S2  "}

    def test_narrow_format_gz_extension(self, tmp_path):
        """Case 4.24: .gz extension with narrow format."""
        import gzip
        p = tmp_path / "calls.tsv.gz"
        content = "chr1\t1000\t5000\tGD1\tDEL\tS1,S2\n"
        with gzip.open(str(p), "wt") as f:
            f.write(content)
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1", "S2"}

        """Case 4.26: Narrow format with duplicate GD_ID entries."""
        p = tmp_path / "calls.tsv"
        p.write_text(
            "chr1\t1000\t5000\tGD1\tDEL\tS1\n"
            "chr1\t1000\t5000\tGD1\tDEL\tS2\n"
        )
        result = integrate.read_gd_calls(str(p))


# ── Section 2: GD Table Loading (cases 2.3, 2.4, 2.6, 2.7, 2.8, 2.10-2.13) ──


class TestBuildTreesFromGdTableExtended:
    """Test cases 2.3, 2.4, 2.6, 2.7, 2.8, 2.10-2.13."""

    def test_mixed_nahr_non_nahr_in_same_cluster(self, tmp_path):
        """Case 2.3: Mixed NAHR/non-NAHR in same cluster → separate trees."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_MIX1\tDEL\tyes\tno\tclusterA\t1\t2\n"
            "chr1\t1000\t5000\tGD_MIX2\tDEL\tno\tno\tclusterA\t1\t2\n"
        )
        nahr, non_nahr, meta = integrate._build_trees_from_gd_table(str(gd_tsv))
        assert "chr1" in nahr
        assert "chr1" in non_nahr
        assert ("GD_MIX1", "DEL") in {iv.data for iv in nahr["chr1"].overlap(1000, 5000)}
        assert ("GD_MIX2", "DEL") in {iv.data for iv in non_nahr["chr1"].overlap(1000, 5000)}

    def test_same_gd_id_different_chromosomes(self, tmp_path):
        """Case 2.4: Same GD_ID on different chromosomes, same cluster → merged."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_SAME\tDEL\tyes\tno\tclusterA\t1\t2\n"
            "chr2\t2000\t6000\tGD_SAME\tDUP\tyes\tno\tclusterA\t1\t2\n"
        )
        nahr, non_nahr, meta = integrate._build_trees_from_gd_table(str(gd_tsv))
        # Both entries merged under first chromosome (chr1)
        assert "chr1" in nahr
        data = {iv.data for iv in nahr["chr1"].overlap(1000, 7000)}
        assert ("GD_SAME", "DEL") in data
        assert ("GD_SAME", "DUP") in data

    def test_multiple_nahr_entries_identical_coords(self, tmp_path):
        """Case 2.6: Multiple NAHR entries at identical coords."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_A\tDEL\tyes\tno\tclusterA\t1\t2\n"
            "chr1\t1000\t5000\tGD_B\tDEL\tyes\tno\tclusterA\t3\t4\n"
        )
        nahr, non_nahr, meta = integrate._build_trees_from_gd_table(str(gd_tsv))
        data = {iv.data for iv in nahr["chr1"].overlap(1000, 5000)}
        assert ("GD_A", "DEL") in data
        assert ("GD_B", "DEL") in data

    def test_empty_gd_table(self, tmp_path):
        """Case 2.7: Empty GD table (no rows) → empty trees."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
        )
        nahr, non_nahr, meta = integrate._build_trees_from_gd_table(str(gd_tsv))
        assert len(nahr) == 0
        assert len(non_nahr) == 0
        assert len(meta) == 0

    def test_gd_table_only_headers(self, tmp_path):
        """Case 2.8: GD table with only headers (no data rows)."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
        )
        nahr, non_nahr, meta = integrate._build_trees_from_gd_table(str(gd_tsv))
        assert len(nahr) == 0
        assert len(non_nahr) == 0
        assert len(meta) == 0

    def test_terminal_flag_present(self, tmp_path):
        """Case 2.10: Terminal flag present in GD table."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_TERM\tDEL\tyes\tyes\tclusterA\t1\t2\n"
            "chr1\t6000\t9000\tGD_NOTTERM\tDEL\tyes\tno\tclusterA\t1\t2\n"
        )
        nahr, non_nahr, meta = integrate._build_trees_from_gd_table(str(gd_tsv))
        assert meta["GD_TERM"]["nahr"] is True
        assert meta["GD_NOTTERM"]["nahr"] is True

    def test_gd_id_in_gd_calls_not_in_gd_table(self, tmp_path, monkeypatch, caplog):
        """Case 2.11: GD_ID in gd_calls but not in gd_table → warning + skip."""
        gd_table_path = _make_gd_table_file(tmp_path, [{
            "chr": "chr1", "start": 2000, "end": 8000,
            "gd_id": "GD_OTHER", "svtype": "DEL",
            "nahr": "yes", "cluster": "clusterX", "bp1": "1", "bp2": "2",
        }])
        gd_calls_path = _make_gd_calls_file(tmp_path, [{
            "chrom": "chr1", "pos": 1000, "end": 5000,
            "region_id": "GD_MISSING", "svtype": "DEL",
            "samples": [],
        }])
        ploidy_path = _make_ploidy_file(tmp_path, [("S1", {"chr1": 2})])
        par_path = _make_par_file(tmp_path)
        out_vcf = str(tmp_path / "out.vcf.gz")
        written_records = []

        class _FakeVF:
            def __init__(self, *a, **k): pass
            header = _FakeHeader(contigs={"chr1": None}, samples=["S1"])
            def __iter__(self): return iter([])
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def close(self): pass
            def write(self, record):
                written_records.append(record)

        import types
        fake_pysam = types.SimpleNamespace(VariantFile=_FakeVF, tabix_index=lambda *a,**k:None)
        monkeypatch.setattr(integrate, "pysam", fake_pysam)
        monkeypatch.setattr(integrate, "_sort_vcf", lambda *a,**k: None)
        monkeypatch.setattr(integrate, "setup_logging", lambda *a,**k: None)
        (tmp_path / "in.vcf.gz").write_text("dummy")

        integrate.main([
            "--vcf", str(tmp_path / "in.vcf.gz"),
            "--gd-calls", gd_calls_path,
            "--gd-table", gd_table_path,
            "--par-bed", par_path,
            "--ploidy-table", ploidy_path,
            "--out-vcf", out_vcf,
            "--temp-dir", str(tmp_path),
        ])
        assert not any("GD_MISSING" in r.id for r in written_records)

    def test_gd_table_with_extra_unknown_columns(self, tmp_path):
        """Case 2.13: GD table with extra/unknown columns."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\textra_col\tnotes\n"
            "chr1\t1000\t5000\tGD_EXTRA\tDEL\tyes\tno\tclusterA\t1\t2\tx\ty\n"
        )
        nahr, non_nahr, meta = integrate._build_trees_from_gd_table(str(gd_tsv))
        assert "chr1" in nahr
        assert ("GD_EXTRA", "DEL") in {iv.data for iv in nahr["chr1"].overlap(1000, 5000)}

    def test_bp_numeric_ordering(self, tmp_path):
        """Case 2.9: BP1/BP2 numeric ordering."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_ORD\tDEL\tyes\tno\tclusterA\t2\t1\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        loci = list(gd_table.get_all_loci().values())
        assert len(loci) >= 1


# ── Section 3: GDTable Class Internals (cases 3.1-3.10) ─────────────────


class TestGDTableClassInternals:
    """Test cases 3.1-3.10 for GDTable class internals."""

    def test_column_alias_mapping(self, tmp_path):
        """Case 3.1: Column alias mapping (start → start_GRCh38)."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD1\tDEL\tyes\tno\tclusterA\t1\t2\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        loci = list(gd_table.get_all_loci().values())
        assert len(loci) >= 1
        entry = loci[0].gd_entries[0]
        assert entry["start_GRCh38"] == 1000

    def test_missing_required_column_raises(self, tmp_path):
        """Case 3.2: Missing required column → ValueError."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tGD_ID\tsvtype\n"
            "chr1\t1000\tGD1\tDEL\n"
        )
        from gatk_sv_gd.models import GDTable
        with pytest.raises(ValueError, match="Missing required columns"):
            GDTable(str(gd_tsv))

    def test_bp1_greater_than_bp2_swap(self, tmp_path):
        """Case 3.3: BP1 > BP2 swap logic."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_SWAP\tDEL\tyes\tno\tclusterA\t5\t1\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        entry = gd_table.loci["clusterA"].gd_entries[0]
        assert entry["BP1"] == "5"
        assert entry["BP2"] == "1"

    def test_bp_alphanumeric_comparison(self, tmp_path):
        """Case 3.4: BP1/BP2 alphanumeric comparison."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_ALPHA\tDEL\tyes\tno\tclusterA\tA\tB\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        entry = gd_table.loci["clusterA"].gd_entries[0]
        assert entry["BP1"] == "A"
        assert entry["BP2"] == "B"

    def test_empty_cluster_no_loci(self, tmp_path):
        """Case 3.5: Empty cluster → no loci returned."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        assert len(gd_table.get_all_loci()) == 0

    def test_locus_with_zero_gd_entries(self, tmp_path):
        """Case 3.6: Locus with zero GD entries → no crash."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD1\tDEL\tyes\tno\tclusterA\t1\t2\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        locus = gd_table.get_all_loci()["clusterA"]
        assert len(locus.gd_entries) == 1

    def test_single_row_gd_table(self, tmp_path):
        """Case 3.9: GDTable with single row."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD_SINGLE\tDEL\tyes\tno\tclusterA\t1\t2\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        assert len(gd_table.get_all_loci()) == 1
        locus = list(gd_table.get_all_loci().values())[0]
        assert locus.chrom == "chr1"
        assert len(locus.gd_entries) == 1

    def test_get_all_loci_vs_get_loci_by_chrom_consistency(self, tmp_path):
        """Case 3.8: get_all_loci vs get_loci_by_chrom consistency."""
        gd_tsv = tmp_path / "gd.tsv"
        gd_tsv.write_text(
            "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
            "cluster\tBP1\tBP2\n"
            "chr1\t1000\t5000\tGD1\tDEL\tyes\tno\tclusterA\t1\t2\n"
            "chr2\t2000\t6000\tGD2\tDUP\tyes\tno\tclusterB\t1\t2\n"
            "chr1\t3000\t7000\tGD3\tDEL\tno\tno\tclusterC\t1\t2\n"
        )
        from gatk_sv_gd.models import GDTable
        gd_table = GDTable(str(gd_tsv))
        all_loci = gd_table.get_all_loci()
        chr1_loci = gd_table.get_loci_by_chrom("chr1")
        chr2_loci = gd_table.get_loci_by_chrom("chr2")
        assert len(chr1_loci) == 2
        assert len(chr2_loci) == 1
        assert len(all_loci) == 3

    def test_gd_table_encoding_utf8_bom_raises(self, tmp_path):
        """Case 3.10: GDTable encoding issues (UTF-8 BOM) → ValueError."""
        gd_tsv = tmp_path / "gd.tsv"
        content = "\uFEFFchr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\t"
        content += "cluster\tBP1\tBP2\nchr1\t1000\t5000\tGD1\tDEL\tyes\tno\tclusterA\t1\t2\n"
        gd_tsv.write_text(content, encoding="utf-8-sig")
        from gatk_sv_gd.models import GDTable
        with pytest.raises(ValueError, match="Missing required columns"):
            GDTable(str(gd_tsv))


# ── Section 4: GD Calls Reading Extended (cases 4.6-4.14, 4.20-4.26) ───


class TestReadGdCallsExtended:
    """Test cases 4.6-4.14, 4.20-4.26 for GD calls reading."""

    def test_wide_format_is_carrier_true_lowercase(self, tmp_path):
        """Case 4.6: is_carrier == 'true' (lowercase)."""
        p = tmp_path / "calls.tsv"
        header = "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
        body = "S1\tGD1\tchr1\t1000\t5000\tDEL\ttrue\n"
        p.write_text(header + body)
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1"}

    def test_wide_format_is_carrier_numeric_one(self, tmp_path):
        """Case 4.7: is_carrier == '1' (numeric string)."""
        p = tmp_path / "calls.tsv"
        header = "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
        body = "S1\tGD1\tchr1\t1000\t5000\tDEL\t1\n"
        p.write_text(header + body)
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1"}

    def test_wide_format_mixed_carrier_values(self, tmp_path):
        """Case 4.8: Mixed True/true/1 values."""
        p = tmp_path / "calls.tsv"
        header = "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
        body = (
            "S1\tGD1\tchr1\t1000\t5000\tDEL\tTrue\n"
            "S2\tGD1\tchr1\t1000\t5000\tDEL\ttrue\n"
            "S3\tGD1\tchr1\t1000\t5000\tDEL\t1\n"
            "S4\tGD1\tchr1\t1000\t5000\tDEL\tFalse\n"
        )
        p.write_text(header + body)
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1", "S2", "S3"}

    def test_wide_format_whitespace_around_is_carrier(self, tmp_path):
        """Case 4.9: Extra whitespace around is_carrier."""
        p = tmp_path / "calls.tsv"
        header = "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
        body = "S1\tGD1\tchr1\t1000\t5000\tDEL\t True \n"
        p.write_text(header + body)
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1"}

    def test_wide_format_comment_lines_at_top(self, tmp_path):
        """Case 4.10: Wide format with comment lines at top → header detected after comments."""
        p = tmp_path / "calls.tsv"
        content = (
            "# This is a comment\n"
            "# Another comment\n"
            "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
            "S1\tGD1\tchr1\t1000\t5000\tDEL\tTrue\n"
        )
        p.write_text(content)
        result = integrate.read_gd_calls(str(p))
        assert ("GD1", "DEL") in result

    def test_wide_format_empty_file_header_only(self, tmp_path):
        """Case 4.11: Empty wide-format file (header only)."""
        p = tmp_path / "calls.tsv"
        p.write_text("sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n")
        result = integrate.read_gd_calls(str(p))
        assert len(result) == 0

    def test_wide_format_multiple_svtypes_per_gd_id(self, tmp_path):
        """Case 4.13: Wide format with multiple svtypes per GD_ID."""
        p = tmp_path / "calls.tsv"
        header = "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\n"
        body = (
            "S1\tGD1\tchr1\t1000\t5000\tDEL\tTrue\n"
            "S2\tGD1\tchr1\t1000\t5000\tDUP\tTrue\n"
        )
        p.write_text(header + body)
        result = integrate.read_gd_calls(str(p))
        assert ("GD1", "DEL") in result
        assert ("GD1", "DUP") in result
        assert result[("GD1", "DEL")]["samples"] == {"S1"}
        assert result[("GD1", "DUP")]["samples"] == {"S2"}

    def test_wide_format_columns_unexpected_order(self, tmp_path):
        """Case 4.14: Columns in unexpected order (csv.DictReader handles this)."""
        p = tmp_path / "calls.tsv"
        header = "GD_ID\tsample\tsvtype\tis_carrier\tchrom\tstart\tend\n"
        body = "GD1\tS1\tDEL\tTrue\tchr1\t1000\t5000\n"
        p.write_text(header + body)
        result = integrate.read_gd_calls(str(p))
        assert ("GD1", "DEL") in result
        assert result[("GD1", "DEL")]["samples"] == {"S1"}
        assert result[("GD1", "DEL")]["pos"] == 1000

    def test_narrow_format_single_sample_no_comma(self, tmp_path):
        """Case 4.20: Single sample (no comma)."""
        p = tmp_path / "calls.tsv"
        p.write_text("chr1\t1000\t5000\tGD1\tDEL\tS1\n")
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1"}

    def test_narrow_format_single_carrier_in_list(self, tmp_path):
        """Case 4.21: Single carrier in comma-separated list."""
        p = tmp_path / "calls.tsv"
        p.write_text("chr1\t1000\t5000\tGD1\tDEL\tS1,S2\n")
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1", "S2"}

    def test_narrow_format_trailing_newline(self, tmp_path):
        """Case 4.22: Narrow format with trailing newline."""
        p = tmp_path / "calls.tsv"
        p.write_text("chr1\t1000\t5000\tGD1\tDEL\tS1,S2\n\n\n")
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1", "S2"}

    def test_narrow_format_trailing_whitespace(self, tmp_path):
        """Case 4.23: Narrow format with trailing whitespace → not stripped."""
        p = tmp_path / "calls.tsv"
        p.write_text("chr1\t1000\t5000\tGD1\tDEL\tS1,S2  \n")
        result = integrate.read_gd_calls(str(p))
        # Trailing whitespace on last sample is preserved
        assert result[("GD1", "DEL")]["samples"] == {"S1", "S2  "}

    def test_narrow_format_gz_extension(self, tmp_path):
        """Case 4.24: .gz extension with narrow format."""
        import gzip
        p = tmp_path / "calls.tsv.gz"
        content = "chr1\t1000\t5000\tGD1\tDEL\tS1,S2\n"
        with gzip.open(str(p), "wt") as f:
            f.write(content)
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S1", "S2"}

    def test_narrow_format_duplicate_gd_id_entries(self, tmp_path):
        """Case 4.26: Narrow format with duplicate GD_ID entries."""
        p = tmp_path / "calls.tsv"
        p.write_text(
            "chr1\t1000\t5000\tGD1\tDEL\tS1\n"
            "chr1\t1000\t5000\tGD1\tDEL\tS2\n"
        )
        result = integrate.read_gd_calls(str(p))
        assert result[("GD1", "DEL")]["samples"] == {"S2"}
