import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gatk_sv_gd import extract
from gatk_sv_gd.models import GDLocus


class _HeaderStub:
    def __init__(self, info_ids=None, contigs=None, samples=None):
        self.info = {key: object() for key in (info_ids or [])}
        self.contigs = contigs or {"chr1": None}
        self.samples = samples or ["S1", "S2"]
        self.lines = []

    def add_line(self, line):
        self.lines.append(line)
        info_id = line.split("ID=")[1].split(",")[0]
        self.info[info_id] = object()

    def copy(self):
        copied = _HeaderStub(contigs=dict(self.contigs), samples=list(self.samples))
        copied.info = dict(self.info)
        copied.lines = list(self.lines)
        return copied

    def __str__(self):
        base = ["##fileformat=VCFv4.2"]
        base.extend(self.lines)
        base.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
        return "\n".join(base) + "\n"


class _RecordStub:
    def __init__(self, chrom, pos, stop, record_id, svtype=None, alts=None, line=None):
        self.chrom = chrom
        self.pos = pos
        self.start = pos
        self.stop = stop
        self.id = record_id
        self.info = {} if svtype is None else {"SVTYPE": svtype}
        self.alts = alts
        self._line = line or (
            f"{chrom}\t{pos}\t{record_id}\tN\t<{svtype or 'DEL'}>\t.\tPASS\tSVTYPE={svtype or 'DEL'}"
        )

    def __str__(self):
        return self._line


class _VariantFileStub:
    def __init__(self, path, records_by_region, header):
        self.path = path
        self._records_by_region = records_by_region
        self.header = header
        self.closed = False

    def fetch(self, chrom, start, end):
        key = (chrom, start, end)
        return iter(self._records_by_region.get(key, []))

    def close(self):
        self.closed = True


def _make_locus(cluster, chrom, start, mid, end, entries):
    return GDLocus(
        cluster=cluster,
        chrom=chrom,
        breakpoints=[(start, start), (mid, mid), (end, end)],
        breakpoint_names=["A", "B", "C"],
        gd_entries=entries,
        is_nahr=True,
        is_terminal=False,
    )


def test_overlap_helpers_and_classification_cover_canonical_and_atypical_paths():
    assert extract._overlap_bases(10, 20, 15, 30) == 5
    assert extract._overlap_bases(10, 20, 20, 30) == 0
    assert extract._reciprocal_overlap(10, 20, 15, 30) == pytest.approx(5 / 15)
    assert extract._reciprocal_overlap(10, 10, 15, 30) == pytest.approx(0.0)
    assert extract._fraction_covered(100, 200, 125, 175) == pytest.approx(0.5)
    assert extract._fraction_covered(100, 100, 125, 175) == pytest.approx(0.0)

    gd_entries = [
        {
            "GD_ID": "GD_NAHR",
            "svtype": "DEL",
            "start_GRCh38": 100,
            "end_GRCh38": 200,
            "NAHR": "yes",
        },
        {
            "GD_ID": "GD_NON_NAHR",
            "svtype": "DEL",
            "start_GRCh38": 100,
            "end_GRCh38": 250,
            "NAHR": "no",
        },
    ]

    gd_ids, is_nahr, is_atypical, is_non_nahr = extract._classify_variant(
        100,
        190,
        "DEL",
        gd_entries,
        ro_threshold=0.5,
        atypical_coverage=0.7,
        non_nahr_overlap=0.01,
    )
    assert gd_ids == {"GD_NAHR", "GD_NON_NAHR"}
    assert is_nahr is True
    assert is_atypical is False
    assert is_non_nahr is True

    gd_ids, is_nahr, is_atypical, is_non_nahr = extract._classify_variant(
        100,
        150,
        "DEL",
        gd_entries,
        ro_threshold=0.5,
        atypical_coverage=0.7,
        non_nahr_overlap=0.01,
    )
    assert gd_ids == {"GD_NAHR", "GD_NON_NAHR"}
    assert is_nahr is False
    assert is_atypical is True
    assert is_non_nahr is True


def test_classify_variant_returns_no_flags_when_entries_do_not_qualify():
    gd_entries = [
        {
            "GD_ID": "WRONG_TYPE",
            "svtype": "DUP",
            "start_GRCh38": 100,
            "end_GRCh38": 200,
            "NAHR": "yes",
        },
        {
            "GD_ID": "LOW_NON_NAHR",
            "svtype": "DEL",
            "start_GRCh38": 100,
            "end_GRCh38": 200,
            "NAHR": "no",
        },
        {
            "GD_ID": "LOW_NAHR",
            "svtype": "DEL",
            "start_GRCh38": 100,
            "end_GRCh38": 200,
            "NAHR": "yes",
        },
    ]

    gd_ids, is_nahr, is_atypical, is_non_nahr = extract._classify_variant(
        0,
        20,
        "DEL",
        gd_entries,
        ro_threshold=0.5,
        atypical_coverage=0.7,
        non_nahr_overlap=0.5,
    )

    assert gd_ids == set()
    assert is_nahr is False
    assert is_atypical is False
    assert is_non_nahr is False


def test_svtype_and_info_header_helpers():
    record = _RecordStub("chr1", 100, 200, "var1", svtype="DUP")
    assert extract._svtype_of(record) == "DUP"

    tuple_type = _RecordStub("chr1", 100, 200, "var_tuple", svtype=None, alts=None)
    tuple_type.info["SVTYPE"] = ("DEL",)
    assert extract._svtype_of(tuple_type) == "DEL"

    alt_only = _RecordStub("chr1", 100, 200, "var2", svtype=None, alts=("<DEL>",))
    assert extract._svtype_of(alt_only) == "DEL"

    no_type = _RecordStub("chr1", 100, 200, "var3", svtype=None, alts=("N",))
    assert extract._svtype_of(no_type) is None

    header = _HeaderStub(info_ids=["GD"])
    extract._ensure_info_headers(header)
    assert set(header.info) >= {"GD", "NAHR_GD", "NON_NAHR_GD", "NAHR_GD_atypical"}


def test_build_and_merge_query_regions_merges_overlapping_loci():
    locus_a = _make_locus(
        "clusterA",
        "chr1",
        100,
        150,
        200,
        [],
    )
    locus_b = _make_locus(
        "clusterB",
        "chr1",
        180,
        220,
        260,
        [],
    )
    locus_c = _make_locus(
        "clusterC",
        "chr2",
        300,
        350,
        400,
        [],
    )
    gd_table = SimpleNamespace(get_all_loci=lambda: {
        "clusterA": locus_a,
        "clusterB": locus_b,
        "clusterC": locus_c,
    })

    merged = extract._build_and_merge_query_regions(gd_table)

    assert merged["chr1"] == [(100, 260, ["clusterA", "clusterB"])]
    assert merged["chr2"] == [(300, 400, ["clusterC"])]


def test_process_vcfs_merges_duplicate_records_and_sorts_by_header_order(monkeypatch):
    locus_chr1 = _make_locus(
        "cluster1",
        "chr1",
        100,
        150,
        200,
        [
            {
                "GD_ID": "GD1",
                "svtype": "DEL",
                "start_GRCh38": 100,
                "end_GRCh38": 200,
                "NAHR": "yes",
            }
        ],
    )
    locus_chr2 = _make_locus(
        "cluster2",
        "chr2",
        50,
        80,
        120,
        [
            {
                "GD_ID": "GD2",
                "svtype": "DUP",
                "start_GRCh38": 50,
                "end_GRCh38": 120,
                "NAHR": "yes",
            }
        ],
    )
    gd_table = SimpleNamespace(get_all_loci=lambda: {
        "cluster1": locus_chr1,
        "cluster2": locus_chr2,
    })
    header = _HeaderStub(contigs={"chr2": None, "chr1": None}, samples=["S1", "S2"])
    rec_chr1 = _RecordStub(
        "chr1",
        100,
        190,
        "var1",
        svtype="DEL",
        line="chr1\t100\tvar1\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL\tGT\t0/1\t0/0",
    )
    rec_chr2 = _RecordStub(
        "chr2",
        50,
        120,
        "var2",
        svtype="DUP",
        line="chr2\t50\tvar2\tN\t<DUP>\t.\tPASS\tSVTYPE=DUP\tGT\t1/1\t0/0",
    )
    file_one = _VariantFileStub(
        "a.vcf.gz",
        {("chr1", 100, 200): [rec_chr1], ("chr2", 50, 120): [rec_chr2]},
        header,
    )
    file_two = _VariantFileStub(
        "b.vcf.gz",
        {("chr1", 100, 200): [rec_chr1]},
        header,
    )
    files = {"a.vcf.gz": file_one, "b.vcf.gz": file_two}
    monkeypatch.setattr(extract.pysam, "VariantFile", lambda path: files[path])
    monkeypatch.setattr(
        extract,
        "_build_and_merge_query_regions",
        lambda gd_table: {
            "chr1": [(100, 200, ["cluster1"])],
            "chr2": [(50, 120, ["cluster2"])],
        },
    )

    annotated, out_header, sample_names = extract._process_vcfs(
        ["a.vcf.gz", "b.vcf.gz"],
        gd_table,
        ro_threshold=0.5,
        atypical_coverage=0.7,
        non_nahr_overlap=0.01,
    )

    assert [record[0] for record in annotated] == ["chr2", "chr1"]
    assert len(annotated) == 2
    assert annotated[0][6]["gd_ids"] == {"GD2"}
    assert annotated[1][6]["gd_ids"] == {"GD1"}
    assert out_header is not None
    assert sample_names == ["S1", "S2"]
    assert file_one.closed is True
    assert file_two.closed is True


def test_process_vcfs_skips_missing_contigs_fetch_errors_and_non_matches(monkeypatch):
    gd_table = SimpleNamespace(get_all_loci=lambda: {"cluster1": SimpleNamespace(gd_entries=[{"GD_ID": "GD1"}])})
    header = _HeaderStub(contigs={"chr1": None}, samples=["S1"])

    unsupported = _RecordStub("chr1", 60, 80, "skip_type", svtype=None, alts=("N",))
    no_match = _RecordStub("chr1", 70, 90, "skip_gd", svtype="DEL")
    kept = _RecordStub("chr1", 80, 95, "keep", svtype="DUP")

    class _VariantFileWithErrors:
        def __init__(self):
            self.header = header
            self.closed = False

        def fetch(self, chrom, start, end):
            if (chrom, start, end) == ("chr1", 0, 50):
                raise ValueError("bad region")
            if (chrom, start, end) == ("chr1", 50, 100):
                return iter([unsupported, no_match, kept])
            return iter(())

        def close(self):
            self.closed = True

    vcf = _VariantFileWithErrors()

    def fake_classify(var_start, var_end, var_svtype, gd_entries, ro_threshold, atypical_coverage, non_nahr_overlap):
        if var_start == 70:
            return set(), False, False, False
        return {"GD1"}, False, True, False

    monkeypatch.setattr(extract.pysam, "VariantFile", lambda path: vcf)
    monkeypatch.setattr(
        extract,
        "_build_and_merge_query_regions",
        lambda gd_table: {
            "chr_missing": [(0, 10, ["cluster1"])],
            "chr1": [(0, 50, ["cluster1"]), (50, 100, ["cluster1"])],
        },
    )
    monkeypatch.setattr(extract, "_classify_variant", fake_classify)

    annotated, out_header, sample_names = extract._process_vcfs(
        ["only.vcf.gz"],
        gd_table,
        ro_threshold=0.5,
        atypical_coverage=0.7,
        non_nahr_overlap=0.01,
    )

    assert len(annotated) == 1
    assert annotated[0][3] == "keep"
    assert annotated[0][6] == {"gd_ids": {"GD1"}, "nahr": False, "atypical": True, "non_nahr": False}
    assert out_header is not None
    assert sample_names == ["S1"]
    assert vcf.closed is True


def test_process_vcfs_returns_empty_results_when_no_vcfs_are_provided():
    gd_table = SimpleNamespace(get_all_loci=lambda: {})

    annotated, out_header, sample_names = extract._process_vcfs(
        [],
        gd_table,
        ro_threshold=0.5,
        atypical_coverage=0.7,
        non_nahr_overlap=0.01,
    )

    assert annotated == []
    assert out_header is None
    assert sample_names == []


def test_modify_info_write_vcf_carrier_parsing_and_write_bed(monkeypatch, tmp_path):
    modified = extract._modify_info_field(
        "SVTYPE=DEL;GD=OLD;NON_NAHR_GD",
        {"gd_ids": {"GD1", "GD2"}, "nahr": True, "atypical": False, "non_nahr": False},
    )
    assert modified == "SVTYPE=DEL;GD=GD1,GD2;NAHR_GD"

    assert extract._carriers_from_fields(
        ["chr1", "100", "var1", "N", "<DEL>", ".", "PASS", "SVTYPE=DEL", "GT:DP", "0/1:10", "0/0:20"],
        ["S1", "S2"],
    ) == ["S1"]
    assert extract._carriers_from_fields(
        ["chr1", "100", "var1", "N", "<DEL>", ".", "PASS", "SVTYPE=DEL", "DP:GT", "10:0/0", "20:1/1"],
        ["S1", "S2"],
    ) == ["S2"]

    compressed = {}
    indexed = {}

    def _fake_compress(src, dst, force):
        compressed["src"] = src
        compressed["dst"] = dst
        Path(dst).write_text(Path(src).read_text())

    def _fake_index(path, preset, force):
        indexed["path"] = path
        indexed["preset"] = preset

    monkeypatch.setattr(extract.pysam, "tabix_compress", _fake_compress)
    monkeypatch.setattr(extract.pysam, "tabix_index", _fake_index)

    annotated = [
        (
            "chr1",
            100,
            190,
            "var1",
            "DEL",
            "chr1\t100\tvar1\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL\tGT\t0/1\t0/0",
            {"gd_ids": {"GD1"}, "nahr": True, "atypical": False, "non_nahr": False},
        )
    ]
    header = _HeaderStub(samples=["S1", "S2"])
    vcf_path = tmp_path / "gd_variants.vcf.gz"
    bed_path = tmp_path / "gd_variants.bed"

    extract._write_vcf(annotated, header, str(vcf_path))
    assert compressed["dst"] == str(vcf_path)
    assert indexed == {"path": str(vcf_path), "preset": "vcf"}
    assert "GD=GD1;NAHR_GD" in vcf_path.read_text()

    extract._write_bed(annotated, ["S1", "S2"], str(bed_path))
    bed_text = bed_path.read_text()
    assert "#chrom\tstart\tend\tname\tsvtype" in bed_text
    assert "chr1\t100\t190\tvar1\tDEL\tGD1\tS1\t1\tTrue\tFalse\tFalse" in bed_text


def test_carriers_from_fields_returns_empty_without_genotypes_and_write_vcf_plain_text(monkeypatch, tmp_path):
    assert extract._carriers_from_fields(["chr1", "100", "var1"], ["S1"]) == []
    assert extract._carriers_from_fields(
        ["chr1", "100", "var1", "N", "<DEL>", ".", "PASS", "SVTYPE=DEL", "DP", "10"],
        ["S1"],
    ) == []
    assert extract._carriers_from_fields(
        ["chr1", "100", "var1", "N", "<DEL>", ".", "PASS", "SVTYPE=DEL", "DP:GT", "10"],
        ["S1"],
    ) == []

    compress_calls = []
    index_calls = []
    monkeypatch.setattr(extract.pysam, "tabix_compress", lambda *args, **kwargs: compress_calls.append(args))
    monkeypatch.setattr(extract.pysam, "tabix_index", lambda *args, **kwargs: index_calls.append(args))

    plain_vcf_path = tmp_path / "gd_variants.vcf"
    extract._write_vcf(
        [
            (
                "chr1",
                100,
                190,
                "var1",
                "DEL",
                "chr1\t100\tvar1\tN\t<DEL>\t.\tPASS\t.",
                {"gd_ids": {"GD1"}, "nahr": False, "atypical": True, "non_nahr": True},
            )
        ],
        _HeaderStub(samples=["S1"]),
        str(plain_vcf_path),
    )

    assert compress_calls == []
    assert index_calls == []
    plain_text = plain_vcf_path.read_text()
    assert "GD=GD1;NAHR_GD_atypical;NON_NAHR_GD" in plain_text


def test_main_validates_inputs_and_writes_outputs(monkeypatch, tmp_path, capsys):
    output_dir = tmp_path / "out"
    gd_table_path = tmp_path / "gd.tsv"
    vcf_path = tmp_path / "sample.vcf.gz"
    gd_table_path.write_text("dummy\n")
    vcf_path.write_text("dummy\n")

    gd_table = SimpleNamespace(
        df=[1, 2],
        get_all_loci=lambda: {"cluster1": object()},
    )
    calls = {}

    monkeypatch.setattr(sys, "argv", [
        "gatk-sv-gd extract",
        "--vcf",
        str(vcf_path),
        "--gd-table",
        str(gd_table_path),
        "--output-dir",
        str(output_dir),
    ])
    monkeypatch.setattr(extract, "GDTable", lambda path: gd_table)
    monkeypatch.setattr(extract, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        extract,
        "_process_vcfs",
        lambda *args, **kwargs: (
            [
                (
                    "chr1",
                    100,
                    190,
                    "var1",
                    "DEL",
                    "chr1\t100\tvar1\tN\t<DEL>\t.\tPASS\tSVTYPE=DEL\tGT\t0/1",
                    {"gd_ids": {"GD1"}, "nahr": True, "atypical": False, "non_nahr": False},
                )
            ],
            _HeaderStub(samples=["S1"]),
            ["S1"],
        ),
    )
    monkeypatch.setattr(extract, "_write_vcf", lambda annotated, header, path: calls.setdefault("vcf", path))
    monkeypatch.setattr(extract, "_write_bed", lambda annotated, sample_names, path: calls.setdefault("bed", path))

    extract.main()

    captured = capsys.readouterr()
    assert "Loading GD table" in captured.out
    assert "1 overlapping records found" in captured.out
    assert calls["vcf"].endswith("gd_variants.vcf.gz")
    assert calls["bed"].endswith("gd_variants.bed")


def test_parse_args_accepts_vcf_list_and_threshold_overrides(monkeypatch, tmp_path):
    vcf_list = tmp_path / "vcfs.txt"
    vcf_list.write_text("a.vcf.gz\nb.vcf.gz\n")

    monkeypatch.setattr(sys, "argv", [
        "gatk-sv-gd extract",
        "--vcf-list",
        str(vcf_list),
        "--gd-table",
        "gd.tsv",
        "--output-dir",
        "out",
        "--reciprocal-overlap",
        "0.4",
        "--atypical-coverage",
        "0.8",
        "--non-nahr-overlap",
        "0.02",
    ])

    args = extract._parse_args()

    assert args.vcf_list == str(vcf_list)
    assert args.vcfs is None
    assert args.gd_table == "gd.tsv"
    assert args.output_dir == "out"
    assert args.reciprocal_overlap == pytest.approx(0.4)
    assert args.atypical_coverage == pytest.approx(0.8)
    assert args.non_nahr_overlap == pytest.approx(0.02)


def test_main_reads_vcf_list_and_exits_cleanly_when_no_overlaps(monkeypatch, tmp_path, capsys):
    output_dir = tmp_path / "out"
    gd_table_path = tmp_path / "gd.tsv"
    vcf_a = tmp_path / "a.vcf.gz"
    vcf_b = tmp_path / "b.vcf.gz"
    vcf_list = tmp_path / "vcfs.txt"
    gd_table_path.write_text("dummy\n")
    vcf_a.write_text("dummy\n")
    vcf_b.write_text("dummy\n")
    vcf_list.write_text(f"{vcf_a}\n{vcf_b}\n")

    gd_table = SimpleNamespace(
        df=[1],
        get_all_loci=lambda: {"cluster1": object()},
    )

    monkeypatch.setattr(sys, "argv", [
        "gatk-sv-gd extract",
        "--vcf-list",
        str(vcf_list),
        "--gd-table",
        str(gd_table_path),
        "--output-dir",
        str(output_dir),
    ])
    monkeypatch.setattr(extract, "GDTable", lambda path: gd_table)
    monkeypatch.setattr(extract, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(extract, "_process_vcfs", lambda *args, **kwargs: ([], _HeaderStub(samples=["S1"]), ["S1"]))

    with pytest.raises(SystemExit, match="0"):
        extract.main()

    captured = capsys.readouterr()
    assert "Processing 2 VCF(s)" in captured.out
    assert "No overlapping DEL/DUP records found." in captured.out


def test_main_exits_when_required_input_is_missing(monkeypatch, tmp_path):
    missing_vcf = tmp_path / "missing.vcf.gz"
    gd_table_path = tmp_path / "gd.tsv"
    gd_table_path.write_text("dummy\n")

    monkeypatch.setattr(sys, "argv", [
        "gatk-sv-gd extract",
        "--vcf",
        str(missing_vcf),
        "--gd-table",
        str(gd_table_path),
        "--output-dir",
        str(tmp_path / "out"),
    ])
    monkeypatch.setattr(extract, "setup_logging", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit, match="1"):
        extract.main()


def test_main_exits_when_gd_table_is_missing(monkeypatch, tmp_path):
    vcf_path = tmp_path / "sample.vcf.gz"
    missing_gd_table = tmp_path / "missing.tsv"
    vcf_path.write_text("dummy\n")

    monkeypatch.setattr(sys, "argv", [
        "gatk-sv-gd extract",
        "--vcf",
        str(vcf_path),
        "--gd-table",
        str(missing_gd_table),
        "--output-dir",
        str(tmp_path / "out"),
    ])
    monkeypatch.setattr(extract, "setup_logging", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit, match="1"):
        extract.main()