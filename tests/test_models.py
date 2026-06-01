import pytest
import pandas as pd

from gatk_sv_gd.models import GDLocus, GDTable, validate_gd_table_for_preprocess


def test_get_intervals_between_rejects_unknown_breakpoint_names():
    locus = GDLocus(
        cluster="test_cluster",
        chrom="chr1",
        breakpoints=[(100, 100), (200, 200), (300, 300)],
        breakpoint_names=["A", "B", "C"],
        gd_entries=[],
        is_nahr=True,
        is_terminal=False,
    )

    with pytest.raises(ValueError, match="Unknown breakpoint"):
        locus.get_intervals_between("A", "Z")


def test_gd_locus_helpers_cover_properties_flanks_and_svtypes():
    locus = GDLocus(
        cluster="test_cluster",
        chrom="chr1",
        breakpoints=[(100, 110), (200, 205), (300, 300)],
        breakpoint_names=["A", "B", "C"],
        gd_entries=[
            {"GD_ID": "GD1", "start_GRCh38": 100, "end_GRCh38": 205, "svtype": "DEL"},
            {"GD_ID": "GD2", "start_GRCh38": 200, "end_GRCh38": 300, "svtype": "DUP"},
        ],
        is_nahr=True,
        is_terminal=False,
    )

    assert locus.n_breakpoints == 3
    assert locus.start == 100
    assert locus.end == 300
    assert locus.get_intervals() == [(110, 200, "A-B"), (205, 300, "B-C")]
    assert locus.get_intervals_between("C", "A") == [(110, 200, "A-B"), (205, 300, "B-C")]
    assert locus.get_gd_intervals() == {"GD1": (100, 205), "GD2": (200, 300)}
    assert locus.get_flanking_regions() == [(0, 100, "left_flank"), (300, 500, "right_flank")]
    assert locus.del_entries == [{"GD_ID": "GD1", "start_GRCh38": 100, "end_GRCh38": 205, "svtype": "DEL"}]
    assert locus.dup_entries == [{"GD_ID": "GD2", "start_GRCh38": 200, "end_GRCh38": 300, "svtype": "DUP"}]
    assert set(locus.svtypes) == {"DEL", "DUP"}


def test_gd_table_parses_standalone_loci_and_breakpoint_names(monkeypatch):
    monkeypatch.setattr(
        "gatk_sv_gd.models.pd.read_csv",
        lambda filepath, sep="\t": pd.DataFrame(
            [
                {
                    "chr": "chr10",
                    "start_GRCh38": 100,
                    "end_GRCh38": 200,
                    "GD_ID": "GD1",
                    "svtype": "DEL",
                    "NAHR": "yes",
                    "terminal": "no",
                    "cluster": "",
                    "BP1": float("nan"),
                    "BP2": float("nan"),
                },
                {
                    "chr": "chr15",
                    "start_GRCh38": 400,
                    "end_GRCh38": 500,
                    "GD_ID": "GD2",
                    "svtype": "DEL",
                    "NAHR": "no",
                    "terminal": "yes",
                    "cluster": "clusterA",
                    "BP1": "CHRNA7",
                    "BP2": 5.0,
                },
                {
                    "chr": "chr15",
                    "start_GRCh38": 500,
                    "end_GRCh38": 600,
                    "GD_ID": "GD3",
                    "svtype": "DUP",
                    "NAHR": "no",
                    "terminal": "yes",
                    "cluster": "clusterA",
                    "BP1": 5.0,
                    "BP2": 6.0,
                },
            ]
        ),
    )

    table = GDTable("gd.tsv")

    standalone = table.get_locus("chr10:100-200")
    assert standalone is not None
    assert standalone.breakpoint_names == ["1", "2"]
    assert standalone.breakpoints == [(100, 100), (200, 200)]

    clustered = table.get_locus("clusterA")
    assert clustered is not None
    assert clustered.breakpoint_names == ["CHRNA7", "5", "6"]
    assert clustered.breakpoints == [(400, 400), (500, 500), (600, 600)]
    assert clustered.is_terminal is True
    assert table.get_all_loci().keys() == {"chr10:100-200", "clusterA"}
    assert table.get_loci_by_chrom("chr15").keys() == {"clusterA"}


def test_gd_table_validation_and_preprocess_reject_chr_y(monkeypatch):
    monkeypatch.setattr(
        "gatk_sv_gd.models.pd.read_csv",
        lambda filepath, sep="\t": pd.DataFrame(
            [{"chr": "chr1", "start_GRCh38": 1, "end_GRCh38": 2, "GD_ID": "GD1"}]
        ),
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        GDTable("gd.tsv")

    y_locus = GDLocus(
        cluster="Y_cluster",
        chrom="chrY",
        breakpoints=[(10, 10), (20, 20)],
        breakpoint_names=["A", "B"],
        gd_entries=[],
        is_nahr=False,
        is_terminal=False,
    )

    with pytest.raises(ValueError, match="chrY"):
        validate_gd_table_for_preprocess(type("FakeGDTable", (), {"loci": {"Y_cluster": y_locus}})())