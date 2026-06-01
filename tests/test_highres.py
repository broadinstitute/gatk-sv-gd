import numpy as np
import pandas as pd
import pytest

from gatk_sv_gd import highres


class _StubTabixFile:
    def __init__(self, header, lines):
        self.header = header
        self._lines = lines
        self.closed = False

    def fetch(self, chrom, start, end):
        del chrom, start, end
        return iter(self._lines)

    def close(self):
        self.closed = True


def test_query_highres_bins_rejects_no_common_samples(monkeypatch):
    stub = _StubTabixFile(
        ["#Chr\tStart\tEnd\tOTHER"],
        ["chr1\t100\t150\t10"],
    )
    monkeypatch.setattr(highres.pysam, "TabixFile", lambda path: stub)

    with pytest.raises(ValueError, match="shares no sample columns"):
        highres.query_highres_bins(
            "highres.tsv.gz",
            "chr1",
            100,
            200,
            ["S1"],
        )

    assert stub.closed is True


def test_query_highres_bins_returns_empty_indexed_frame(monkeypatch):
    stub = _StubTabixFile(
        ["#Chr\tStart\tEnd\tS1"],
        [],
    )
    monkeypatch.setattr(highres.pysam, "TabixFile", lambda path: stub)

    result = highres.query_highres_bins(
        "highres.tsv.gz",
        "chr1",
        100,
        200,
        ["S1", "S2"],
    )

    assert list(result.columns) == ["Chr", "Start", "End", "source_file", "S1", "S2"]
    assert result.empty
    assert result.index.name == "Bin"
    assert stub.closed is True


def test_query_highres_bins_coarsens_records_to_max_bins(monkeypatch):
    stub = _StubTabixFile(
        ["#Chr\tStart\tEnd\tS1\tS2"],
        [
            "chr1\t100\t130\t10\t20",
            "chr1\t130\t160\t11\t21",
            "chr1\t160\t190\t12\t22",
            "chr1\t190\t220\t13\t23",
        ],
    )
    monkeypatch.setattr(highres.pysam, "TabixFile", lambda path: stub)

    result = highres.query_highres_bins(
        "highres.tsv.gz",
        "chr1",
        100,
        220,
        ["S1", "S2", "S3"],
        max_bins=2,
    )

    assert result.index.tolist() == ["chr1:100-160", "chr1:160-220"]
    assert result["Start"].tolist() == [100, 160]
    assert result["End"].tolist() == [160, 220]
    assert result["source_file"].tolist() == ["highres", "highres"]
    assert result["S1"].tolist() == pytest.approx([21.0, 25.0])
    assert result["S2"].tolist() == pytest.approx([41.0, 45.0])
    assert result["S3"].isna().all()
    assert stub.closed is True


def test_query_highres_bins_returns_empty_frame_when_coarsened_query_has_no_rows(monkeypatch):
    stub = _StubTabixFile(
        ["#Chr\tStart\tEnd\tS1"],
        [],
    )
    monkeypatch.setattr(highres.pysam, "TabixFile", lambda path: stub)

    result = highres.query_highres_bins(
        "highres.tsv.gz",
        "chr1",
        100,
        200,
        ["S1", "S2"],
        max_bins=4,
    )

    assert result.empty
    assert list(result.columns) == ["Chr", "Start", "End", "source_file", "S1", "S2"]
    assert result.index.name == "Bin"
    assert stub.closed is True


def test_query_highres_bins_returns_numeric_rows_without_coarsening(monkeypatch):
    stub = _StubTabixFile(
        ["Chr\tStart\tEnd\tS1\tS2"],
        [
            "chr1\t100\t150\t10\t20",
            "chr1\t150\t200\t11\t21",
        ],
    )
    monkeypatch.setattr(highres.pysam, "TabixFile", lambda path: stub)

    result = highres.query_highres_bins(
        "highres.tsv.gz",
        "chr1",
        100,
        200,
        ["S2", "S1", "S3"],
    )

    assert result.index.tolist() == ["chr1:100-150", "chr1:150-200"]
    assert result["source_file"].tolist() == ["highres", "highres"]
    assert result["S1"].tolist() == pytest.approx([10.0, 11.0])
    assert result["S2"].tolist() == pytest.approx([20.0, 21.0])
    assert result["S3"].isna().all()
    assert stub.closed is True


def test_normalize_highres_bins_scales_counts_by_bin_size_ratio(monkeypatch):
    monkeypatch.setattr(highres._util, "VERBOSE", False)
    raw_df = pd.DataFrame(
        {
            "Chr": ["chr1", "chr1"],
            "Start": [100, 150],
            "End": [150, 200],
            "source_file": ["highres", "highres"],
            "S1": [25.0, 50.0],
            "S2": [10.0, 20.0],
        },
        index=["chr1:100-150", "chr1:150-200"],
    )

    normalized = highres.normalize_highres_bins(
        raw_df,
        ["S1", "S2"],
        column_medians=np.array([100.0, 40.0]),
        lowres_median_bin_size=100.0,
    )

    assert normalized["S1"].tolist() == pytest.approx([1.0, 2.0])
    assert normalized["S2"].tolist() == pytest.approx([1.0, 2.0])
    assert raw_df["S1"].tolist() == pytest.approx([25.0, 50.0])
    assert raw_df["S2"].tolist() == pytest.approx([10.0, 20.0])


def test_normalize_highres_bins_emits_verbose_summary(monkeypatch, capsys):
    monkeypatch.setattr(highres._util, "VERBOSE", True)
    raw_df = pd.DataFrame(
        {
            "Chr": ["chr1", "chr1"],
            "Start": [100, 140],
            "End": [140, 180],
            "source_file": ["highres", "highres"],
            "S1": [20.0, 40.0],
        },
        index=["chr1:100-140", "chr1:140-180"],
    )

    normalized = highres.normalize_highres_bins(
        raw_df,
        ["S1"],
        column_medians=np.array([80.0]),
        lowres_median_bin_size=80.0,
    )

    out = capsys.readouterr().out
    assert "[verbose] high-res pre-normalisation" in out
    assert "[verbose] high-res post-normalisation" in out
    assert "[verbose] high-res normalisation summarized across 2 bins" in out
    assert normalized["S1"].tolist() == pytest.approx([1.0, 2.0])