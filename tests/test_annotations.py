import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gatk_sv_gd.annotations import (
    FlankCompressor,
    GTFParser,
    GapsAnnotation,
    SegDupAnnotation,
    _draw_overview_column,
    _build_gap_positions,
    _build_heatmap_matrix,
    _build_ploidy_lookup,
    _depths_with_gaps,
    _sort_samples_by_ploidy,
    draw_annotations_panel,
    get_sample_columns,
)
from gatk_sv_gd.models import GDLocus


class _StubGTF:
    def __init__(self, genes):
        self._genes = genes

    def get_genes_in_region(self, chrom, start, end, gene_types=None):
        return list(self._genes)


class _StubRanges:
    def __init__(self, regions):
        self._regions = regions

    def get_regions_in_range(self, chrom, start, end):
        return list(self._regions)


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
            },
            {
                "GD_ID": "GD2",
                "start_GRCh38": 110,
                "end_GRCh38": 210,
                "svtype": "DUP",
                "BP1": "A",
                "BP2": "B",
            },
        ],
        is_nahr=True,
        is_terminal=False,
    )


def test_gtf_parser_parses_records_and_filters_region_queries(tmp_path):
    gtf_path = tmp_path / "genes.gtf"
    gtf_path.write_text(
        "# comment\n"
        "chr1\tsrc\tgene\t100\t200\t.\t+\t.\tgene_id \"g1\"; gene_name \"GENE1\"; gene_type \"protein_coding\";\n"
        "chr1\tsrc\ttranscript\t120\t180\t.\t+\t.\tgene_id \"g1\"; gene_name \"GENE1\"; gene_type \"protein_coding\"; transcript_id \"tx1\"; transcript_type \"protein_coding\";\n"
        "chr1\tsrc\tgene\t250\t320\t.\t-\t.\tgene_id \"g2\"; gene_name \"GENE2\"; gene_type \"lncRNA\";\n"
        "chr1\tsrc\ttranscript\t260\t310\t.\t-\t.\tgene_id \"g2\"; gene_name \"GENE2\"; gene_type \"lncRNA\"; transcript_id \"tx2\"; transcript_type \"lncRNA\";\n"
        "chr1\tsrc\texon\t120\t130\t.\t+\t.\tgene_id \"g1\"; gene_name \"GENE1\";\n"
    )

    parser = GTFParser(str(gtf_path))

    genes = parser.get_genes_in_region("chr1", 150, 270)
    assert [gene["gene_name"] for gene in genes] == ["GENE1", "GENE2"]

    protein_genes = parser.get_genes_in_region("chr1", 150, 270, gene_types=["protein_coding"])
    assert [gene["gene_id"] for gene in protein_genes] == ["g1"]

    transcripts = parser.get_transcripts_in_region("chr1", 150, 270)
    assert [tx["transcript_id"] for tx in transcripts] == ["tx1", "tx2"]

    lncrna_transcripts = parser.get_transcripts_in_region("chr1", 150, 270, transcript_types=["lncRNA"])
    assert [tx["transcript_id"] for tx in lncrna_transcripts] == ["tx2"]
    assert parser.get_genes_in_region("chr9", 0, 10) == []


def test_segdup_and_gap_annotations_return_only_overlaps(tmp_path):
    segdup_path = tmp_path / "segdup.bed"
    segdup_path.write_text(
        "chr1\t100\t150\tseg1\t0\t+\n"
        "chr1\t180\t220\tseg2\t0\t+\n"
        "chr2\t10\t20\tseg3\t0\t+\n"
    )
    gaps_path = tmp_path / "gaps.bed"
    gaps_path.write_text(
        "# gap comment\n"
        "chr1\t90\t95\n"
        "chr1\t140\t160\n"
        "chr1\t300\t320\n"
    )

    segdup = SegDupAnnotation(str(segdup_path))
    gaps = GapsAnnotation(str(gaps_path))

    assert segdup.get_regions_in_range("chr1", 120, 200) == [(100, 150), (180, 220)]
    assert segdup.get_regions_in_range("chr3", 0, 100) == []
    assert gaps.get_regions_in_range("chr1", 100, 200) == [(140, 160)]
    assert gaps.get_regions_in_range("chr2", 0, 100) == []


def test_ploidy_gap_and_depth_helpers():
    ploidy_df = pd.DataFrame(
        {
            "sample": ["S2", "S1", "S1"],
            "contig": ["chr1", "chr1", "chr2"],
            "ploidy": [1, 2, 3],
        }
    )

    lookup = _build_ploidy_lookup(ploidy_df)

    assert lookup == {("S2", "chr1"): 1, ("S1", "chr1"): 2, ("S1", "chr2"): 3}
    assert _build_ploidy_lookup(None) == {}
    assert _sort_samples_by_ploidy(["S3", "S1", "S2"], "chr1", lookup) == ["S2", "S1", "S3"]

    bin_mids = np.array([5.0, 15.0, 25.0, 130.0])
    bin_starts = np.array([0, 10, 20, 120])
    bin_ends = np.array([10, 20, 30, 140])
    gap_positions = _build_gap_positions(bin_mids, bin_starts, bin_ends)
    assert np.allclose(gap_positions[[0, 1, 2, 4]], [5.0, 15.0, 25.0, 130.0])
    assert np.isnan(gap_positions[3])
    assert _build_gap_positions(np.array([]), np.array([]), np.array([])).size == 0
    assert np.array_equal(_build_gap_positions(np.array([7.0]), np.array([0]), np.array([10])), np.array([7.0]))

    depths = _depths_with_gaps(np.array([1.0, 2.0, 3.0, 4.0]), gap_positions)
    assert np.allclose(depths[[0, 1, 2, 4]], [1.0, 2.0, 3.0, 4.0])
    assert np.isnan(depths[3])


def test_build_heatmap_matrix_supports_empty_standard_and_compressed_views():
    region_df = pd.DataFrame(
        {
            "Start": [0, 10, 20, 30],
            "End": [10, 20, 30, 40],
            "S1": [1.0, 2.0, 3.0, 4.0],
            "S2": [10.0, 20.0, 30.0, 40.0],
        }
    )
    sample_cols = ["S1", "S2"]

    matrix = _build_heatmap_matrix(region_df, sample_cols, 0, 40, n_viz_bins=4)
    assert np.allclose(matrix, np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]))

    xform = FlankCompressor(0, 40, 10, 30, flank_scale=0.2)
    compressed = _build_heatmap_matrix(region_df, sample_cols, 0, 40, xform=xform, n_viz_bins=4)
    assert np.allclose(compressed, matrix)

    empty = _build_heatmap_matrix(region_df.iloc[0:0], sample_cols, 0, 40, n_viz_bins=3)
    assert empty.shape == (2, 3)
    assert np.isnan(empty).all()


def test_draw_annotations_panel_and_get_sample_columns_render_expected_labels():
    locus = _make_locus()
    gtf = _StubGTF(
        [
            {"start": 120, "end": 180, "gene_name": "GENE1"},
            {"start": 240, "end": 290, "gene_name": "GENE2"},
        ]
    )
    segdup = _StubRanges([(130, 170)])
    gaps = _StubRanges([(260, 280)])
    region_df = pd.DataFrame(
        {
            "Cluster": ["cluster1"],
            "Chr": ["chr1"],
            "Start": [50],
            "End": [350],
            "source_file": ["x"],
            "Bin": [0],
            "S1": [1.0],
            "S2": [2.0],
        }
    )

    assert get_sample_columns(region_df) == ["S1", "S2"]

    fig, ax = plt.subplots()
    try:
        draw_annotations_panel(
            ax,
            locus,
            50,
            350,
            "chr1",
            "Annotation Title",
            gtf=gtf,
            segdup=segdup,
            gaps=gaps,
            show_gd_entries=True,
        )

        text_labels = {text.get_text() for text in ax.texts}
        assert "Annotation Title" == ax.get_title()
        assert "Annotations" == ax.get_ylabel()
        assert {"left flank", "right flank", "A-B", "B-C", "BP A", "BP B", "BP C", "GENE1", "GENE2"} <= text_labels
        assert list(ax.get_yticks()) == []
        assert len(ax.collections) >= 1
        assert len(ax.patches) >= 8
    finally:
        plt.close(fig)


def test_draw_overview_column_renders_heatmaps_mean_depths_and_trace_panels():
    locus = _make_locus()
    region_df = pd.DataFrame(
        {
            "Cluster": ["cluster1"] * 4,
            "Chr": ["chr1"] * 4,
            "Start": [50, 110, 210, 310],
            "End": [100, 200, 300, 350],
            "S1": [1.1, 0.8, 0.9, 1.0],
            "S2": [2.1, 2.2, 2.0, 2.1],
            "S3": [2.0, 2.1, 1.9, 2.0],
        }
    )
    calls_df = pd.DataFrame(
        {
            "cluster": ["cluster1", "cluster1"],
            "sample": ["S1", "S2"],
            "is_carrier": [True, True],
            "is_best_match": [True, True],
            "start": [110, 110],
            "end": [300, 210],
            "svtype": ["DEL", "DUP"],
        }
    )
    ploidy_lookup = {("S1", "chr1"): 2, ("S2", "chr1"): 2, ("S3", "chr1"): 2}
    xform = FlankCompressor(50, 350, locus.start, locus.end, flank_scale=0.2)

    fig, axes = plt.subplots(5, 1, figsize=(6, 10))
    try:
        _draw_overview_column(
            list(axes),
            region_df,
            locus,
            calls_df,
            50,
            350,
            carrier_cols=["S1", "S2"],
            non_carrier_cols=["S3"],
            all_ploidies=[2],
            ploidy_lookup=ploidy_lookup,
            sample_cols=["S1", "S2", "S3"],
            carriers={"S1", "S2"},
            gtf=_StubGTF([{"start": 120, "end": 180, "gene_name": "GENE1"}]),
            segdup=_StubRanges([(130, 170)]),
            col_title="Overview",
            show_colorbar=False,
            gaps=_StubRanges([(260, 280)]),
            xform=xform,
            heatmap_viz_bins=8,
        )

        assert axes[0].get_title() == "Overview"
        assert axes[1].get_ylabel() == "Carriers (n=2)"
        assert axes[2].get_ylabel() == "Non-carriers (n=1)"
        assert axes[3].get_ylabel() == "Mean depth (P=2)"
        assert axes[4].get_ylabel() == "All depths"
        assert axes[4].get_xlabel() == "Position on chr1"
        assert len(axes[1].images) == 1
        assert len(axes[2].images) == 1
        assert len(axes[3].lines) >= 4
        assert len(axes[4].lines) >= 3
    finally:
        plt.close(fig)


def test_draw_overview_column_handles_missing_carrier_and_noncarrier_groups():
    locus = _make_locus()
    region_df = pd.DataFrame(
        {
            "Cluster": ["cluster1", "cluster1"],
            "Chr": ["chr1", "chr1"],
            "Start": [100, 200],
            "End": [150, 250],
        }
    )
    fig, axes = plt.subplots(4, 1, figsize=(6, 8))
    try:
        _draw_overview_column(
            list(axes),
            region_df,
            locus,
            pd.DataFrame(columns=["cluster", "sample", "is_carrier", "is_best_match", "start", "end", "svtype"]),
            100,
            250,
            carrier_cols=[],
            non_carrier_cols=[],
            all_ploidies=[],
            ploidy_lookup={},
            sample_cols=[],
            carriers=set(),
            gtf=None,
            segdup=None,
            col_title="Empty Overview",
            show_colorbar=False,
            gaps=None,
            xform=FlankCompressor(100, 250, locus.start, locus.end, flank_scale=None),
            heatmap_viz_bins=4,
        )

        assert axes[1].texts[0].get_text() == "No carriers"
        assert axes[2].texts[0].get_text() == "No non-carriers"
        assert axes[3].get_ylabel() == "All depths"
    finally:
        plt.close(fig)