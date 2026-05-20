import pytest

from gatk_sv_gd.models import GDLocus


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