"""Regression checks for sample-level CNV reconciliation."""

import pytest
from test_integrate import _FakeHeader, _FakeRecord, _run_integrate_main

from gatk_sv_gd import integrate
from gatk_sv_gd.reconcile import reconcile_records


def _carrier(record, sample):
    return any(a is not None and a > 0 for a in record.samples.get(sample, {}).get("GT", ()))


def assert_sample_invariants(written, vcf_header, gd_table_rows, gd_calls_entries,
                             samples_ploidy=None, par_intervals=None):
    """Independently check positive-call coverage, uniqueness, and valid coordinates."""
    nahr = {r["gd_id"]: r.get("nahr", "yes") == "yes" for r in gd_table_rows}
    ploidy = dict(samples_ploidy or [])
    rebuilt = [r for r in written if r.info.get("GD_CALL_IDS") or r.info.get("GD_SOURCE_IDS")]
    for record in rebuilt:
        assert 0 <= record.start < record.stop
        assert any(_carrier(record, sample) for sample in vcf_header.samples)
    for sample in vcf_header.samples:
        called = [r for r in written if r.info.get("SVTYPE") in ("DEL", "DUP") and _carrier(r, sample)]
        for event in gd_calls_entries:
            if not nahr.get(event["region_id"], True) or sample not in event.get("samples", []):
                continue
            baseline = ploidy.get(sample, {}).get(event["chrom"], 2)
            if any(c == event["chrom"] and max(0, min(e, event["end"]) - max(s, event["pos"]))
                   >= (event["end"] - event["pos"]) / 2 for c, s, e in (par_intervals or [])):
                baseline = 2
            cn = event.get("cn_state", baseline + (-1 if event["svtype"] == "DEL" else 1))
            for original in called:
                if original in rebuilt or original.chrom != event["chrom"]:
                    continue
                overlap = max(0, min(original.stop, event["end"]) - max(original.start, event["pos"]))
                assert overlap / max(original.stop - original.start, event["end"] - event["pos"]) < 0.5
            covered = sum(max(0, min(r.stop, event["end"]) - max(r.start, event["pos"])) for r in called
                          if r.chrom == event["chrom"] and r.info["SVTYPE"] == event["svtype"]
                          and r.samples[sample].get("RD_CN") == cn
                          and event["region_id"] in r.info.get("GD_CALL_IDS", ()))
            assert covered == event["end"] - event["pos"], (sample, event, covered)


def _record(start, end, samples=("S1",), cn=1, svtype="DEL", name="original", quality=30):
    return _FakeRecord("chr1", start + 1, end, name, (f"<{svtype}>",),
                       {"SVTYPE": svtype, "ALGORITHMS": ("manta",)},
                       {s: {"GT": (1, 1) if cn == 0 else (0, 1), "RD_CN": cn,
                            "GQ": quality, "RD_GQ": quality, "EV": ("RD",)} for s in samples})


def _gd(start=1000, end=2000, samples=("S1",), name="GD1", svtype="DEL", cn=1):
    return {"chrom": "chr1", "pos": start, "end": end, "samples": set(samples),
            "copy_states": dict.fromkeys(samples, cn)}


def _reconcile(records, calls=None, metadata=None, ploidy=None, cutoff=0.5):
    header = _FakeHeader({"chr1": None}, ["S1", "S2"])
    integrate._ensure_headers(header)
    calls = calls if calls is not None else {("GD1", "DEL"): _gd()}
    metadata = metadata or {"GD1": {"cluster": "C", "start": 1000, "end": 2000}}
    return reconcile_records(header, calls, metadata, [(r, {key[0] for key in calls}) for r in records],
                             ploidy or {}, {}, cutoff)


def _calls(records, sample):
    return [(r.start, r.stop, r.info["SVTYPE"], r.samples[sample].get("RD_CN"))
            for r in records if _carrier(r, sample)]


@pytest.mark.parametrize("start,end", [(900, 2100), (500, 3000), (1500, 2500), (500, 1500)])
def test_shifted_calls_replaced_only_in_gd_carriers(start, end):
    original = _record(start, end, ("S1", "S2"))
    result = _reconcile([original])
    gd = next(r for r in result if r.info.get("GD_CALL_IDS"))
    assert (gd.start, gd.stop) == (1000, 2000)
    assert _calls(result, "S2") == [(start, end, "DEL", 1)]
    retained = next(r for r in result if r.id == "original")
    assert retained.samples["S2"] == original.samples["S2"]
    if (start, end) == (500, 3000):  # RO=0.4, below the replacement cutoff.
        assert _calls(result, "S1") == [(start, end, "DEL", 1), (1000, 2000, "DEL", 1)]
        assert retained.samples == original.samples
    else:
        assert _calls(result, "S1") == [(1000, 2000, "DEL", 1)]
        assert retained.samples["S1"]["GT"] == (0, 0)
        assert gd.info["GD_SOURCE_IDS"] == ("original",)
    assert not gd.info.get("GD_ATYPICAL")
    assert gd.info["ALGORITHMS"] == ("depth",)


def test_conflicting_copy_state_replaced_without_flanks():
    result = _reconcile([_record(900, 2100)], {("GD1", "DEL"): _gd(cn=0)})
    assert len(result) == 1
    assert _calls(result, "S1") == [(1000, 2000, "DEL", 0)]
    assert result[0].samples["S1"]["GT"] == (1, 1)
    assert result[0].info["GD_SOURCE_IDS"] == ("original",)


def test_conflicting_svtype_replaced_without_flanks():
    result = _reconcile([_record(900, 2100, cn=3, svtype="DUP")])
    assert len(result) == 1
    assert _calls(result, "S1") == [(1000, 2000, "DEL", 1)]
    assert result[0].info["GD_SOURCE_IDS"] == ("original",)


def test_canonical_noncarrier_removed_but_atypical_noncarrier_retained():
    calls = {("GD1", "DEL"): _gd(samples=())}
    assert _reconcile([_record(1000, 2000)], calls) == []
    assert _calls(_reconcile([_record(900, 2100)], calls), "S1") == [(900, 2100, "DEL", 1)]


def test_overlapping_gd_labels_collapse_without_losing_carriers():
    calls = {("GD1", "DEL"): _gd(), ("GD2", "DEL"): _gd(samples=("S1", "S2"))}
    result = _reconcile([], calls)
    assert len(result) == 1
    assert result[0].info["GD_CALL_IDS"] == ("GD1", "GD2")
    assert _calls(result, "S1") == _calls(result, "S2") == [(1000, 2000, "DEL", 1)]


@pytest.mark.parametrize("svtype,cn", [("DEL", 0), ("DUP", 3)])
def test_conflicting_positive_gd_calls_rejected(svtype, cn):
    calls = {("GD1", "DEL"): _gd(), ("GD2", svtype): _gd(start=1500, end=2500, cn=cn)}
    with pytest.raises(ValueError, match="Conflicting GD copy states"):
        _reconcile([], calls)


def test_adjacent_independent_gd_calls_remain_distinct():
    calls = {("GD1", "DEL"): _gd(), ("GD2", "DEL"): _gd(start=2000, end=3000)}
    assert _calls(_reconcile([], calls), "S1") == [(1000, 2000, "DEL", 1), (2000, 3000, "DEL", 1)]


def test_sample_specific_intervals_and_row_order(tmp_path):
    rows = ["S1\tGD1\tchr1\t1000\t2000\tDEL\tTrue\t0", "S2\tGD1\tchr1\t1200\t2400\tDEL\tTrue\t1"]
    results = []
    for order in (rows, rows[::-1]):
        path = tmp_path / "calls.tsv"
        path.write_text("sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\tcn_state\n" + "\n".join(order))
        result = _reconcile([], integrate.read_gd_calls(str(path)))
        assert _calls(result, "S1") == [(1000, 2000, "DEL", 0)]
        assert _calls(result, "S2") == [(1200, 2400, "DEL", 1)]
        results.append([(r.id, r.start, r.stop, dict(r.info)) for r in result])
    assert results[0] == results[1]


def test_matching_does_not_spread_through_overlap_chains(monkeypatch, tmp_path):
    monkeypatch.setattr(integrate, "_concat_vcf", lambda *a: None)
    header = _FakeHeader({"chr1": None}, ["S1", "S2"])
    records = [_record(100, 800, name="early"), _record(500, 2500, name="broad")]
    table = [{"chr": "chr1", "start": 1000, "end": 2000, "gd_id": "GD1", "svtype": "DEL",
              "nahr": "yes", "cluster": "C", "bp1": "1", "bp2": "2"}]
    calls = [{"chrom": "chr1", "pos": 1000, "end": 2000, "region_id": "GD1", "svtype": "DEL", "samples": ["S1"]}]
    result = _run_integrate_main(monkeypatch, tmp_path, records, header, table, calls)
    assert _calls(result, "S1") == [(100, 800, "DEL", 1), (1000, 2000, "DEL", 1)]
    assert result[0].id == "early"
    assert result[0].samples == records[0].samples and result[0].info == records[0].info
    assert result[1].info["GD_SOURCE_IDS"] == ("broad",)
    assert_sample_invariants(result, header, table, calls)


def test_gd_quality_replaces_original_quality():
    calls = {("GD1", "DEL"): _gd()}
    calls[("GD1", "DEL")]["copy_probabilities"] = {"S1": {1: 0.99, 2: 0.01}}
    result = _reconcile([_record(900, 2100, quality=10)], calls)
    assert len(result) == 1
    assert result[0].samples["S1"]["GQ"] == 20
    assert result[0].samples["S1"]["RD_GQ"] == 20


def test_unmatched_original_fields_stay_identical():
    original = _record(900, 2100, samples=("S2",))
    del original.samples["S2"]["RD_CN"]
    del original.samples["S2"]["EV"]
    result = _reconcile([original])
    retained = next(r for r in result if r.id == original.id)
    assert retained.samples == original.samples
    assert retained.info == original.info


def test_atypical_haploid_input_uses_diploid_encoded_allosome_gt():
    original = _record(900, 2100, cn=0)
    original.samples["S1"]["GT"] = (1,)
    result = _reconcile([original], {("GD1", "DEL"): _gd(cn=0)}, ploidy={"S1": {"chr1": 1}})
    assert len(result) == 1
    assert result[0].samples["S1"]["GT"] == (0, 1)
    assert result[0].samples["S1"]["ECN"] == 1


def test_vcf_conflicts_without_positive_gd_are_not_arbitrarily_resolved():
    records = [_record(900, 2100, cn=0, name="first"), _record(900, 2100, cn=1, name="second")]
    calls = {("GD1", "DEL"): _gd(samples=())}
    result = _reconcile(records, calls)
    assert [(r.id, r.samples) for r in result] == [(r.id, r.samples) for r in records]


def test_retained_vcf_event_without_algorithm_does_not_invent_depth_source():
    original = _record(900, 2100, svtype="DUP", cn=3)
    del original.info["ALGORITHMS"]
    result = _reconcile([original], {("GD1", "DEL"): _gd(samples=())})
    assert len(result) == 1
    assert "ALGORITHMS" not in result[0].info


@pytest.mark.parametrize("start,end,cutoff,replaced", [
    (1500, 2500, 0.5, True),
    (1501, 2501, 0.5, False),
    (900, 2100, 0.9, False),
    (500, 3000, 0.4, True),
    (2000, 3000, 0.5, False),
])
def test_replacement_overlap_threshold(start, end, cutoff, replaced):
    original = _record(start, end)
    result = _reconcile([original], cutoff=cutoff)
    assert any(r.id == "original" for r in result) is not replaced
    gd = next(r for r in result if r.info.get("GD_CALL_IDS"))
    assert (gd.start, gd.stop) == (1000, 2000)
    assert len(result) == (1 if replaced else 2)


@pytest.mark.parametrize("original_type,original_cn,gd_type,gd_cn", [
    ("DEL", 0, "DEL", 1), ("DEL", 1, "DEL", 0),
    ("DUP", 3, "DUP", 5), ("DEL", 1, "DUP", 4), ("DUP", 3, "DEL", 0),
])
def test_model_copy_state_and_type_win_regardless_of_original_quality(original_type, original_cn, gd_type, gd_cn):
    original = _record(900, 2100, cn=original_cn, svtype=original_type, quality=99)
    gd = _gd(cn=gd_cn)
    gd["copy_probabilities"] = {"S1": {gd_cn: 0.51}}
    result = _reconcile([original], {("GD1", gd_type): gd})
    assert len(result) == 1
    assert _calls(result, "S1") == [(1000, 2000, gd_type, gd_cn)]
    assert result[0].samples["S1"]["GQ"] == 3
    assert result[0].samples["S1"]["RD_GQ"] == 3


def test_multiple_distorted_original_calls_are_all_replaced():
    result = _reconcile([_record(900, 2100, name="one"), _record(800, 2200, name="two", cn=0)])
    assert len(result) == 1
    assert _calls(result, "S1") == [(1000, 2000, "DEL", 1)]
    assert result[0].info["GD_SOURCE_IDS"] == ("one", "two")


@pytest.mark.parametrize("gt", [(0, 0), (None, None)])
def test_incidentally_overlapping_zero_carrier_records_survive(gt):
    original = _record(500, 3000)
    original.samples["S1"]["GT"] = gt
    result = _reconcile([original])
    assert len(result) == 2
    assert result[0].id == original.id
    assert result[0].samples == original.samples and result[0].info == original.info


def test_dense_chain_across_samples_does_not_expand_replacement(monkeypatch, tmp_path):
    monkeypatch.setattr(integrate, "_concat_vcf", lambda *a: None)
    header = _FakeHeader({"chr1": None}, ["S1", "S2"])
    records = [_record(900, 2100, name="target")]
    records.extend(_record(start, start + 1000, samples=("S2",), name=f"other_{start}")
                   for start in range(1900, 20000, 500))
    table = [{"chr": "chr1", "start": 1000, "end": 2000, "gd_id": "GD1", "svtype": "DEL",
              "nahr": "yes", "cluster": "C", "bp1": "1", "bp2": "2"}]
    calls = [{"chrom": "chr1", "pos": 1000, "end": 2000, "region_id": "GD1", "svtype": "DEL", "samples": ["S1"]}]
    result = _run_integrate_main(monkeypatch, tmp_path, records, header, table, calls)
    by_id = {r.id: r for r in result}
    assert len(result) == len(records)
    for original in records[1:]:
        assert by_id[original.id].samples == original.samples
        assert by_id[original.id].info == original.info
    assert _calls(result, "S1") == [(1000, 2000, "DEL", 1)]
    assert by_id["GD1"].info["GD_SOURCE_IDS"] == ("target",)
