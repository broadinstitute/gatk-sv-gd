"""Actual VCF round-trip checks for sample-level replacement and breakpoints."""

from itertools import combinations

import pysam

from gatk_sv_gd import integrate


def run_case(root, name, original, calls, expected, error=None):
    work = root / name
    work.mkdir()
    header = pysam.VariantHeader()
    header.contigs.add("chr1")
    integrate._ensure_headers(header)
    for line in (
        '##INFO=<ID=AC,Number=A,Type=Integer,Description="Alternate allele count">',
        '##INFO=<ID=AN,Number=1,Type=Integer,Description="Called allele count">',
        '##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency">',
    ):
        header.add_line(line)
    for field in ("PE_GT", "PE_GQ", "SR_GT", "SR_GQ"):
        header.add_line(f'##FORMAT=<ID={field},Number=1,Type=Integer,Description="Breakpoint evidence">')
    for sample in ("S1", "S2"):
        header.add_sample(sample)
    with pysam.VariantFile(str(work / "in.vcf"), "w", header=header) as out:
        for start, end, svtype, samples in original:
            record = header.new_record(contig="chr1", start=start, stop=end,
                                       id=f"original_{start}_{end}", alleles=("N", f"<{svtype}>"))
            record.info["SVTYPE"] = svtype
            record.info["AC"], record.info["AN"], record.info["AF"] = (len(samples),), 4, (len(samples) / 4,)
            record.info["SVLEN"] = end - start - 1
            record.info["ALGORITHMS"] = ("manta",)
            for sample in header.samples:
                record.samples[sample]["GT"] = (0, 1) if sample in samples else (0, 0)
                record.samples[sample]["PE_GT"] = 1 if sample in samples else 0
                record.samples[sample]["PE_GQ"] = 30
                record.samples[sample]["RD_CN"] = (1 if svtype == "DEL" else 3) if sample in samples else 2
            out.write(record)
    loci = sorted({(gd_id, chrom, svtype) for sample, gd_id, chrom, start, end, svtype, cn in calls})
    (work / "gd.tsv").write_text(
        "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\tcluster\tBP1\tBP2\n"
        + "".join(f"{chrom}\t1000\t2000\t{gd_id}\t{svtype}\tyes\tno\tC\t1\t2\n"
                  for gd_id, chrom, svtype in loci)
    )
    rows = [f"{s}\t{g}\t{c}\t{a}\t{b}\t{t}\tTrue\t{cn}\n" for s, g, c, a, b, t, cn in calls]
    for gd_id, chrom, svtype in loci:
        for sample in header.samples:
            if not any(row[0] == sample and row[1] == gd_id and row[5] == svtype for row in calls):
                rows.append(f"{sample}\t{gd_id}\t{chrom}\t1000\t2000\t{svtype}\tFalse\t\n")
    (work / "calls.tsv").write_text(
        "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\tcn_state\n" + "".join(rows)
    )
    (work / "ploidy.tsv").write_text("sample\tchr1\tchr2\nS1\t2\t2\nS2\t2\t2\n")
    (work / "par.bed").write_text("")
    out_path = work / "out.vcf.gz"
    argv = ["--vcf", str(work / "in.vcf"), "--gd-calls", str(work / "calls.tsv"),
            "--gd-table", str(work / "gd.tsv"), "--ploidy-table", str(work / "ploidy.tsv"),
            "--par-bed", str(work / "par.bed"), "--out-vcf", str(out_path), "--temp-dir", str(work)]
    if error:
        out_path.write_bytes(b"previous valid output")
        try:
            integrate.main(argv)
        except ValueError as exc:
            assert error in str(exc)
        else:
            raise AssertionError("Invalid GD input was accepted")
        assert out_path.read_bytes() == b"previous valid output"
        return
    with pysam.VariantFile(str(work / "in.vcf")) as vcf:
        originals = {r.id: str(r) for r in vcf}
    integrate.main(argv)
    with pysam.VariantFile(str(out_path)) as vcf:
        records = list(vcf.fetch())  # Require a usable index and actual serialization.
        assert {r.chrom for r in records}.issubset(vcf.header.contigs)
    assert len({r.id for r in records}) == len(records)
    observed = {s: [] for s in header.samples}
    for record in records:
        if name == "weak_overlap" and record.id in originals:
            assert str(record) == originals[record.id]
        if name == "indirect_overlap" and record.id == "original_100_800":
            assert str(record) == originals[record.id]
        if name == "atypical_carrier_preserved" and record.id in originals:
            assert record.samples["S1"]["GT"] == (0, 0)
            assert record.samples["S1"].get("PE_GT") is None
            assert record.samples["S1"].get("PE_GQ") is None
            assert record.samples["S2"]["PE_GT"] == 1
            assert record.samples["S2"]["PE_GQ"] == 30
        ac = sum(a == 1 for s in record.samples.values() for a in s["GT"])
        an = sum(a is not None for s in record.samples.values() for a in s["GT"])
        assert record.info["AC"] == (ac,) and record.info["AN"] == an
        assert abs(record.info["AF"][0] - ac / an) < 1e-6
        assert record.info["SVLEN"] == record.stop - record.pos
        for sample, gt in record.samples.items():
            if any(a is not None and a > 0 for a in gt["GT"]):
                observed[sample].append((record.chrom, record.start, record.stop, gt["RD_CN"]))
                if name in ("state_transition", "opposite_vcf_type"):
                    assert gt.get("PE_GT") is None and gt.get("PE_GQ") is None
                assert gt["GT"] == ((1, 1) if abs(gt["RD_CN"] - 2) >= 2 else (0, 1))
    assert observed == expected, (name, observed, expected)
    for intervals in observed.values():
        for left, right in combinations(intervals, 2):
            if left[0] == right[0]:
                overlap = max(0, min(left[2], right[2]) - max(left[1], right[1]))
                assert overlap / max(left[2] - left[1], right[2] - right[1]) < 0.5
    return [(r.id, str(r).strip()) for r in records]


def run_regressions(root):
    gd = ("S1", "GD1", "chr1", 1000, 2000, "DEL", 1)
    run_case(root, "atypical_carrier_preserved", [(900, 2100, "DEL", ("S1", "S2"))], [gd],
             {"S1": [("chr1", 1000, 2000, 1)], "S2": [("chr1", 900, 2100, 1)]})
    run_case(root, "indirect_overlap", [(100, 800, "DEL", ("S1",)), (500, 2500, "DEL", ("S1",))], [gd],
             {"S1": [("chr1", 100, 800, 1), ("chr1", 1000, 2000, 1)], "S2": []})
    run_case(root, "state_transition", [(900, 2100, "DEL", ("S1",))], [(*gd[:-1], 0)],
             {"S1": [("chr1", 1000, 2000, 0)], "S2": []})
    run_case(root, "opposite_vcf_type", [(900, 2100, "DUP", ("S1",))], [gd],
             {"S1": [("chr1", 1000, 2000, 1)], "S2": []})
    run_case(root, "weak_overlap", [(500, 3000, "DUP", ("S1",))], [gd],
             {"S1": [("chr1", 500, 3000, 3), ("chr1", 1000, 2000, 1)], "S2": []})
    calls = [gd, ("S2", "GD1", "chr1", 1200, 2400, "DEL", 0)]
    expected = {"S1": [("chr1", 1000, 2000, 1)], "S2": [("chr1", 1200, 2400, 0)]}
    forward = run_case(root, "sample_specific", [], calls, expected)
    reverse = run_case(root, "sample_specific_reversed", [], calls[::-1], expected)
    assert forward == reverse
    run_case(root, "duplicate_gd", [], [gd, ("S1", "GD2", "chr1", 1000, 2000, "DEL", 1)],
             {"S1": [("chr1", 1000, 2000, 1)], "S2": []})
    run_case(root, "new_contig", [], [("S1", "GD2", "chr2", 1000, 2000, "DUP", 4)],
             {"S1": [("chr2", 1000, 2000, 4)], "S2": []})
    for svtype, cn in (("DEL", 0), ("DUP", 3)):
        run_case(root, f"conflicting_gd_{svtype}", [],
                 [gd, ("S1", "GD2", "chr1", 1500, 2500, svtype, cn)], None, "Conflicting GD copy states")
