"""Real VCF regression checks; launched separately to avoid pytest dependency stubs."""

import json
import os
import sys
from pathlib import Path

import pysam

from gatk_sv_gd import integrate


def run_case(root, name, svtype="DEL", cn=0, ecn=2, other_ecn=2, matched=True,
             par=False, carrier=True, incomplete=False, nahr=True,
             probabilities=None, expected_gq=None, expected_rd_gq=None):
    work = root / name
    work.mkdir()
    header = pysam.VariantHeader()
    chrom = "chr1" if ecn == 2 else "chrX"
    header.contigs.add(chrom)
    # Intentionally omit EV and ECN: integrate must declare every field it writes.
    header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type">')
    header.add_line('##INFO=<ID=END,Number=1,Type=Integer,Description="End">')
    header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    for sample in ("S1", "S2"):
        header.add_sample(sample)
    with pysam.VariantFile(str(work / "in.vcf"), "w", header=header) as out:
        for start, end, record_id in [(100, 200, "before"), (1000, 2000, "matched"), (3000, 4000, "after")]:
            if record_id == "matched" and not matched:
                continue
            record = header.new_record(contig=chrom, start=start, stop=end, id=record_id, alleles=("N", f"<{svtype}>"))
            record.info["SVTYPE"] = svtype
            record.samples["S1"]["GT"] = (0, 0)
            record.samples["S2"]["GT"] = (0, 1)
            out.write(record)
    (work / "gd.tsv").write_text(
        "chr\tstart_GRCh38\tend_GRCh38\tGD_ID\tsvtype\tNAHR\tterminal\tcluster\tBP1\tBP2\n"
        f"{chrom}\t1000\t2000\tGD1\t{svtype}\t{'yes' if nahr else 'no'}\tno\tC\t1\t2\n"
    )
    summary = json.dumps(probabilities) if probabilities is not None else ""
    calls = f"S1\tGD1\t{chrom}\t1000\t2000\t{svtype}\t{carrier}\t{cn}\t{summary}\n"
    if not incomplete:
        ref_summary = json.dumps({2 if par else other_ecn: 1.0}) if probabilities is not None else ""
        calls += f"S2\tGD1\t{chrom}\t1000\t2000\t{svtype}\tFalse\t\t{ref_summary}\n"
    (work / "calls.tsv").write_text(
        "sample\tGD_ID\tchrom\tstart\tend\tsvtype\tis_carrier\tcn_state\tcn_probabilities\n" + calls
    )
    (work / "ploidy.tsv").write_text(f"sample\t{chrom}\nS1\t{ecn}\nS2\t{other_ecn}\n")
    (work / "par.bed").write_text(f"{chrom}\t900\t2100\n" if par else "")
    out_path = work / "out.vcf.gz"
    argv = [
        "--vcf", str(work / "in.vcf"), "--gd-calls", str(work / "calls.tsv"),
        "--gd-table", str(work / "gd.tsv"), "--ploidy-table", str(work / "ploidy.tsv"),
        "--par-bed", str(work / "par.bed"), "--out-vcf", str(out_path), "--temp-dir", str(work),
    ]
    if (incomplete and nahr) or (ecn == 0 and carrier):
        original = (work / "in.vcf").read_bytes()
        out_path.write_bytes(b"existing output must survive validation failure")
        try:
            integrate.main(argv)
        except ValueError as exc:
            assert ("zero sample ploidy" if ecn == 0 else "missing evaluations") in str(exc)
        else:
            raise AssertionError("Inconsistent calls must be rejected")
        assert (work / "in.vcf").read_bytes() == original
        assert out_path.read_bytes() == b"existing output must survive validation failure"
        return
    integrate.main(argv)
    with pysam.VariantFile(str(out_path)) as result:
        records = list(result)
    assert [r.pos for r in records] == sorted(r.pos for r in records)
    assert records[0].id == "before" and records[-1].id == "after"
    assert records[0].samples["S2"]["GT"] == (0, 1)
    with pysam.VariantFile(str(out_path)) as result:
        gd_records = list(result.fetch(chrom, 900, 2100))
    if not nahr:
        assert len(gd_records) == 1
        assert gd_records[0].id == "matched"
        assert gd_records[0].samples["S2"]["GT"] == (0, 1)
        return
    if not carrier or ecn == 0:
        assert gd_records == []
        return
    assert len(gd_records) == 1
    record = gd_records[0]
    assert record.id == ("GD1" if matched else f"GD1_{svtype}_novel")
    assert (record.start, record.stop) == (1000, 2000)
    assert record.info["SVLEN"] == 999
    baseline = 2 if par else ecn
    assert record.samples["S1"]["GT"] == ((0, 1) if abs(cn - baseline) == 1 else (1, 1))
    assert record.samples["S1"]["RD_CN"] == cn
    assert record.samples["S1"]["ECN"] == ecn
    assert record.samples["S2"]["ECN"] == other_ecn
    if other_ecn == 0:
        assert record.samples["S2"]["GT"] == (None, None)
        assert record.samples["S2"]["RD_CN"] is None
    else:
        assert record.samples["S2"]["GT"] == (0, 0)
        assert record.samples["S2"]["RD_CN"] == (2 if par else other_ecn)
    for sample_name, sample in record.samples.items():
        gq, rd_gq = expected_gq, expected_rd_gq
        if sample_name == "S2":
            gq = rd_gq = 99 if other_ecn > 0 and probabilities is not None else None
        assert sample["GQ"] == gq and sample["RD_GQ"] == rd_gq
        assert len(sample["GT"]) == 2


def main():
    root = Path(sys.argv[1])
    bin_dir = root / "bin"
    bin_dir.mkdir()
    # Run the bundled, real bcftools implementation through the production
    # subprocess path without requiring a separately installed executable.
    executable = bin_dir / "bcftools"
    executable.write_text(
        f"#!{sys.executable}\nimport sys\nimport pysam.bcftools\n"
        "getattr(pysam.bcftools, sys.argv[1])(*sys.argv[2:], catch_stdout=False)\n"
    )
    executable.chmod(0o755)
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    for matched in (True, False):
        for cn in (0, 1, 3, 4, 5, 6):
            run_case(root, f"diploid_{cn}_{matched}", "DEL" if cn < 2 else "DUP", cn, matched=matched)
    for svtype, cn in (("DEL", 0), ("DUP", 2), ("DUP", 3), ("DUP", 4)):
        run_case(root, f"allosome_{svtype}_{cn}", svtype, cn, ecn=1, other_ecn=0)
    run_case(root, "par", cn=0, ecn=1, par=True)
    run_case(root, "no_carriers", ecn=1, other_ecn=0, carrier=False)
    run_case(root, "zero_ploidy_carrier", ecn=0, carrier=True)
    run_case(root, "incomplete", carrier=False, incomplete=True)
    run_case(root, "incomplete_non_nahr", carrier=False, incomplete=True, nahr=False)
    run_case(root, "quality_ambiguous", probabilities={0: 0.51, 1: 0.49}, expected_gq=3, expected_rd_gq=3)
    run_case(root, "quality_strong", probabilities={0: 0.99, 1: 0.01}, expected_gq=20, expected_rd_gq=20)
    run_case(root, "quality_dup", "DUP", 4, probabilities={4: 0.51, 5: 0.49}, expected_gq=99, expected_rd_gq=3)
    run_case(root, "quality_allosome", "DUP", 3, ecn=1, other_ecn=0,
             probabilities={3: 0.51, 4: 0.49}, expected_gq=99, expected_rd_gq=3)
    from reconcile_vcf_smoke import run_regressions

    run_regressions(root)
    (root / "passed").touch()


if __name__ == "__main__":
    main()
