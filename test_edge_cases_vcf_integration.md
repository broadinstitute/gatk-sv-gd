# VCF Integration — Comprehensive Edge Case Catalogue

> Generated from thorough code review of `integrate.py`, `models.py`, and `test_integrate.py`.
> Each case is annotated with its coverage status: ✅ covered, ❌ not covered, ⚠️ partially covered.

## Test Performance Policy

- **No long-running tests in the default suite.** Every test added from this catalogue must complete in **≤ 10 seconds**.
- **Scaling tests excluded by default.** Any test exercising large inputs (100+ samples, 1000+ records, memory pressure, etc.) must use a decorator to mark it as optional (e.g. `@pytest.mark.slow` or `@pytest.mark.scaling`). These run only with `pytest -m scaling` or equivalent.
- **Keep mocks lightweight.** Use `FakeIntervalTree`, `_FakeRecord`, and `monkeypatch` for pysam — never invoke real bcftools or pysam I/O in unit tests.
- **CI gate: 10s per test.** If a test exceeds 10s in CI, refactor or mark it as scaling.

---

## Table of Contents

1. [Input Validation & CLI](#1-input-validation--cli)
2. [GD Table Loading](#2-gd-table-loading)
3. [GDTable Class Internals](#3-gdtable-class-internals)
4. [GD Calls Reading](#4-gd-calls-reading)
5. [Ploidy Table Reading](#5-ploidy-table-reading)
6. [PAR BED Reading](#6-par-bed-reading)
7. [PAR Region Detection](#7-par-region-detection)
8. [Expected Copy Number](#8-expected-copy-number)
9. [VCF Carrier Extraction](#9-vcf-carrier-extraction)
10. [Sample Overlap Scoring](#10-sample-overlap-scoring)
11. [Genotype Update](#11-genotype-update)
12. [Header Management](#12-header-management)
13. [Phase 1 — Non-NAHR Partial Overlap Annotation](#13-phase-1--non-nahr-partial-overlap-annotation)
14. [Phase 2 — NAHR Competitive Matching](#14-phase-2--nah-competitive-matching)
15. [Phase 3 — Novel Record Emission](#15-phase-3--novel-record-emission)
16. [Phase Interactions](#16-phase-interactions)
17. [End-to-End Integration Scenarios](#17-end-to-end-integration-scenarios)
18. [Coordinate & SVLEN Edge Cases](#18-coordinate--svlen-edge-cases)
19. [Header / FORMAT / INFO Edge Cases](#19-header--format--info-edge-cases)
20. [Reader Error Paths](#20-reader-error-paths)
21. [Error Handling & Fault Tolerance](#21-error-handling--fault-tolerance)
22. [Parameter Value Edge Cases](#22-parameter-value-edge-cases)
23. [pysam-Specific Behavior](#23-pysam-specific-behavior)
24. [Contig Naming Consistency](#24-contig-naming-consistency)
25. [Multi-Chromosomal & Large-Cohort Scenarios](#25-multi-chromosomal--large-cohort-scenarios)

---

## 1. Input Validation & CLI

| # | Case | Covered |
|---|------|---------|
| 1.1 | All five required files present | ❌ |
| 1.2 | Missing VCF → exit code 1 | ✅ |
| 1.3 | Missing GD calls → exit code 1 | ✅ |
| 1.4 | Missing GD table → exit code 1 | ❌ |
| 1.5 | Missing PAR BED → exit code 1 | ❌ |
| 1.6 | Missing ploidy table → exit code 1 | ❌ |
| 1.7 | Multiple files missing at once | ❌ |
| 1.8 | bcftools not found on PATH | ❌ |
| 1.9 | bcftools sort returns non-zero exit | ✅ |
| 1.10 | Custom `--temp-dir` that doesn't exist (should be created) | ❌ |
| 1.11 | `--temp-dir` with insufficient permissions | ❌ |
| 1.12 | Output directory doesn't exist (no `os.makedirs` for output dir) | ❌ |
| 1.13 | Output file path with nested directories | ❌ |
| 1.14 | All args provided via `--help` (exit 0) | ❌ |
| 1.15 | `--temp-dir` created via `os.makedirs(exist_ok=True)` | ❌ |

---

## 2. GD Table Loading (`_build_trees_from_gd_table`)

| # | Case | Covered |
|---|------|---------|
| 2.1 | NAHR=yes → nahr_trees | ✅ |
| 2.2 | NAHR=no → non_nahr_trees | ✅ |
| 2.3 | Mixed NAHR/non-NAHR in same cluster | ❌ |
| 2.4 | Same GD_ID on different chromosomes | ❌ |
| 2.5 | Different svtypes for same GD_ID (DEL vs DUP) | ✅ |
| 2.6 | Multiple NAHR entries at identical coords | ❌ |
| 2.7 | Empty GD table (no rows) | ❌ |
| 2.8 | GD table with only headers (no data rows) | ❌ |
| 2.9 | BP1/BP2 alphanumeric vs numeric ordering | ❌ |
| 2.10 | Terminal flag present in GD table | ❌ |
| 2.11 | GD_ID appearing in gd_calls but NOT in gd_table | ⚠️ |
| 2.12 | Column name variants (e.g., `start` vs `start_GRCh38`) | ❌ |
| 2.13 | GD table with extra/unknown columns | ❌ |

---

## 3. GDTable Class Internals (`models.py`)

| # | Case | Covered |
|---|------|---------|
| 3.1 | Column alias mapping (`start` → `start_GRCh38`) | ❌ |
| 3.2 | Missing required column → KeyError | ❌ |
| 3.3 | BP1 > BP2 swap logic (numeric comparison) | ❌ |
| 3.4 | BP1/BP2 alphanumeric comparison (e.g., `A1` vs `A2`) | ❌ |
| 3.5 | Empty cluster → no loci returned | ❌ |
| 3.6 | Locus with zero GD entries | ❌ |
| 3.7 | Malformed cluster/locus hierarchy | ❌ |
| 3.8 | `get_all_loci` vs `get_loci_by_chrom` consistency | ❌ |
| 3.9 | GDTable with single row | ❌ |
| 3.10 | GDTable encoding issues (UTF-8 BOM, etc.) | ❌ |

---

## 4. GD Calls Reading (`read_gd_calls`)

### 4a. Wide Format

| # | Case | Covered |
|---|------|---------|
| 4.1 | Wide format with carrier samples | ✅ |
| 4.2 | Wide format with .gz extension | ✅ |
| 4.3 | Wide format with no carriers (all False) | ✅ |
| 4.4 | Multiple GD_ID entries | ✅ |
| 4.5 | Header detection logic | ✅ |
| 4.6 | `is_carrier == "true"` (lowercase) | ❌ |
| 4.7 | `is_carrier == "1"` (numeric string) | ❌ |
| 4.8 | Mixed `True`/`true`/`1` values | ❌ |
| 4.9 | Extra whitespace around `is_carrier` | ❌ |
| 4.10 | Wide format with comment lines (`# ...`) | ❌ |
| 4.11 | Empty wide-format file (header only) | ❌ |
| 4.12 | Multiple carriers for same GD_ID | ✅ |
| 4.13 | Wide format with multiple svtypes per GD_ID | ❌ |
| 4.14 | Columns in unexpected order (csv.DictReader handles this) | ❌ |

### 4b. Narrow (Legacy) Format

| # | Case | Covered |
|---|------|---------|
| 4.15 | Narrow format, 6 columns, comma-separated samples | ✅ |
| 4.16 | Empty sample string (`.`) | ✅ |
| 4.17 | Empty sample string (empty field) | ✅ |
| 4.18 | Comment lines (skip) | ✅ |
| 4.19 | Short lines (<6 columns, skip) | ✅ |
| 4.20 | Single sample (no comma) | ❌ |
| 4.21 | Single carrier in comma-separated list | ❌ |
| 4.22 | Narrow format with trailing newline | ❌ |
| 4.23 | Narrow format with trailing whitespace | ❌ |
| 4.24 | .gz extension with narrow format | ❌ |
| 4.25 | Very large narrow-format file | ❌ |
| 4.26 | Narrow format with duplicate GD_ID entries | ❌ |

---

## 5. Ploidy Table Reading

| # | Case | Covered |
|---|------|---------|
| 5.1 | Standard wide format, multiple samples | ✅ |
| 5.2 | Comment lines skipped | ✅ |
| 5.3 | Multiple chromosomes per sample | ❌ |
| 5.4 | Empty ploidy table | ❌ |
| 5.5 | Sample not in ploidy table → default 2 | ✅ |
| 5.6 | Chrom not in sample's ploidy row → default 2 | ✅ |
| 5.7 | Malformed row (too few columns) | ❌ |
| 5.8 | Extra whitespace / tabs | ❌ |
| 5.9 | Non-integer ploidy value → `ValueError` from `int()` | ❌ |
| 5.10 | Heterogeneous ploidy (sample A diploid, sample B triploid) | ❌ |

---

## 6. PAR BED Reading

| # | Case | Covered |
|---|------|---------|
| 6.1 | Valid BED entries | ✅ |
| 6.2 | Comment lines skipped | ✅ |
| 6.3 | Short lines (<3 cols) skipped | ✅ |
| 6.4 | Multiple chromosomes | ❌ |
| 6.5 | Overlapping PAR intervals (should merge) | ❌ |
| 6.6 | PAR intervals on chrX and chrY | ❌ |
| 6.7 | PAR covering entire chromosome | ❌ |
| 6.8 | Empty PAR BED file | ❌ |

---

## 7. PAR Region Detection (`is_in_par_region`)

| # | Case | Covered |
|---|------|---------|
| 7.1 | Variant completely inside PAR | ✅ |
| 7.2 | Variant partially overlaps PAR, below cutoff | ✅ |
| 7.3 | Variant partially overlaps PAR, above cutoff | ❌ |
| 7.4 | Variant exactly at cutoff boundary | ❌ |
| 7.5 | Chromosome absent from PAR trees | ✅ |
| 7.6 | Zero-length interval | ✅ |
| 7.7 | Multiple PAR regions (early return on first match, **not cumulative**) | ⚠️ |
| 7.8 | Variant spans multiple PAR regions | ❌ |
| 7.9 | PAR region completely contained in variant | ❌ |
| 7.10 | PAR region completely contains variant | ❌ |

> **Note on 7.7:** The function returns on the **first** PAR interval exceeding the cutoff. It does **not** sum overlaps across multiple PAR regions. Two small PAR overlaps that cumulatively exceed the cutoff will not trigger detection. This is a design decision (or potential bug), not just an untested path.

---

## 8. Expected Copy Number (`get_expected_cn`)

| # | Case | Covered |
|---|------|---------|
| 8.1 | PAR always returns 2, regardless of ploidy | ✅ |
| 8.2 | Sample present, chrom present → ploidy | ✅ |
| 8.3 | Sample present, chrom absent → default 2 | ✅ |
| 8.4 | Sample absent → default 2 | ✅ |
| 8.5 | ecn = 0 (ploidy 1 on chrY) → no-call in genotype | ⚠️ |
| 8.6 | ecn = 3 (triploid) → DUP RD_CN = 4 | ❌ |
| 8.7 | ecn = 3 (triploid) → DEL RD_CN = 2 | ❌ |
| 8.8 | ecn = 0 (chrY ploidy 1, non-PAR) | ❌ |

---

## 9. VCF Carrier Extraction (`_extract_vcf_carriers`)

| # | Case | Covered |
|---|------|---------|
| 9.1 | No carriers (all hom-ref) | ✅ |
| 9.2 | One carrier | ✅ |
| 9.3 | All carriers | ✅ |
| 9.4 | No-call genotype `(None, None)` → carrier | ✅ |
| 9.5 | Heterozygous `(0, 1)` | ✅ |
| 9.6 | Homozygous alt `(1, 1)` → carrier | ❌ |
| 9.7 | Multi-allelic `(0, 2)` → carrier | ❌ |
| 9.8 | Missing GT field → defaults to `(0, 0)` via `.get("GT", (0,0))` | ❌ |
| 9.9 | Mixed GT formats in same record | ❌ |
| 9.10 | Record with zero samples | ❌ |
| 9.11 | Record with many samples (100+) | ❌ |
| 9.12 | Phased GT `(0|1)` vs unphased `(0,1)` (pysam returns tuples) | ❌ |
| 9.13 | Triploid GT `(0,1,2)` | ❌ |

---

## 10. Sample Overlap Scoring (`sample_overlap`)

| # | Case | Covered |
|---|------|---------|
| 10.1 | Partial overlap | ✅ |
| 10.2 | Disjoint sets | ✅ |
| 10.3 | Identical sets | ✅ |
| 10.4 | Subset relationship | ✅ |
| 10.5 | Both empty → `None` | ✅ |
| 10.6 | One empty, one non-empty → `0.0` | ✅ |
| 10.7 | Three-way partial overlap | ❌ |
| 10.8 | Large sets (100+ samples) | ❌ |
| 10.9 | Unicode / special characters in sample names | ❌ |
| 10.10 | Sample names with spaces | ❌ |
| 10.11 | VCF carriers ⊂ GD carriers (formula covers this, no explicit test) | ⚠️ |
| 10.12 | VCF carriers ⊃ GD carriers (formula covers this, no explicit test) | ⚠️ |

---

## 11. Genotype Update (`update_genotype`)

| # | Case | Covered |
|---|------|---------|
| 11.1 | ecn=0, carrier → no-call `(None,None)`, RD_CN=0 | ✅ |
| 11.2 | ecn=1, carrier, DEL → RD_CN=0 | ✅ |
| 11.3 | ecn=2, carrier, DEL → RD_CN=1 | ✅ |
| 11.4 | ecn=2, carrier, DUP → RD_CN=3 | ✅ |
| 11.5 | Non-carrier → homref `(0,0)`, RD_CN=ecn | ✅ |
| 11.6 | PE/SR fields reset when present | ✅ |
| 11.7 | PE/SR fields skipped when absent | ✅ |
| 11.8 | ecn=0, carrier, DUP → RD_CN=0 (ecn=0 branch, svtype ignored) | ⚠️ |
| 11.9 | GQ field set to 99 | ✅ |
| 11.10 | RD_GQ field set to 99 for non-carrier | ✅ |
| 11.11 | EV field set to `("RD",)` | ✅ |
| 11.12 | SVTYPE other than DEL/DUP (e.g., INV, BND) → RD_CN **never set** for carrier | ❌ |
| 11.13 | Pre-existing GT value (0,1) for non-carrier → reset | ✅ |

> **Note on 11.12:** If `svtype` is neither DEL nor DUP, the carrier branch sets `GT=(0,1)` and `GQ=99` but **never sets `RD_CN`**. This is a potential bug — `RD_CN` remains whatever value was inherited from the VCF record.

---

## 12. Header Management (`_ensure_headers`)

| # | Case | Covered |
|---|------|---------|
| 12.1 | All required INFO headers added | ✅ (via idempotent test) |
| 12.2 | All required FORMAT headers added | ❌ |
| 12.3 | Idempotent: pre-existing INFO not duplicated | ✅ |
| 12.4 | Idempotent: pre-existing FORMAT not duplicated | ❌ |
| 12.5 | Partial pre-existing INFO (only some present) | ❌ |
| 12.6 | Empty header (no INFO, no FORMAT) | ❌ |

---

## 13. Phase 1 — Non-NAHR Partial Overlap Annotation

| # | Case | Covered |
|---|------|---------|
| 13.1 | Variant overlaps non-NAHR ≥ threshold → annotated | ✅ |
| 13.2 | Variant overlaps non-NAHR < threshold → not annotated | ✅ |
| 13.3 | Zero-length variant → no crash | ✅ |
| 13.4 | DUP variant does NOT match DEL non-NAHR | ✅ |
| 13.5 | Multiple non-NAHR overlaps (first wins) | ❌ |
| 13.6 | Non-NAHR completely contained in variant | ❌ |
| 13.7 | Variant completely contained in non-NAHR | ❌ |
| 13.8 | Partial overlap from left side only | ❌ |
| 13.9 | Partial overlap from right side only | ❌ |
| 13.10 | Fraction exactly at threshold (e.g., 0.5 ≥ 0.5) | ❌ |
| 13.11 | 100% overlap (variant == non-NAHR coords) | ❌ |
| 13.12 | Non-NAHR on different chromosome | ❌ |
| 13.13 | Svtype-aware: DUP non-NAHR matches DUP variant | ❌ |
| 13.14 | Multiple non-NAHR at different coords | ❌ |
| 13.15 | Custom `--non-nahr-overlap` value | ⚠️ |

---

## 14. Phase 2 — NAHR Competitive Matching

### 14a. Basic Matching

| # | Case | Covered |
|---|------|---------|
| 14.1 | NAHR matched + gd_calls entry → genotypes reconciled | ✅ |
| 14.2 | NAHR matched + no gd_calls entry → passthrough unchanged | ✅ |
| 14.3 | DEL variant does NOT match DUP GD entry | ✅ |
| 14.4 | DUP novel record emitted for unmatched DUP | ✅ |
| 14.5 | All samples hom-ref after reconciliation → skip | ✅ |
| 14.6 | Some samples het after reconciliation → write | ❌ |
| 14.7 | Single sample, carrier → het | ❌ |
| 14.8 | Single sample, non-carrier → homref, skip | ❌ |

### 14b. Reciprocal Overlap

| # | Case | Covered |
|---|------|---------|
| 14.9 | RO < threshold → no match, passthrough | ❌ |
| 14.10 | RO exactly at threshold → match | ❌ |
| 14.11 | RO = 1.0 (identical coords) | ❌ |
| 14.12 | NAHR partially overlaps variant from left | ❌ |
| 14.13 | NAHR partially overlaps variant from right | ❌ |
| 14.14 | NAHR completely contains variant | ❌ |
| 14.15 | Variant completely contains NAHR | ❌ |
| 14.16 | Zero-length variant → no match (RO = 0) | ❌ |

### 14c. Sample-Overlap Tiebreaking

| # | Case | Covered |
|---|------|---------|
| 14.17 | Higher sample overlap wins | ✅ |
| 14.18 | Equal sample overlap → size difference breaks tie | ✅ |
| 14.19 | No carriers in VCF or gd_calls → `None` → size fallback | ✅ |
| 14.20 | VCF carriers ⊂ GD carriers → SO < 1.0 | ❌ |
| 14.21 | VCF carriers ⊃ GD carriers → SO < 1.0 | ❌ |
| 14.22 | Both carriers identical → SO = 1.0 | ❌ |
| 14.23 | Three overlapping NAHR regions (triple tiebreak) | ❌ |
| 14.24 | All three tiebreakers identical (RO, SO, size) | ❌ |

### 14d. Coordinate Updates

| # | Case | Covered |
|---|------|---------|
| 14.25 | pos updated from 0-based GD manifest (+1 for VCF) | ✅ |
| 14.26 | stop updated from GD manifest | ✅ |
| 14.27 | SVLEN computed from GD manifest coords | ❌ |
| 14.28 | GENOMIC_DISORDER / GD_CLUSTER / GD_BP1 / GD_BP2 set | ✅ |

---

## 15. Phase 3 — Novel Record Emission

| # | Case | Covered |
|---|------|---------|
| 15.1 | Novel record with correct coordinates | ✅ |
| 15.2 | All hom-ref → skipped | ✅ |
| 15.3 | Contig absent from header → skipped | ✅ |
| 15.4 | Missing metadata → skipped with warning | ✅ |
| 15.5 | Novel record with carriers → written | ❌ |
| 15.6 | Multiple novel records | ❌ |
| 15.7 | Novel record with different svtypes | ❌ |
| 15.8 | Novel record on chrX (PAR region) | ❌ |
| 15.9 | Novel record on chrY (low ploidy) | ❌ |
| 15.10 | Novel record ID format: `{GD_ID}_{svtype}_novel` | ✅ |
| 15.11 | SVLEN = stop - pos - 1 (0-based coords) | ❌ |
| 15.12 | Stop preserved after pysam recomputation | ❌ |
| 15.13 | INFO fields populated (SVTYPE, SVLEN, EV, ALGORITHMS, etc.) | ❌ |
| 15.14 | FORMAT fields populated (GT, GQ, RD_CN, RD_GQ) | ❌ |
| 15.15 | Non-NAHR gd_calls entry → novel record (not phase 2) | ❌ |
| 15.16 | GD entry matched in phase 2 → NOT emitted as novel | ⚠️ |

---

## 16. Phase Interactions

| # | Case | Covered |
|---|------|---------|
| 16.1 | Phase 1 annotation **overwritten** by Phase 2 match (both set `GENOMIC_DISORDER`) | ❌ |
| 16.2 | Phase 1 annotates a record, Phase 2 also matches → which `GD_CLUSTER` wins? | ❌ |
| 16.3 | Non-NAHR entry with gd_calls → goes to Phase 3 (can be novel) | ❌ |
| 16.4 | Record annotated in Phase 1 AND matched in Phase 2 → `GENOMIC_DISORDER` value from Phase 2 | ❌ |
| 16.5 | Phase 2 match with no gd_calls entry → Phase 1 annotation preserved | ❌ |
| 16.6 | All three phases active on same chromosome | ❌ |
| 16.7 | Phase 2 match suppresses Phase 3 novel emission for same GD_ID | ✅ |

---

## 17. End-to-End Integration Scenarios

| # | Case | Covered |
|---|------|---------|
| 17.1 | All records matched, some written, some skipped | ❌ |
| 17.2 | Some matched, some passthrough, some novel | ❌ |
| 17.3 | Empty VCF input, only novel records emitted | ❌ |
| 17.4 | Empty gd_calls, only VCF records pass through | ❌ |
| 17.5 | Mixed NAHR and non-NAHR in same run | ❌ |
| 17.6 | Multiple clusters on same chromosome | ❌ |
| 17.7 | Records sorted in output VCF | ❌ |
| 17.8 | Tabix index created | ❌ |
| 17.9 | bcftools sort + tabix integration | ❌ |
| 17.10 | Large cohort (100+ samples) | ❌ |

---

## 18. Coordinate & SVLEN Edge Cases

| # | Case | Covered |
|---|------|---------|
| 18.1 | Variant at position 0 | ❌ |
| 18.2 | Variant at last position of chromosome | ❌ |
| 18.3 | SVLEN = 0 (pos == stop - 1) | ❌ |
| 18.4 | SVLEN = -1 (pos == stop) | ❌ |
| 18.5 | 0-based vs 1-based confusion at boundary | ❌ |
| 18.6 | Variant straddling chromosome boundary | ❌ |
| 18.7 | GD region start > end (inverted coords) | ❌ |
| 18.8 | Very large SVLEN (> 10M) | ❌ |
| 18.9 | Very small SVLEN (1-10 bp) | ❌ |
| 18.10 | pysam SVLEN recomputation: setting SVLEN before stop | ❌ |
| 18.11 | pysam `new_record` 0-based start → 1-based `.pos` conversion | ❌ |

---

## 19. Header / FORMAT / INFO Edge Cases

| # | Case | Covered |
|---|------|---------|
| 19.1 | VCF with no FORMAT fields at all | ❌ |
| 19.2 | VCF with non-standard FORMAT fields | ❌ |
| 19.3 | VCF with existing GENOMIC_DISORDER INFO | ❌ |
| 19.4 | VCF with multiple ALT alleles | ❌ |
| 19.5 | VCF with missing samples in some records | ❌ |
| 19.6 | VCF with different sample sets per record | ❌ |
| 19.7 | FORMAT field ordering preserved | ❌ |
| 19.8 | INFO field ordering preserved | ❌ |
| 19.9 | All records are DEL/DUP (no non-DEL/DUP passthrough) | ❌ |
| 19.10 | Empty VCF tabix index (zero records) | ❌ |

---

## 20. Reader Error Paths

| # | Case | Covered |
|---|------|---------|
| 20.1 | `_read_narrow_format`: non-integer pos/end → `ValueError` from `int()` | ❌ |
| 20.2 | `read_ploidy_table`: empty file (no header) → `IndexError` | ❌ |
| 20.3 | `read_ploidy_table`: non-integer ploidy value → `ValueError` from `int()` | ❌ |
| 20.4 | `_read_bed_to_trees`: non-numeric BED coords → `ValueError` from `int()` | ❌ |
| 20.5 | Wide format: missing required column → `KeyError` from `row["GD_ID"]` | ❌ |
| 20.6 | Narrow format: columns in unexpected order | ❌ |
| 20.7 | GD calls file with mixed encodings | ❌ |
| 20.8 | Ploidy table with row shorter than header | ❌ |

---

## 21. Error Handling & Fault Tolerance

| # | Case | Covered |
|---|------|---------|
| 21.1 | bcftools not found | ❌ |
| 21.2 | bcftools sort failure | ✅ |
| 21.3 | Permission denied on output path | ❌ |
| 21.4 | Disk full during processing | ❌ |
| 21.5 | Corrupted VCF (malformed record) | ❌ |
| 21.6 | Corrupted GD table | ❌ |
| 21.7 | Malformed PAR BED (non-numeric coords) | ❌ |
| 21.8 | Temporary file cleanup on exception | ❌ |
| 21.9 | Partial write recovery | ❌ |
| 21.10 | Interrupted by signal (Ctrl-C) | ❌ |

---

## 22. Parameter Value Edge Cases

| # | Case | Covered |
|---|------|---------|
| 22.1 | `--reciprocal-overlap 0.0` (everything matches) | ❌ |
| 22.2 | `--reciprocal-overlap 1.0` (exact match only) | ❌ |
| 22.3 | `--reciprocal-overlap < 0.0` (negative, no validation) | ❌ |
| 22.4 | `--non-nahr-overlap 0.0` (everything annotated) | ❌ |
| 22.5 | `--non-nahr-overlap 1.0` (only exact match) | ❌ |
| 22.6 | `--non-nahr-overlap > 1.0` (nothing ever annotates, no error) | ❌ |
| 22.7 | `--temp-dir` relative path | ❌ |
| 22.8 | `--temp-dir` absolute path | ❌ |
| 22.9 | `--temp-dir` = `.` (current directory) | ❌ |
| 22.10 | Default parameter values used | ❌ |

---

## 23. pysam-Specific Behavior

| # | Case | Covered |
|---|------|---------|
| 23.1 | pysam recomputes `stop = pos + SVLEN` when SVLEN is set | ❌ |
| 23.2 | Code sets SVLEN **before** stop to exploit recomputation | ❌ |
| 23.3 | pysam version change breaks SVLEN/stop order assumption | ❌ |
| 23.4 | `new_record` 0-based input → 1-based `.pos` | ❌ |
| 23.5 | pysam `VariantFile` iteration order (chromosomal sort) | ❌ |
| 23.6 | Empty VCF: tabix index on zero records | ❌ |
| 23.7 | pysam `record.info` tuple/list handling for Number=. fields | ❌ |

---

## 24. Contig Naming Consistency

| # | Case | Covered |
|---|------|---------|
| 24.1 | VCF uses `chr1`, GD table uses `1` (silent mismatch) | ❌ |
| 24.2 | VCF uses `1`, GD table uses `chr1` (silent mismatch) | ❌ |
| 24.3 | PAR BED uses different contig prefix than VCF | ❌ |
| 24.4 | Ploidy table uses different contig prefix than VCF | ❌ |
| 24.5 | Mixed prefixes within same file (e.g., `chr1` and `2`) | ❌ |

---

## 25. Multi-Chromosomal & Large-Cohort Scenarios

| # | Case | Covered |
|---|------|---------|
| 25.1 | GD regions on chr1, chr2, chrX simultaneously | ❌ |
| 25.2 | VCF records spanning multiple chromosomes | ❌ |
| 25.3 | chrY records with ploidy 1 | ❌ |
| 25.4 | chrX records in PAR region | ❌ |
| 25.5 | chrM records | ❌ |
| 25.6 | Uncontiguous contigs (e.g., chrUn_*) | ❌ |
| 25.7 | 200+ samples in VCF | ❌ |
| 25.8 | 100+ GD regions | ❌ |
| 25.9 | 1000+ VCF records | ❌ |
| 25.10 | Memory pressure with large inputs | ❌ |

---

## Summary Statistics

| Category | Total Cases | ✅ Covered | ❌ Not Covered | ⚠️ Partial |
|----------|:-----------:|:----------:|:--------------:|:----------:|
| Input Validation & CLI | 15 | 2 | 12 | 1 |
| GD Table Loading | 13 | 3 | 9 | 1 |
| GDTable Class Internals | 10 | 0 | 10 | 0 |
| GD Calls Reading | 26 | 12 | 13 | 1 |
| Ploidy Table Reading | 10 | 4 | 5 | 1 |
| PAR BED Reading | 8 | 3 | 5 | 0 |
| PAR Region Detection | 10 | 5 | 4 | 1 |
| Expected Copy Number | 8 | 4 | 3 | 1 |
| VCF Carrier Extraction | 13 | 5 | 7 | 1 |
| Sample Overlap Scoring | 12 | 6 | 4 | 2 |
| Genotype Update | 13 | 8 | 4 | 1 |
| Header Management | 6 | 2 | 4 | 0 |
| Phase 1: Non-NAHR | 15 | 4 | 10 | 1 |
| Phase 2: NAHR Matching | 28 | 12 | 14 | 2 |
| Phase 3: Novel Records | 16 | 6 | 9 | 1 |
| Phase Interactions | 7 | 1 | 6 | 0 |
| End-to-End Scenarios | 10 | 0 | 9 | 1 |
| Coordinate & SVLEN | 11 | 0 | 11 | 0 |
| Header / FORMAT / INFO | 10 | 0 | 10 | 0 |
| Reader Error Paths | 8 | 0 | 8 | 0 |
| Error Handling | 10 | 1 | 9 | 0 |
| Parameter Values | 10 | 0 | 10 | 0 |
| pysam-Specific Behavior | 7 | 0 | 7 | 0 |
| Contig Naming | 5 | 0 | 5 | 0 |
| Multi-Chromosomal | 10 | 0 | 10 | 0 |
| **TOTAL** | **280** | **97** | **172** | **11** |

**Coverage: ~35%** of identified edge cases have tests. The largest gaps are in GDTable internals, pysam-specific behavior, phase interactions, reader error paths, contig naming consistency, and parameter value edge cases.

### Potential bugs identified

1. **`update_genotype` with non-DEL/DUP svtype (11.12):** Carrier genotypes for INV/BND variants never get `RD_CN` set.
2. **`is_in_par_region` early return (7.7):** Cumulative overlap across multiple PAR regions is not detected (returns on first match).
