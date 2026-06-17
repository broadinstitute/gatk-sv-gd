# Integration Plan: streaming, non-NAHR annotation & GD-ID robustness in `integrate.py`

**File:** `src/gatk_sv_gd/integrate.py`
**Tests:** `tests/test_integrate.py`
**Status as of:** 2026-06-17 (commit `c76e050`)

> This plan supersedes the older `PLAN.md` in the companion repo
> (`/Users/markw/Work/mw_gd_external/ref_panel_1kgp/PLAN.md`). That plan was
> written **before** the GD-call-centric refactor
> (commits `c7ad2fd`, `453291c`) and several of its items are now obsolete or
> implemented by a different mechanism. The reconciliation below reflects the
> code as it actually stands.

---

## Current architecture (verified)

`main()` runs three phases:

1. **Phase 1 — buffer + in-place non-NAHR annotation** (`integrate.py:705-749`)
   Reads every VCF record into `vcf_records: List[(record, meta)]`. For each
   DEL/DUP record overlapping a non-NAHR locus by `>= --non-nahr-overlap`
   (default `0.02`), it sets `GENOMIC_DISORDER` / `GD_CLUSTER` INFO **in place**.
   Records are **not** dropped and carry no authoritative per-sample GD calls.

2. **Phase 2 — GD-call-centric emission** (`integrate.py:761-840`)
   Iterates `gd_calls` (parsed from the `--gd-calls` wide TSV). For each
   `(gd_id, svtype)` with carriers:
   - `_find_overlapping_vcf_records(...)` (`integrate.py:548`) returns matching
     buffered records by SVTYPE + reciprocal overlap `>= --reciprocal-overlap`
     (default `0.5`).
   - Matched originals are marked for replacement.
   - `_build_gd_record(..., is_novel=False/True)` (`integrate.py:589`) emits one
     authoritative record at **GD-table coordinates**, carriers → het, others →
     hom-ref (ploidy-aware via `get_expected_cn` / `update_genotype`).
   - All-hom-ref records are skipped.
   - **`gd_id not in gd_metadata` → `logger.warning(...); continue`**
     (`integrate.py:785-790`) — the call is silently dropped.

3. **Phase 3 — write** survivors + new GD records, then bgzip/sort/tabix via
   `bcftools`.

Carriers in the current design come **only** from the gd_calls TSV
(`gd_info["samples"]`), never derived from existing VCF genotypes.

---

## Scaling constraint — NO in-memory record buffer

**Target: 250K samples × ~5M records. Holding all records in memory is not an
option.** The current `main()` builds a full `vcf_records` list (every record)
plus a `replaced_vcf` index set — this must be removed. The redesign below is a
prerequisite for items A–C, not an afterthought; implement it first.

### What makes streaming possible (verified)

- **Output is `bcftools sort`-ed at the end** (`_sort_vcf`, `integrate.py:470`),
  so emission order is free — records and GD records may be written in any order.
- **The GD axis is small.** GD calls + non-NAHR loci are bounded by the GD table
  (~10²–10³). Only the *record* axis is large. Per-GD-entry state fits in memory;
  per-record state must not persist.
- **`main()` already streams** via `for record in pysam.VariantFile(args.vcf)`
  (`integrate.py:697,717`); it only buffers because Phase 2 runs after the loop.
- **The drop/keep decision for any record is recomputable per-record** from the
  GD interval trees alone — it needs no cross-record aggregation (see the
  non-NAHR model below). So a **single pass** suffices and no per-record state
  ever persists.
- **Test surface is safe:** `vcf_records=` is a kwarg of the test helper
  `_run_integrate_main`, which feeds records through a fake `pysam.VariantFile`.
  The refactor lives entirely inside `main()` + helpers; the ~100
  `vcf_records=[...]` call sites do not change.

### Design: single-pass streaming, zero record buffer

The two paths have different mechanics, but neither needs the buffer:
- **NAHR (gd_calls):** replace — drop overlapping records, emit one record per
  GD call. The drop test (SVTYPE + RO ≥ `--reciprocal-overlap`) is decidable
  per-record from a small `gd_call_index`; emission happens after the stream.
- **Non-NAHR (table, `NAHR=no`):** annotate-only — tag every overlapping record
  in place, drop nothing, synthesize nothing (see "Non-NAHR model"). Pure
  per-record operation.

**The pass — stream once, write or drop each record immediately:**
- Build `gd_call_index`: `chrom -> IntervalTree` of `(gd_id, svtype, start, stop)`
  from the gd_calls TSV. (Replaces the buffer-scanning
  `_find_overlapping_vcf_records`.)
- For each streamed record:
  - **NAHR drop:** DEL/DUP matching a `gd_call_index` entry (SVTYPE + RO ≥
    cutoff) → **do not write**; add `(gd_id, svtype)` to a `matched_calls` set.
  - **Non-NAHR annotate:** else, if it overlaps `non_nahr_trees` by
    `fraction_covered >= --non-nahr-overlap` → set `GENOMIC_DISORDER` /
    `GD_CLUSTER` in place (any and all overlapping records), then write.
  - Otherwise → write unchanged.
  - Write goes to a temp "passthrough" VCF — still coordinate-sorted because the
    input is sorted and order is preserved.
- After the stream: for each gd_call, append one `_build_gd_record(is_novel =
  (gd_id, svtype) not in matched_calls)` (carriers from the TSV) to a separate
  small GD-records file.
- Merge: `bcftools sort` the small GD-records file, then `bcftools concat -a`
  (allow-overlaps positional merge) with the already-sorted passthrough → final.
  (See JC3 — this beats re-sorting all 5M.)

Memory is O(GD entries + `matched_calls`); nothing scales with record count. No
GT parsing is needed for the drop/annotate decision (coordinate-only); GTs are
read only when `_build_gd_record` constructs NAHR records.

### Notes

- **pysam pointer safety:** write-or-drop per record never holds a stale C
  pointer, unlike buffering 5M `VariantRecord`s — streaming is also safer here.
- **Single pass works** because non-NAHR is annotate-only; there is no
  cross-record carrier aggregation that would force a barrier or a second pass.

---

## Reconciliation with old `PLAN.md`

| Old item | Intent | Status | Evidence |
|---|---|---|---|
| 1 | Remove in-place Phase 1 annotation | **REJECTED** — keep it; non-NAHR is annotate-only (`integrate.py:733-747`) |
| 2 | Add `_carriers_from_vcf_record` | **REJECTED** — no synthesized non-NAHR record; existing GTs are the calls |
| 3 | New Phase 1 carrier-calling block | **REJECTED** — non-NAHR does not drop/replace; see Non-NAHR model |
| 4 | Phase 2 GD-ID fallback instead of `continue` | **OPEN** (item A) | `integrate.py:785-790` |
| 5 | Guard empty `bp1`/`bp2` in `_build_gd_record` | **OPEN** (item B) | `integrate.py:637-640` set unconditionally |

> Old plan items 1–3 assumed non-NAHR loci should be collapsed into one
> synthesized "authoritative" record with derived carriers, dropping the
> originals. That is **wrong** for non-NAHR (polymorphic, multi-location — see
> below). The real non-NAHR work is just to preserve in-place annotation through
> the streaming refactor. Only items A, B, and the streaming refactor are live.

---

## Non-NAHR model — annotate-only (the key correction)

**`infer`/`call` emit nothing for non-NAHR sites — integration is the only stage
that handles them** (confirmed 2026-06-17). **Identify non-NAHR sites by their
NAHR status, never by presence/absence in gd_calls.** The GD table is the source
of truth: `_build_trees_from_gd_table` splits entries into `nahr_trees` /
`non_nahr_trees` on the `NAHR` flag (`models.py:227`, `is_nahr`). Integration
must handle **every** non-NAHR site in the table, on every run.

**Non-NAHR events are polymorphic and can occur at multiple locations within a
single GD locus.** There is therefore no single "authoritative" record to
synthesize and nothing to collapse or drop. The correct behavior is:

> Annotate **any and all** VCF records overlapping a non-NAHR region (by
> `fraction_covered >= --non-nahr-overlap`) with `GENOMIC_DISORDER` / `GD_CLUSTER`,
> in place. Keep every such record. Do not derive carriers into a new record, do
> not drop originals, do not emit a synthesized record. The existing per-sample
> genotypes on those records ARE the non-NAHR calls.

This is essentially what the current in-place Phase 1 annotation already does
(`integrate.py:733-747`). The non-NAHR work reduces to **preserving that
annotation through the streaming refactor** — no new helper, no record
replacement. (This corrects old plan items 1–3, which wrongly proposed
collapse-and-drop.)

The two paths are disjoint by construction:
- **NAHR sites** (`nahr_trees`, gd_calls TSV) → replace: drop overlapping records
  by reciprocal overlap, emit one record per GD call with TSV carriers.
- **Non-NAHR sites** (`non_nahr_trees`, GD table) → annotate-only, as above.

No gd_calls cross-check on the non-NAHR path — non-NAHR rows are not expected in
the calls TSV.

---

## Work items

### A. Phase 2 GD-ID fallback (was old item 4) — **independent, do first**

Replace the `continue` at `integrate.py:785-790`. When a gd_calls entry's
`gd_id` is absent from `gd_metadata` (calls produced against an older GD-table
version), synthesize fallback metadata instead of dropping the call:

```python
meta = gd_metadata.get(gd_id) or {
    "cluster": gd_id, "bp1": "", "bp2": "", "nahr": True, "svtype": svtype,
}
```

The call is then emitted using the old ID and the coordinates already present in
the gd_calls TSV row. Keep the `logger.warning`.

### B. Guard empty `bp1`/`bp2` in `_build_gd_record` (was old item 5)

At `integrate.py:637-640`, only set the INFO fields when non-empty (fallback
metadata from item A uses `""`):

```python
if meta.get("bp1"):
    new_rec.info["GD_BP1"] = meta["bp1"]
if meta.get("bp2"):
    new_rec.info["GD_BP2"] = meta["bp2"]
```

### C. Non-NAHR annotation — preserve through the streaming refactor

Non-NAHR is **annotate-only** (see Non-NAHR model). No carrier derivation, no
synthesized record, no dropping. The work is to carry the existing in-place
annotation into the single-pass design:

1. Drive it off `non_nahr_trees` (every non-NAHR site in the GD table, every run).
2. During the stream, for each record that is **not** a NAHR drop and overlaps a
   non-NAHR region by `fraction_covered >= --non-nahr-overlap`, set
   `GENOMIC_DISORDER` / `GD_CLUSTER` in place and write it. Annotate **any and
   all** overlapping records (a polymorphic locus may have several).
3. No `_carriers_from_vcf_record`, no `_build_gd_record` call for non-NAHR. This
   is largely the existing `integrate.py:733-747` block, relocated into the
   single streaming pass.

Carrier-less and carrier-bearing records are treated identically: both get the
annotation, both are kept (resolves JC1 — always annotate). The only ordering
rule: a record that is a NAHR drop is removed and never reaches annotation.

---

## Resolved decisions (were open judgment calls)

1. **Carrier-less non-NAHR site → still annotate as GD.** Every record
   overlapping a non-NAHR region gets `GENOMIC_DISORDER` / `GD_CLUSTER`
   regardless of carrier status. (Folded into item C.)
2. **Non-NAHR overlapping records → annotate, never drop.** Non-NAHR events are
   polymorphic and occur at multiple locations within a locus; any and all
   overlapping records are annotated and kept. There is no synthesized
   "authoritative" record for non-NAHR. (Dropping/replacement is the NAHR/gd_calls
   path only.)
3. **Sort GD records independently, then merge.** Stream the passthrough (already
   coordinate-sorted) to one temp file; write GD (NAHR) records to a small
   separate file; `bcftools sort` only that small file; `bcftools concat -a`
   (allow-overlaps positional merge) the two → final output. Avoids re-sorting
   all 5M records.

No open decisions remain.

---

## Tests to add / update

> Old `PLAN.md` referenced `TestMainNonNahrAnnotation`,
> `TestMainNonNahrCarrierCalling`, `TestMainGdIdFallback` — **none exist**.
> Current relevant classes: `TestPhase3NovelRecords`, `TestPhaseInteractions`,
> `TestMainNovelRecord`, `TestBuildTreesFromGdTable`.

| For item | New test | Verifies |
|---|---|---|
| streaming | `TestStreaming::test_single_pass_matches_buffered` | same output VCF as the pre-refactor buffered path on a fixture |
| streaming | `TestStreaming::test_no_buffer_growth` | peak record-holding state independent of record count (e.g. via the existing large-N fixture) |
| A | `TestPhaseInteractions::test_missing_gd_id_uses_fallback` | gd_id absent from table → record emitted with `GENOMIC_DISORDER == old_id`, `GD_CLUSTER == old_id`, coords from TSV |
| B | `TestBuildGdRecord::test_empty_bp_not_set` | fallback meta (`bp1=bp2=""`) → `GD_BP1`/`GD_BP2` absent from INFO |
| C | `TestMainNonNahrAnnotation::test_all_overlapping_records_annotated` | several records overlapping one non-NAHR locus → **all** get `GENOMIC_DISORDER`/`GD_CLUSTER`, **none** dropped |
| C | `...::test_homref_overlap_still_annotated` | all-hom-ref overlapping record → still annotated, still kept (JC1) |
| C | `...::test_nahr_drop_takes_precedence` | record matching a NAHR gd_call AND grazing a non-NAHR region → dropped (not annotated+kept) |

Existing tests should pass unchanged for items A & B. The streaming refactor
must preserve all existing output assertions. Run the full suite (`pytest`, 90%
coverage gate) after each item.

---

## Handoff TODO

- [x] **T0 — Open Question resolved:** `infer`/`call` emit nothing for non-NAHR
      sites; integration is the only stage that handles them. Item C is in scope.
- [ ] **T1 — Streaming refactor (do FIRST; prerequisite for all else).**
      Replace the `vcf_records` buffer + `replaced_vcf` set with the single-pass
      design (Scaling section): stream once, drop NAHR matches, annotate non-NAHR
      overlaps in place, write the rest to a passthrough temp; append NAHR GD
      records to a small separate file; `bcftools sort` that file + `concat -a`
      with the passthrough. Delete `_find_overlapping_vcf_records`; add
      `gd_call_index`. Confirm the full existing suite still passes. Add
      `TestStreaming::test_single_pass_matches_buffered` + a large-N case
      alongside the existing `test_1000_vcf_records`
      (`tests/test_integrate.py:7266`).
- [ ] **T2 — Item A:** GD-ID fallback at `integrate.py:785-790`. Add
      `test_missing_gd_id_uses_fallback`.
- [ ] **T3 — Item B:** guard empty `bp1`/`bp2` at `integrate.py:637-640`. Add
      `test_empty_bp_not_set`. (Do with/after T2 — B is what makes A's `""` safe.)
- [ ] **T4 — Item C:** preserve non-NAHR in-place annotation in the single pass
      (annotate any/all overlapping records, drop none). Relocate the
      `integrate.py:733-747` block. Add the three `TestMainNonNahrAnnotation`
      tests. Builds on T1. (No `_carriers_from_vcf_record` — non-NAHR is
      annotate-only.)
- [ ] **T5 — Full gate:** `pytest` (90% coverage) + `ruff check src/`. Sanity-check
      peak RSS stays flat as record count grows (no buffer regressions).
- [ ] **T6 — Update docstrings:** module header (`integrate.py:10`) and CLAUDE.md
      integrate row to reflect single-pass streaming + annotate-only non-NAHR.
