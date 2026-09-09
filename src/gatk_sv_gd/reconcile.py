"""Reconcile the affected GD intervals per sample before regrouping VCF sites."""

from collections import defaultdict


def affected_records(records, index):
    """Select direct GD overlaps without following chains of VCF records."""
    for record in records:
        loci = set()
        if (_values(record.info.get("SVTYPE")) in (("DEL",), ("DUP",))
                and record.start < record.stop and record.chrom in index):
            loci = {iv.data[0] for iv in index[record.chrom].overlap(record.start, record.stop)}
        yield record, loci


def _values(value):
    return (value,) if isinstance(value, str) else tuple(value or ())


def _called(gt):
    return any(allele is not None and allele > 0 for allele in (gt.get("GT") or ()))


def _canonical(record, meta):
    """Both breakpoints must lie within the corresponding GD breakpoint ranges."""
    left = meta.get("start_range", (meta.get("start"),) * 2)
    right = meta.get("end_range", (meta.get("end"),) * 2)
    return (left[0] is not None and right[0] is not None
            and left[0] <= record.start <= left[1] and right[0] <= record.stop <= right[1])


def _segments(events):
    """Collapse redundant GD detections; contradictory model states are errors."""
    changes = defaultdict(lambda: [[], []])
    for index, event in enumerate(events):
        changes[event["start"]][0].append(index)
        changes[event["end"]][1].append(index)
    active = set()
    pieces = []
    positions = sorted(changes)
    for start, end in zip(positions, positions[1:]):
        added, removed = changes[start]
        active.difference_update(removed)
        active.update(added)
        if not active:
            continue
        covering = [events[index] for index in sorted(active)]
        if len({event["state"] for event in covering}) > 1:
            raise ValueError("Conflicting GD copy states overlap in one sample; resolve the calls before integration")
        # Never add copy states from redundant model detections.
        winner = min(covering, key=lambda event: (
            -(event["gt"].get("GQ") or 0), event["start"], event["end"], event["label"],
            event["state"], str(event["gt"]),
        ))
        piece = dict(winner, start=start, end=end)
        piece["gd_ids"] = set().union(*(event["gd_ids"] for event in covering))
        piece["loci"] = set().union(*(event["loci"] for event in covering))
        piece["sources"] = set().union(*(event["sources"] for event in covering))
        piece["algorithms"] = set().union(*(event["algorithms"] for event in covering))
        piece["tokens"] = set(active)
        piece["qualities"] = [event["gt"] for event in covering]
        # Overlapping same-state GD calls share one event. Merely touching
        # independent GD calls retain their own boundaries.
        if (pieces and pieces[-1]["end"] == start and pieces[-1]["state"] == piece["state"]
                and pieces[-1]["svtype"] == piece["svtype"]
                and pieces[-1]["gt"]["GT"] == piece["gt"]["GT"]
                and pieces[-1]["tokens"] & piece["tokens"]):
            previous = pieces[-1]
            previous["end"] = end
            for key in ("gd_ids", "loci", "sources", "tokens", "algorithms"):
                previous[key].update(piece[key])
            previous["qualities"].extend(piece["qualities"])
            previous["gt"] = piece["gt"]
        else:
            pieces.append(piece)
    return pieces


def reconcile_records(header, gd_calls, metadata, affected, ploidy, par_trees, overlap_cutoff):
    """Replace sufficiently overlapping VCF genotypes with positive GD calls.

    Matching is per sample, independent of VCF copy state and DEL/DUP type.
    A matched VCF call is removed entirely; no flanks or inferred segments are
    retained. Unmatched original calls keep their coordinates and annotations.
    Canonical events remain subject to complete-cohort GD reevaluation.
    """
    from gatk_sv_gd import integrate

    buckets = defaultdict(list)
    expected = []
    matched = set()
    for (gd_id, svtype), info in sorted(gd_calls.items()):
        if not metadata.get(gd_id, {}).get("nahr", True):
            continue
        for sample in sorted(info["samples"]):
            if sample not in header.samples:
                continue
            events = info.get("carrier_calls", {}).get(sample)
            if events is None:  # Legacy narrow input has one interval per GD ID.
                events = [dict(info, cn_state=info.get("copy_states", {}).get(sample),
                               cn_probabilities=info.get("copy_probabilities", {}).get(sample))]
            for event in events:
                chrom, start, end = event["chrom"], event["pos"], event["end"]
                if start < 0 or start >= end:
                    raise ValueError(f"Invalid interval for detected GD call {gd_id}/{svtype}")
                gt = {}
                baseline = integrate.get_expected_cn(chrom, start, end, sample, ploidy, par_trees)
                integrate.update_genotype(
                    gt, sample, True, ploidy.get(sample, {}).get(chrom, 2), svtype,
                    cn_state=event.get("cn_state"), baseline_cn=baseline,
                    cn_probabilities=event.get("cn_probabilities"),
                )
                if not _called(gt):
                    raise ValueError(f"Detected GD call {gd_id}/{svtype} has zero sample ploidy")
                buckets[(chrom, sample)].append({
                    "start": start, "end": end, "gt": gt, "state": gt["RD_CN"],
                    "svtype": svtype, "algorithms": {"depth"},
                    "gd_ids": {gd_id}, "loci": {gd_id}, "sources": set(), "label": gd_id,
                })
                expected.append((chrom, svtype, sample, start, end, gd_id, gt["RD_CN"]))

    records = []
    for original, loci in affected:
        svtype = _values(original.info["SVTYPE"])[0]
        canonical_ids = {
            gd_id for gd_id in loci
            if (gd_id, svtype) in gd_calls
            and metadata.get(gd_id, {}).get("chrom", original.chrom) == original.chrom
            and _canonical(original, metadata.get(gd_id, {}))
            and integrate.reciprocal_overlap(original.start, original.stop,
                                             metadata[gd_id]["start"], metadata[gd_id]["end"]) >= overlap_cutoff
        }
        matched.update((gd_id, svtype) for gd_id in canonical_ids)
        changed = False
        replaced = bool(canonical_ids)
        record = original.copy()
        for sample, genotype in record.samples.items():
            matches = [
                event for event in buckets.get((record.chrom, sample), ())
                if event["start"] < record.stop and record.start < event["end"]
                and integrate.reciprocal_overlap(record.start, record.stop,
                                                event["start"], event["end"]) >= overlap_cutoff
            ]
            replaced = replaced or bool(matches)
            for event in matches:
                matched.update((gd_id, event["svtype"]) for gd_id in event["gd_ids"])
                if _called(genotype) and original.id:
                    event["sources"].add(original.id)
            if not _called(genotype) or not (matches or canonical_ids):
                continue
            changed = True
            # The original event is absent in this sample. Do not transfer
            # its breakpoint evidence or qualities to the replacement GD call.
            for field in list(genotype):
                if field != "GT":
                    genotype[field] = None
            if hasattr(genotype, "phased"):
                genotype.phased = False
            integrate.update_genotype(
                genotype, sample, False, ploidy.get(sample, {}).get(record.chrom, 2), svtype,
                baseline_cn=integrate.get_expected_cn(record.chrom, record.start, record.stop,
                                                      sample, ploidy, par_trees),
            )
        if replaced and integrate.all_homref_record(record.samples.values()):
            continue
        if changed:
            _update_counts(record, header)
            records.append(record)
        else:
            # Incidental overlap alone cannot erase zero-carrier or missing
            # records, or change annotations on an untouched event.
            records.append(original)

    sites = defaultdict(dict)
    sample_segments = {}
    for (chrom, sample), events in sorted(buckets.items()):
        pieces = _segments(events)
        sample_segments[(chrom, sample)] = pieces
        for piece in pieces:
            sites[(chrom, piece["start"], piece["end"], piece["svtype"])][sample] = piece

    # Audit coverage of every accepted positive GD call, including its copy
    # state and identity. A serialization-ready result must not lose a call.
    for chrom, svtype, sample, start, end, gd_id, cn in expected:
        covered = sum(
            max(0, min(end, piece["end"]) - max(start, piece["start"]))
            for piece in sample_segments[(chrom, sample)]
            if gd_id in piece["gd_ids"] and piece["state"] == cn and piece["svtype"] == svtype
        )
        if covered != end - start:
            raise ValueError(f"Reconciliation lost coverage of GD call {gd_id}/{svtype}")

    used_ids = {record.id for record in records}
    for (chrom, start, end, svtype), carriers in sorted(sites.items()):
        gd_ids = set().union(*(piece["gd_ids"] for piece in carriers.values()))
        loci = set().union(*(piece["loci"] for piece in carriers.values()))
        sources = set().union(*(piece["sources"] for piece in carriers.values()))
        gd_id = min(gd_ids or loci)
        meta = metadata.get(gd_id, {"cluster": gd_id})
        atypical = (chrom, start, end) != (meta.get("chrom", chrom), meta.get("start"), meta.get("end"))
        new_meta = dict(meta)
        if atypical:
            new_meta.pop("bp1", None)
            new_meta.pop("bp2", None)
        # Qualities for reference samples apply only to unchanged GD geometry.
        probabilities = {}
        info = gd_calls.get((gd_id, svtype), {})
        if info.get("intervals", {(info.get("chrom"), info.get("pos"), info.get("end"))}) == {(chrom, start, end)}:
            probabilities = info.get("copy_probabilities", {})
        record = integrate._build_gd_record(
            header, chrom, start, end, gd_id, svtype, new_meta, set(), ploidy, par_trees,
            is_novel=(gd_id, svtype) not in matched, copy_probabilities=probabilities,
        )
        if atypical:
            record.id = f"{gd_id}_{svtype}_{chrom}_{start}_{end}_atypical"
            record.info["GD_ATYPICAL"] = True
        if record.id in used_ids:
            record.id = f"{record.id}_{svtype}_{chrom}_{start}_{end}"
        used_ids.add(record.id)
        algorithms = set().union(*(piece["algorithms"] for piece in carriers.values()))
        if algorithms:
            record.info["ALGORITHMS"] = tuple(sorted(algorithms))
        elif "ALGORITHMS" in record.info:
            del record.info["ALGORITHMS"]
        evidence = set().union(*(set(_values(piece["gt"].get("EV"))) for piece in carriers.values()))
        if evidence:
            record.info["EV"] = tuple(sorted(evidence))
        elif "EV" in record.info:
            del record.info["EV"]
        if gd_ids:
            record.info["GD_CALL_IDS"] = tuple(sorted(gd_ids))
        if sources:
            record.info["GD_SOURCE_IDS"] = tuple(sorted(sources))
        for sample, piece in carriers.items():
            for field, value in piece["gt"].items():
                record.samples[sample][field] = value
            for field in ("GQ", "RD_GQ"):
                qualities = [gt.get(field) for gt in piece["qualities"]]
                record.samples[sample][field] = None if None in qualities else min(qualities)
        _update_counts(record, header)
        records.append(record)
    return records


def _update_counts(record, header):
    """Refresh allele counts after replacing sample genotypes."""
    alleles = [a for gt in record.samples.values() for a in (gt.get("GT") or ())]
    ac = tuple(alleles.count(a) for a in range(1, max(1, len(record.alts or ())) + 1))
    an = sum(a is not None for a in alleles)
    if "AC" in header.info:
        record.info["AC"] = ac
    if "AN" in header.info:
        record.info["AN"] = an
    if "AF" in header.info:
        record.info["AF"] = tuple(n / an if an else 0.0 for n in ac)
