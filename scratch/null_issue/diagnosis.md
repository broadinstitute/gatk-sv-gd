# Synthetic null-heavy posterior diagnosis

Synthetic setup:
- Haploid chrX locus with two deleted body bins and two flanking non-event bins.
- Among informative pair states, body bins are 90% DEL and flank bins are 90% non-DEL.
- The body_null_only scenario sets body prob_null=0.90 and flank prob_null=0.00.

Key result:
- High null in the deleted body alone is sufficient to collapse the old final QUAL.
- Clean flanks do not rescue the old score because final qual_score is limited by the minimum interval/flank confidence component.
- The fixed score keeps log_prob_score unchanged but computes body/flank confidence from informative support only.

Analytic body-bin comparison for this setup:
- Old DEL probability fed into body QUAL = 0.9 * (1 - body_null) + 0.5 * body_null = 0.9 - 0.4 * body_null.
- New DEL support fed into body QUAL = 0.9, independent of body_null.
- At body_null = 0.90, old body input is 0.54 while new body input remains 0.90.

Representative numbers for body_null_only:
- Old min_interval_confidence = 0.696
- New min_interval_confidence = 9.542
- Old min_flank_non_event_confidence = 9.542
- New min_flank_non_event_confidence = 9.542
- Old qual_score = 0.696
- New qual_score = 9.542
- Old log_prob_score = 0.540
- New log_prob_score = 0.540

Interpretation:
- The old bug was not that the event summary was wrong; log_prob_score stays the same before and after the fix.
- The bug was that null-neutralized event probabilities were reused for called-state confidence.
- In a null-heavy hemizygous deletion, that drags body-bin QUAL toward zero even when the informative posterior strongly favors deletion.

Per-bin body_null_only details:

| scenario | bin_index | bin_label | expected_state | null_probability | informative_mass | del_informative_mass | non_del_informative_mass | old_event_probability_used_for_qual | new_informative_support_used_for_qual | old_expected_state_qual | new_expected_state_qual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| body_null_only | 0 | body_A-B | DEL | 0.900 | 0.100 | 0.090 | 0.010 | 0.540 | 0.900 | 0.696 | 9.542 |
| body_null_only | 1 | body_B-C | DEL | 0.900 | 0.100 | 0.090 | 0.010 | 0.540 | 0.900 | 0.696 | 9.542 |
| body_null_only | 2 | left_flank | non-DEL | 0.000 | 1.000 | 0.100 | 0.900 | 0.100 | 0.100 | 9.542 | 9.542 |
| body_null_only | 3 | right_flank | non-DEL | 0.000 | 1.000 | 0.100 | 0.900 | 0.100 | 0.100 | 9.542 | 9.542 |

