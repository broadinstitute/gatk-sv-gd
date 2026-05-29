# Synthetic null-inference diagnosis

Setup:
- One synthetic haploid chrX bin with no BAF contribution.
- Normalized depths use the model's convention that CN=2 corresponds to depth 2.0, so haploid neutral CN=1 sits at depth 1.0.
- The exact inference model itself is not sample-ploidy-aware; ploidy is estimated upstream for filtering and then only reintroduced later during calling.
- Default null prior = 0.001; default pair-state prior is reference-biased with alpha_ref=50.0 and alpha_non_ref=1.0.
- Count-anchored depth config: raw_count_median=200.0, interval_size=10000, sample_var=0.200, length_scale_var=20000.

Key mechanism:
- The explicit CN0 deletion state has expected depth 0.
- Its Poisson term is multiplied by depth_scale(expected_depth), and expected_depth=0 is clamped only to a tiny floor of 1e-6.
- Its excess variance term is expected_depth^2 * sample_var * spatial_factor, which is exactly 0 when expected_depth=0.
- The outer null state has no depth or BAF likelihood penalty at all; it contributes only its prior mass.

Representative state geometry:
- CN0 state (0,0): expected_depth=0.000, std=0.000141
- Haploid-neutral state (0,1): expected_depth=1.000, std=0.424793
- Diploid-reference state (1,1): expected_depth=2.000, std=0.837733

Consequence at observed normalized depth 0.10:
- Default prior: posterior(CN0)=0.000000, posterior(CN1)=0.048632, posterior(CN2 ref)=0.888599, posterior(null)=0.026898
- Uniform priors: posterior(CN0)=0.000000, posterior(CN1)=0.462247, posterior(CN2 ref)=0.168921, posterior(null)=0.027891
- So the immediate effect of residual depth is to destroy the explicit CN0 state. Where that mass goes next depends on the rest of the model; in this depth-only toy it mostly spills into broader nonzero-CN states, with null remaining nontrivial rather than dominant.

Interpretation:
- A real hemizygous deletion can still have residual normalized depth around 0.05-0.20 from background counts, mismapping, repeats, or imperfect normalization.
- Under this model, those residual depths are catastrophically unlikely under the explicit CN0 state because its variance is nearly zero.
- The inference model is also ploidy-agnostic, so haploid bins compete against the same diploid pair-state family and reference-biased prior used everywhere else.
- In the pure depth-only toy above, that makes broad higher-CN states surprisingly competitive. In a real deletion region, any additional evidence that weakens those alternatives will push more of the off-model mass into null.
- That means high prob_null is best understood as a symptom that the explicit haploid-deletion state is too brittle and the upstream inference model is not calibrated for haploid copy-0 bins.

State geometry table:

| state | total_cn | expected_depth | poisson_variance | excess_variance | total_variance | std |
| --- | --- | --- | --- | --- | --- | --- |
| (0,0) | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000141 |
| (0,1) | 1.000000 | 1.000000 | 0.010000 | 0.170449 | 0.180449 | 0.424793 |
| (0,2) | 2.000000 | 2.000000 | 0.020000 | 0.681796 | 0.701796 | 0.837733 |
| (1,1) | 2.000000 | 2.000000 | 0.020000 | 0.681796 | 0.701796 | 0.837733 |
| (1,2) | 3.000000 | 3.000000 | 0.030000 | 1.534041 | 1.564041 | 1.250616 |
| (2,2) | 4.000000 | 4.000000 | 0.040000 | 2.727185 | 2.767185 | 1.663486 |

Observed-depth sweep excerpt:

| prior_regime | observed_depth | posterior_del_state_00 | posterior_haploid_neutral_01 | posterior_dup_state_11 | posterior_null |
| --- | --- | --- | --- | --- | --- |
| default_ref_biased | 0.000000 | 0.999451 | 0.000021 | 0.000488 | 0.000020 |
| default_ref_biased | 0.010000 | 0.000000 | 0.038963 | 0.888786 | 0.034525 |
| default_ref_biased | 0.020000 | 0.000000 | 0.040007 | 0.888916 | 0.033568 |
| default_ref_biased | 0.030000 | 0.000000 | 0.041060 | 0.889004 | 0.032639 |
| default_ref_biased | 0.040000 | 0.000000 | 0.042122 | 0.889053 | 0.031740 |
| default_ref_biased | 0.050000 | 0.000000 | 0.043192 | 0.889064 | 0.030868 |
| default_ref_biased | 0.060000 | 0.000000 | 0.044269 | 0.889038 | 0.030024 |
| default_ref_biased | 0.070000 | 0.000000 | 0.045352 | 0.888977 | 0.029205 |
| default_ref_biased | 0.080000 | 0.000000 | 0.046441 | 0.888883 | 0.028412 |
| default_ref_biased | 0.090000 | 0.000000 | 0.047535 | 0.888756 | 0.027643 |
| default_ref_biased | 0.100000 | 0.000000 | 0.048632 | 0.888599 | 0.026898 |

Observed-depth 0.10 per-state breakdown:

| state | expected_depth | std | pair_prior | log_likelihood | posterior |
| --- | --- | --- | --- | --- | --- |
| (0,0) | 0.000000 | 0.000141 | 0.018182 | -249992.055172 | 0.000000 |
| (0,1) | 1.000000 | 0.424793 | 0.018182 | -2.307186 | 0.048632 |
| (0,2) | 2.000000 | 0.837733 | 0.018182 | -3.313854 | 0.017772 |
| (1,1) | 2.000000 | 0.837733 | 0.909091 | -3.313854 | 0.888599 |
| (1,2) | 3.000000 | 1.250616 | 0.018182 | -3.831123 | 0.010595 |
| (2,2) | 4.000000 | 1.663486 | 0.018182 | -4.176134 | 0.007503 |

Raw-count-median sweep at observed depth 0.10:

| raw_count_median | cn0_std | posterior_del_state_00 | posterior_haploid_neutral_01 | posterior_null |
| --- | --- | --- | --- | --- |
| 25.000000 | 0.000400 | 0.000000 | 0.055356 | 0.019262 |
| 50.000000 | 0.000283 | 0.000000 | 0.052805 | 0.022904 |
| 100.000000 | 0.000200 | 0.000000 | 0.050307 | 0.025407 |
| 200.000000 | 0.000141 | 0.000000 | 0.048632 | 0.026898 |
| 500.000000 | 0.000089 | 0.000000 | 0.047468 | 0.027886 |
| 1000.000000 | 0.000063 | 0.000000 | 0.047051 | 0.028232 |
| 2000.000000 | 0.000045 | 0.000000 | 0.046837 | 0.028408 |

