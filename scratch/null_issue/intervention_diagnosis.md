# Exact-inference intervention comparison

Synthetic scenarios:
- `haploid_reference_bin`: observed normalized depth 1.5 with the default diploid-reference-biased pair-state prior.
- `deletion_biased_lowdepth_bin`: observed normalized depth 0.1 with an upstream prior that already leans toward CN0 but still includes substantial neutral mass.

Interventions compared:
- `baseline`: current old exact-inference behavior (no ploidy input, no CN0 variance floor).
- `ploidy_only`: sample ploidy attached to exact inference, but no CN0 variance floor.
- `variance_floor_only`: CN0 residual-depth tolerance only.
- `ploidy_and_variance_floor`: both changes together.

Interpretation:
- On the haploid-reference synthetic bin, the ploidy-aware exact-inference path does what it is supposed to do: it moves posterior mass from diploid-reference CN2 toward haploid-neutral CN1.
- On the deletion-biased low-depth bin, the CN0 variance floor is the intervention that directly recovers deletion posterior and reduces null/off-model behavior.
- The combined regime shows that these two interventions solve different problems and can trade off against each other depending on the learned pair-state prior for the bin.

Haploid-reference comparison:

| scenario | regime | observed_depth | min_variance_expected_depth | sample_ploidy | posterior_cn0 | posterior_cn1 | posterior_cn2_ref | posterior_null |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| haploid_reference_bin | baseline | 1.500000 | 0.000000 | nan | 0.000000 | 0.023915 | 0.942343 | 0.002743 |
| haploid_reference_bin | ploidy_only | 1.500000 | 0.000000 | 1.000000 | 0.000000 | 0.957873 | 0.015098 | 0.002197 |
| haploid_reference_bin | variance_floor_only | 1.500000 | 0.100000 | nan | 0.000000 | 0.023915 | 0.942343 | 0.002743 |
| haploid_reference_bin | ploidy_and_variance_floor | 1.500000 | 0.100000 | 1.000000 | 0.000000 | 0.957873 | 0.015098 | 0.002197 |

Deletion-biased low-depth comparison:

| scenario | regime | observed_depth | min_variance_expected_depth | sample_ploidy | posterior_cn0 | posterior_cn1 | posterior_cn2_ref | posterior_null |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deletion_biased_lowdepth_bin | baseline | 0.100000 | 0.000000 | nan | 0.000000 | 0.506355 | 0.195077 | 0.040073 |
| deletion_biased_lowdepth_bin | ploidy_only | 0.100000 | 0.000000 | 1.000000 | 0.000000 | 0.980827 | 0.000151 | 0.009007 |
| deletion_biased_lowdepth_bin | variance_floor_only | 0.100000 | 0.100000 | nan | 0.970619 | 0.014877 | 0.005732 | 0.001177 |
| deletion_biased_lowdepth_bin | ploidy_and_variance_floor | 0.100000 | 0.100000 | 1.000000 | 0.561370 | 0.430220 | 0.000066 | 0.003951 |

