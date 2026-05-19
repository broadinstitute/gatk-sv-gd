# gatk-sv-gd

Genomic disorder copy-number variant detection from binned read-depth data.

`gatk-sv-gd` is a Python command-line package for detecting recurrent genomic
disorder (GD) deletions and duplications at known loci. It is designed as part
of the broader [GATK-SV](https://github.com/broadinstitute/gatk-sv) ecosystem
and focuses on loci where segmental duplications, recurrent breakpoint pairs,
and cohort-level depth effects make standard structural-variant calling hard.

The package provides a staged workflow:

```text
preprocess -> infer -> call -> plot -> eval
```

It also includes utilities to extract putative GD events from VCFs and to spike
synthetic GD events into read-depth and BAF matrices for benchmarking.

## Overview

Genomic disorders are recurrent copy-number changes often mediated by
non-allelic homologous recombination between low-copy repeats or segmental
duplications. A single locus can contain several valid recurrent breakpoint
pairs, such as BP1-BP2, BP1-BP3, and BP2-BP3. `gatk-sv-gd` models these
breakpoint-defined intervals directly instead of treating the whole locus as a
single undifferentiated region.

At a high level, the package:

- Reads binned depth matrices with genomic coordinates and sample columns.
- Loads a GD definition table containing locus, breakpoint, event type, and
  cluster metadata.
- Filters unreliable bins with interval masks, ploidy-aware depth summaries,
  flanking-region checks, optional high-resolution replacement bins, and
  optional BAF summaries.
- Fits a hierarchical Bayesian model implemented in Pyro/PyTorch across all
  retained GD bins and samples.
- Emits per-bin posterior probabilities, per-locus carrier calls, confidence
  scores, plots, and optional truth-set evaluation reports.

The main decision target is sample-level carrier status for each modeled GD
entry: for every sample and GD_ID, decide whether the sample carries the
expected deletion or duplication between the annotated breakpoint pair, while
separating canonical GD events from larger copy-number changes that also affect
flanking sequence.

## Usage

After installation in a Python environment with the package dependencies, the
entry point is:

```bash
gatk-sv-gd --help
```

### Input Tables

The primary depth input is a tab-separated bin matrix with at least these
metadata columns:

| Column | Description |
| --- | --- |
| `Chr` | Contig or chromosome name. |
| `Start` | Zero-based bin start coordinate. |
| `End` | Bin end coordinate. |
| sample columns | One column per sample containing read-depth or normalized coverage values. |

The preprocessing step rescales sample columns so autosomal median depth is
approximately 2.0, corresponding to diploid copy number. Optional high-resolution
count files must be bgzipped and tabix-indexed, use the same sample columns, and
contain raw counts that can be rescaled onto the low-resolution depth scale.

The GD table is a TSV parsed by `GDTable`. Required columns are:

| Column | Description |
| --- | --- |
| `chr` | GD chromosome or contig. |
| `start_GRCh38` | Event start coordinate. |
| `end_GRCh38` | Event end coordinate. |
| `GD_ID` | Stable GD event identifier. |
| `svtype` | `DEL` or `DUP`. |
| `NAHR` | `yes` for modeled NAHR events. |
| `terminal` | Terminal-locus flag, usually `yes` or `no`. |
| `cluster` | Locus cluster key; blank rows are grouped by coordinates. |
| `BP1` | First breakpoint label. |
| `BP2` | Second breakpoint label. |

Exclusion inputs are BED files with at least `chrom`, `start`, and `end` columns.
They can be plain text or gzip-compressed. If chrX bins are present,
`preprocess` currently requires `--par-intervals` so pseudoautosomal bins can be
handled during ploidy-aware filtering.

Optional BAF input is a table with `Chr`, `Pos`, `BAF`, and `Sample` columns. The
preprocessor filters BAF records to retained GD regions and summarizes them by
bin and sample.

### End-to-End Pipeline

For large cohorts, run preprocessing once and then run inference from the cached
preprocessed directory:

```bash
gatk-sv-gd preprocess \
  --input counts.tsv.gz \
  --gd-table gd_table.tsv \
  --exclusion-intervals segdups.bed.gz centromeres.bed.gz \
  --flank-exclusion-intervals problematic_flanks.bed.gz \
  --par-intervals par.hg38.bed \
  --baf-table all_samples.baf.txt.gz \
  --output-dir preprocessed
```

```bash
gatk-sv-gd infer \
  --preprocessed-dir preprocessed \
  --output-dir inference \
  --device cpu
```

Call carriers from posterior probabilities. The default calling mode is direct
posterior marginal scoring:

```bash
gatk-sv-gd call \
  --cn-posteriors inference/cn_posteriors.tsv.gz \
  --bin-mappings preprocessed/bin_mappings.tsv.gz \
  --gd-table preprocessed/gd_table_filtered.tsv \
  --ploidy-table preprocessed/ploidy_estimates.tsv \
  --output-dir calls
```

Generate plots:

```bash
gatk-sv-gd plot \
  --calls calls/gd_cnv_calls.tsv.gz \
  --cn-posteriors inference/cn_posteriors.tsv.gz \
  --sample-posteriors inference/sample_posteriors.tsv.gz \
  --gd-table preprocessed/gd_table_filtered.tsv \
  --ploidy-table preprocessed/ploidy_estimates.tsv \
  --viterbi-paths calls/viterbi_paths.tsv.gz \
  --event-marginals calls/event_marginals.tsv.gz \
  --gtf genes.gtf.gz \
  --segdup-bed segdups.bed.gz \
  --gaps-bed gaps.bed.gz \
  --output-dir plots
```

Evaluate against a truth set:

```bash
gatk-sv-gd eval \
  --calls calls/gd_cnv_calls.tsv.gz \
  --truth-table truth_table.tsv \
  --gd-table preprocessed/gd_table_filtered.tsv \
  --ploidy-table preprocessed/ploidy_estimates.tsv \
  --min-confidence \
  --output-dir evaluation
```

When `--min-confidence` is provided without a value, evaluation uses the default
QUAL equivalent of posterior probability 0.95. If the flag is omitted, no
confidence threshold is applied.

### One-Step Inference Mode

`infer` can also read the raw depth and GD table directly. This is convenient
for small runs or parameter experiments where cached preprocessing is not
needed:

```bash
gatk-sv-gd infer \
  --input counts.tsv.gz \
  --gd-table gd_table.tsv \
  --exclusion-intervals segdups.bed.gz centromeres.bed.gz \
  --output-dir inference
```

This mode performs bin collection before model training and writes the same
posterior outputs as the two-step workflow.

### Auxiliary Commands

Extract GD-overlapping DEL/DUP records from indexed VCF/BCF files:

```bash
gatk-sv-gd extract \
  --vcf cohort.vcf.gz \
  --gd-table gd_table.tsv \
  --output-dir extracted
```

or from a file list:

```bash
gatk-sv-gd extract \
  --vcf-list vcfs.txt \
  --gd-table gd_table.tsv \
  --output-dir extracted
```

Synthesize benchmark events by spiking GD copy-number signal into count and BAF
tables:

```bash
gatk-sv-gd synthesize \
  --lo-res-counts lowres.rd.txt.gz \
  --hi-res-counts highres.rd.txt.gz \
  --baf-table all_samples.baf.txt.gz \
  --ploidy-table preprocessed/ploidy_estimates.tsv \
  --gd-table gd_table.tsv \
  --gd-probability 0.5 \
  --seed 42 \
  --threads 8 \
  --output-dir synthetic
```

## Methods

### Objective and Decision Target

The model supports classification and ranking of GD carrier calls. The target
variable is the latent copy-number state for each retained bin and sample, with
GD_ID-level carrier status derived by aggregating posterior evidence across the
breakpoint-defined body intervals and checking that flanking regions do not show
the same event signal.

The primary outputs for decision-making are posterior probabilities,
posterior-derived QUAL scores, carrier flags, and best-breakpoint assignments
per sample, event type, and locus.

### Data-Generating Model

The current Pyro model uses six unordered diploid pair states over
per-haplotype copy numbers 0, 1, and 2:

```text
(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
```

Each pair state implies a total copy number and an expected minor-allele BAF.
For bin `b` and sample `s`:

```text
z[b,s] ~ Categorical(pair_state_probs[b])
depth[b,s] ~ Normal(total_cn[z[b,s]] * bin_bias[b], sqrt(variance[b,s]))
variance[b,s] = (sample_var[s] + bin_var[b]) * bin_size_factor / interval_size[b]
```

When BAF summaries are available, an additional centered log-likelihood term
compares observed minor-allele BAF with the expected BAF for the pair state. BAF
variance is estimated upstream from the number and spread of SNP sites in each
bin, then scaled by a global BAF temperature so weak or contradictory BAF
evidence lowers confidence rather than dominating depth evidence.

### Assumptions and Justifications

The material assumptions are:

| Assumption | Type | Justification and consequence |
| --- | --- | --- |
| Retained depth bins are comparable across samples after median scaling. | Domain-supported | Upstream GATK-SV coverage normalization and per-sample median rescaling put diploid depth near 2.0. Residual bin and sample effects are modeled explicitly. |
| Most bins and samples are reference at any given GD locus. | Domain-supported | GD events are rare in a cohort, so priors favor the diploid pair state `(1,1)`. |
| Bin-specific bias is multiplicative. | Convenience-driven | GC, mappability, and recurrent bin effects often act as stable multiplicative depth shifts. |
| Sample and bin variance contributions are additive before bin-size scaling. | Convenience-driven | This is a simple identifiable noise model that captures noisy samples and noisy bins without a large interaction term. |
| BAF observations are conditionally informative when SNP support exists. | Domain-supported | Allele balance distinguishes copy states with the same or similar depth evidence, but unsupported bins are masked out. |
| GD entries are represented by annotated breakpoint intervals. | Domain-supported | The GD table defines the recurrent events being tested. Unknown or atypical breakpoints are outside the primary model target. |
| Samples are conditionally exchangeable within a bin given shared bin parameters. | Unverified | This enables cohort-level pooling; batch-specific effects not captured by filtering or variance terms can reduce calibration. |

### Likelihood, Priors, and Inference

The model uses these main latent variables and priors:

| Variable | Prior or construction | Role |
| --- | --- | --- |
| `pair_state_probs[b]` | Dirichlet with `alpha_ref` on `(1,1)` and `alpha_non_ref` on other pair states | Per-bin state prior shared across samples. |
| `z[b,s]` | Categorical over `pair_state_probs[b]` | Latent sample/bin copy state. |
| `bin_bias[b]` | LogNormal centered at 1.0, unless frozen | Per-bin mean depth bias. |
| `sample_var[s]` | Exponential | Sample-level noise. |
| `bin_var[b]` | Exponential, unless frozen | Bin-level noise. |
| `baf_temperature` | LogNormal, or fixed with `--fixed-baf-temperature` | Global scale for BAF variance. |

Continuous latent variables are fit with stochastic variational inference using
Pyro. The default guide is `AutoDiagonalNormal` with an optional AutoDelta MAP
warmup; `--guide-type delta` uses a point-mass MAP approximation. Optimization
uses Adam with an exponential learning-rate schedule and optional early stopping
based on rolling ELBO change.

After continuous parameters are estimated, discrete pair-state posteriors are
computed analytically with Bayes' rule over the six-state table. The legacy
`--n-discrete-samples` option is still accepted for compatibility, but the
implemented discrete inference is exact for the fitted continuous parameters.

### Robustness Strategy

The workflow includes several defenses against sparse, noisy, or biased data:

- Exclusion masks remove bins overlapping segmental duplications, centromeres,
  satellites, or user-specified unreliable intervals.
- Flank-specific masks can remove problematic baseline bins without dropping GD
  body bins.
- Ploidy estimates adjust quality filtering on sex chromosomes and other
  non-diploid contigs.
- PAR intervals are explicitly required when chrX bins are present.
- Body intervals with too few bins can be rescued from optional high-resolution
  tabix-indexed counts.
- Rebinning limits excessive per-interval bin counts while enforcing minimum
  coverage in rebinned intervals.
- Depth values can be clamped to reduce the influence of extreme outliers.
- BAF evidence is masked when missing or unsupported, variance-scaled by SNP
  support, and temperature-scaled globally.
- Posterior-marginal calls require both body-interval evidence and flank
  non-event confidence.

### Scaling Strategy

For small datasets, the strongest stabilizers are the reference-favoring
Dirichlet prior, partial pooling through shared bin parameters, optional fixed
variance or bias parameters, and exact six-state posterior computation after
fitting. These keep the model identifiable when carrier counts are sparse.

For large datasets, preprocessing can be cached and reused; high-resolution
counts and BAF records are queried by tabix over regions of interest; and model
training uses SVI with optional JIT compilation on CPU or CUDA. The main
approximation is variational inference for continuous latent variables, so
posterior uncertainty in those variables depends on the chosen guide family.

### Calling and Confidence Scores

The default `posterior-marginal` caller uses pair-state posterior marginals.
Deletion evidence is the posterior mass where total copy number is below the
sample ploidy; duplication evidence is the posterior mass where total copy
number is above the sample ploidy. Per-bin probabilities are converted to
Phred-like QUAL scores capped at 99. Calls are marked carriers only when covered
body intervals and available flanks pass configurable confidence thresholds:

```bash
--min-posterior-interval-confidence
--min-flank-non-event-confidence
```

The `viterbi` caller smooths per-bin posterior evidence with a user-provided
transition matrix and compares the resulting segments against the expected
breakpoint pattern. An optional breakpoint-specific transition matrix can make
state changes more likely at annotated breakpoint boundaries.

### Validation and Falsification

The intended validation loop is:

- Run prior and parameter-sensitivity checks by varying `alpha_ref`,
  `alpha_non_ref`, variance priors, BAF temperature, and calling thresholds.
- Inspect posterior predictive behavior through carrier plots, locus overview
  plots, event-marginal traces, and BAF panels.
- Evaluate calls against either a curated BED-style truth table or the
  synthesize-format `sample_id`/`GD_ID` truth table.
- Compare sensitivity and precision overall and by GD_ID using
  `truth_evaluation_report.tsv`.
- Stress-test with `synthesize` by changing event rates, salted background
  events, trisomy/YY spike-ins, random seeds, and region filters.
- Check subgroup calibration across sex chromosomes, noisy samples,
  low-bin-count loci, and loci rescued by high-resolution bins.

### Expected Failure Modes

Known limitations and distrust signals include:

- GD entries with missing or incorrect breakpoint labels cannot be assigned to
  the intended body intervals.
- Atypical or non-NAHR events that do not match the annotated recurrent
  breakpoint structure can be missed or classified as non-canonical.
- Broad CNVs spanning both the GD body and flanks may be rejected as GD carriers
  by the flank confidence checks.
- Strong unmodeled batch effects can make the exchangeability and shared-prior
  assumptions poorly calibrated.
- Very sparse body intervals can fail preprocessing unless high-resolution bins
  are provided.
- Missing BAF data reduces state resolution, especially where depth-only states
  are ambiguous.
- Incorrect ploidy estimates or missing PAR annotations can distort chrX/chrY
  filtering and calling.
- Variational approximations can understate continuous-parameter uncertainty,
  especially with `--guide-type delta`.

## Outputs

All subcommands write logs in their output directories. Most tabular outputs are
TSV or bgzipped TSV files.

### `preprocess` Outputs

| File | Description |
| --- | --- |
| `preprocessed_bins.tsv.gz` | Retained, normalized, filtered, and optionally rebinned depth bins for modeled GD loci and flanks. |
| `bin_mappings.tsv.gz` | Mapping from model array index to cluster, interval, chromosome, start, and end. Required by `call`. |
| `locus_intervals.tsv.gz` | Interval coordinates for each retained locus. |
| `gd_entry_intervals.tsv.gz` | Mapping from each GD_ID and breakpoint pair to the one or more modeled sub-intervals it covers. |
| `ploidy_estimates.tsv` | Per-sample, per-contig median depth and rounded ploidy. |
| `gd_table_filtered.tsv` | GD table restricted to loci that survived preprocessing. Recommended for downstream `call`, `plot`, and `eval`. |
| `preprocessed_baf.tsv.gz` | Optional BAF records filtered to retained regions. |
| `preprocessed_baf_summary.tsv.gz` | Optional bin-by-sample BAF summaries used by `infer`. |
| `preprocess_log.txt` | Command and diagnostic log. |

### `infer` Outputs

| File | Description |
| --- | --- |
| `cn_posteriors.tsv.gz` | One row per bin/sample with depth, total-CN posterior columns such as `prob_cn_0`, pair-state posterior columns such as `prob_pair_0_1`, MAP state columns, and optional BAF summaries. |
| `sample_posteriors.tsv.gz` | Sample-level MAP noise parameters and BAF temperature/variance scale when present. |
| `bin_posteriors.tsv.gz` | Bin-level MAP bias, variance, total-CN priors, and pair-state priors. |
| `bin_mappings.tsv.gz` | Written in direct-input mode; reused from `preprocess` in preprocessed mode. |
| `locus_intervals.tsv.gz` | Written in direct-input mode. |
| `gd_entry_intervals.tsv.gz` | Written in direct-input mode. |

### `call` Outputs

| File | Description |
| --- | --- |
| `gd_cnv_calls.tsv.gz` | Sample/GD_ID call table with coordinates, event type, breakpoint labels, interval evidence, carrier flag, best-match flag, calling method, and confidence/QUAL scores. |
| `viterbi_paths.tsv.gz` | Per-sample, per-cluster CN path segments. In posterior-marginal mode this file may be empty but is still written for interface consistency. |
| `event_marginals.tsv.gz` | Per-bin event marginal probabilities and QUAL scores for deletion and duplication evidence. |
| `call_log.txt` | Command and calling log. |

Important `gd_cnv_calls.tsv.gz` columns include `sample`, `cluster`, `GD_ID`,
`chrom`, `start`, `end`, `svtype`, `BP1`, `BP2`, `n_bins`, `sample_ploidy`,
`interval_coverage`, `reciprocal_overlap`, `min_interval_confidence`,
`min_flank_non_event_confidence`, `is_carrier`, `is_best_match`,
`confidence_score`, `qual_score`, and `calling_method`.

### `plot` Outputs

| File or directory | Description |
| --- | --- |
| `carrier_summary.png` | Summary of carrier counts by locus and event type. |
| `confidence_distribution.png` | Distribution of call confidence scores. |
| `locus_plots/*.png` | Per-locus overview plots. |
| `sample_plots/<cluster>/*.png` | Individual sample plots when requested or selected. |
| `carrier_plots.pdf` | Multi-page PDF of carrier plots. |
| `true_positives.pdf`, `false_positives.pdf`, `false_negatives.pdf` | Evaluation-stratified PDFs written when `--eval-report` is supplied. |

### `eval` Outputs

| File | Description |
| --- | --- |
| `truth_evaluation_report.tsv` | Per-GD_ID truth/prediction comparison with truth carriers, predicted carriers, TP, FP, FN, sensitivity, precision, and false-positive/false-negative sample lists. |
| `eval_log.txt` | Command and evaluation log. |

The evaluator accepts two truth-table formats:

- BED-style curated truth table with `#chrom`, `start`, `end`, `name`, `svtype`,
  `samples`, `NAHR_GD`, and `NAHR_GD_atypical` columns.
- Synthesize-format truth table with `sample_id` and `GD_ID` columns.

### `extract` Outputs

| File | Description |
| --- | --- |
| `gd_variants.vcf.gz` | Annotated VCF containing GD-overlapping DEL/DUP records. |
| `gd_variants.vcf.gz.tbi` | Tabix index for the annotated VCF. |
| `gd_variants.bed` | BED-style summary with event coordinates, variant IDs, event type, matched GD IDs, carrier samples, and NAHR/atypical/non-NAHR flags. |
| `extract_log.txt` | Command and extraction log. |

### `synthesize` Outputs

| File | Description |
| --- | --- |
| `lo_res_counts.synthesized.rd.txt.gz` | Low-resolution count table with synthetic events spiked in, when `--lo-res-counts` is provided. |
| `hi_res_counts.synthesized.rd.txt.gz` | High-resolution count table with synthetic events spiked in, when `--hi-res-counts` is provided. |
| `all_samples.synthesized.baf.txt.gz` | BAF table with synthetic allele-balance shifts, when `--baf-table` is provided. |
| `*.tbi` | Tabix indexes for synthesized bgzipped tables. |
| `truth_table.tsv` | Two-column `sample_id`/`GD_ID` truth table for primary synthetic GD carriers. |
| `background_events.tsv` | Manifest of salted flank-bleed events and viable trisomy/YY spike-ins. |
| `synthesize_log.txt` | Command and synthesis log. |

## Development

Run tests from the repository root with the source tree on `PYTHONPATH`:

```bash
PYTHONPATH=src pytest
```

With runtime dependencies installed, run CLI smoke tests without installing the
console script with:

```bash
PYTHONPATH=src python3 -m gatk_sv_gd.cli --help
PYTHONPATH=src python3 -m gatk_sv_gd.cli preprocess --help
PYTHONPATH=src python3 -m gatk_sv_gd.cli infer --help
PYTHONPATH=src python3 -m gatk_sv_gd.cli call --help
```