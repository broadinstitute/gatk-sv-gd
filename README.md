# gatk-sv-gd

Genomic disorder copy-number variant detection from binned read-depth data.

`gatk-sv-gd` is a Python command-line package for detecting recurrent genomic
disorder (GD) deletions and duplications at known loci. It is designed as part
of the broader [GATK-SV](https://github.com/broadinstitute/gatk-sv) ecosystem
and focuses on loci where segmental duplications, recurrent breakpoint pairs,
and cohort-level depth effects make standard structural-variant calling hard.

The package provides a staged workflow:

```text
preprocess -> infer -> call -> plot -> eval -> aggregate
```

## Installation

Install the package from the repository root with:

```bash
python -m pip install .
```

For editable installs during development, use a modern `pip` with PEP 660
support or `uv`:

```bash
python -m pip install -e '.[dev]'
```

```bash
uv pip install -e '.[dev]'
```

If your virtual environment does not have `pip` bootstrapped yet, run:

```bash
python -m ensurepip --upgrade
```

Editable installs with very old `pip` releases may fall back to deprecated
`setup.py develop` behavior. This project is configured for standards-based
builds and editable installs.

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

For a one-command local run, the repository includes `run_gd.sh`, which wraps
the current CLI stages and writes `preprocess`, `infer`, `call`, and `plot`
subdirectories under a work directory. The truth-set evaluation step, and its
`eval` subdirectory, are only run when `--truth-table` is supplied:

```bash
./run_gd.sh \
  --work-dir gd_work \
  --input-depth counts.tsv.gz \
  --gd-table gd_table.tsv \
  --segdup-bed segdups.bed.gz \
  --centromere-bed centromeres.bed.gz \
  --par-bed par.hg38.bed \
  --baf-table all_samples.baf.txt.gz \
  --high-res-counts highres.rd.txt.gz \
  --gtf genes.gtf.gz \
  --gaps-bed gaps.bed.gz
```

Optional inputs such as high-resolution counts, BAF, annotation tracks, and
truth tables can be omitted when they are not available. Stage-specific options
can be forwarded with `--preprocess-args`, `--infer-args`, `--call-args`,
`--eval-args`, and `--plot-args`.

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

If specific bins must survive preprocessing even when they overlap exclusion
regions or fail median/MAD filtering, provide BED intervals with
`--hard-inclusion-intervals forced_keep.bed.gz`.

```bash
gatk-sv-gd infer \
  --preprocessed-dir preprocessed \
  --output-dir inference \
  --device cpu
```

`preprocess` writes `normalization_metadata.tsv`, which records the per-sample
raw-count medians and the reference low-resolution bin size used by the
count-anchored spatial variance model. `infer` now requires that metadata when
reading a preprocessed directory; if it is missing, rerun `preprocess` or also
provide `--input` so `infer` can recompute it.

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
  --output-dir evaluation
```

The eval command scores the confident carrier calls emitted by `call`; it does
not apply an additional confidence cutoff.

Aggregate one or more completed `run_gd.sh` work directories into a cohort-level
PDF report and machine-readable sidecar tables:

```bash
gatk-sv-gd aggregate gd_work_a gd_work_b \
  --output-dir aggregate \
  --min-confidence 0.5 \
  --batch-label batch_a \
  --batch-label batch_b
```

The aggregate command expects each input directory to contain standard
`preprocess` and `call` outputs, including `preprocess/ploidy_estimates.tsv`,
`preprocess/gd_table_filtered.tsv`, and `call/gd_cnv_calls.tsv.gz`. Optional
`infer`, `eval`, and `plot` artifacts are inventoried when present and listed in
`aggregate_missing_artifacts.tsv` when absent. Aggregate does not re-render
full signal plots; it summarizes calls and evaluation reports across work
directories, adds per-case evidence plots from call metrics, records where
existing plot artifacts were found, and derives call-selection criteria from the
call outputs so confident and non-confident case sections stay consistent with
the original calling step. `--min-confidence` is only a lower bound for showing
non-confident best-match calls in the aggregate outputs; it does not affect the
confident carriers emitted by `call`.

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
posterior outputs as the two-step workflow. In direct-input mode, `infer`
computes and writes `normalization_metadata.tsv` automatically from the raw
depth matrix before fitting the count-anchored spatial variance model.

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
expected_depth[b,s] = total_cn[z[b,s]] * bin_bias[b]
depth[b,s] ~ Normal(expected_depth[b,s], sqrt(variance[b,s]))
reference_variance[s] = 4 / raw_count_median[s] * reference_bin_size / bin_size_factor
poisson_variance[b,s] = reference_variance[s] * bin_size_factor / interval_size[b] * expected_depth[b,s] / 2
variance[b,s] = poisson_variance[b,s] + expected_depth[b,s]^2 * sample_var[s] * f(interval_size[b]; length_scale_var)
```

Here `raw_count_median[s]` and `reference_bin_size` come from
`normalization_metadata.tsv`, and `f(L; ell)` is the continuous-AR(1) spatial
aggregation factor. It approaches 1 for short bins and decays approximately as
`2 * ell / L` once the interval length is much larger than the learned shared
correlation length `length_scale_var`.

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
| Sample-level raw-count medians capture the dominant Poisson baseline at the reference bin size. | Domain-supported | The count-anchored variance term uses preprocessing-time raw-count medians to tie the depth likelihood to observed sampling depth instead of a free variance floor. |
| Most bins and samples are reference at any given GD locus. | Domain-supported | GD events are rare in a cohort, so priors favor the diploid pair state `(1,1)`. |
| Bin-specific bias is multiplicative. | Convenience-driven | GC, mappability, and recurrent bin effects often act as stable multiplicative depth shifts. |
| Excess depth variance is sample-specific and decays with interval length according to a shared physical correlation scale. | Mechanistic approximation | The spatial aggregation factor preserves full excess variance for short bins and shrinks it toward zero for long bins, matching the idea that adjacent counting noise averages out over longer intervals. |
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
| `reference_variance[s]` | Deterministic from `raw_count_median[s]` and `reference_bin_size` | Sample-specific count-anchored Poisson baseline at diploid depth. |
| `sample_var[s]` | Exponential | Sample-level excess variance above the Poisson baseline. |
| `length_scale_var` | Exponential | Shared physical correlation length controlling how quickly excess variance averages out across longer bins. |
| `baf_temperature` | LogNormal, or fixed with `--fixed-baf-temperature` | Global scale for BAF variance. |

The old per-bin excess variance latent is no longer part of the fitted model.
Compatibility CLI flags such as `--var-bin` and `--unfreeze-bin-var` are still
accepted, but they do not change the spatial count-anchored likelihood.

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
Dirichlet prior, the count-anchored Poisson baseline, partial pooling through
shared bin parameters, optional fixed bias parameters, and exact six-state
posterior computation after fitting. These keep the model identifiable when
carrier counts are sparse.

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

### Validation and Falsification

The intended validation loop is:

- Run prior and parameter-sensitivity checks by varying `alpha_ref`,
  `alpha_non_ref`, `var_sample`, `var_length_scale`, BAF temperature, and
  calling thresholds.
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

Logs are intentionally quiet and privacy-safe. They include run metadata,
dependency versions, lifecycle messages, and warning/error diagnostics, but do
not include raw data values, sample identifiers, or input/output file paths.
Routine per-sample, per-bin, and progress-style messages are suppressed.

### `preprocess` Outputs

| File | Description |
| --- | --- |
| `preprocessed_bins.tsv.gz` | Retained, normalized, filtered, and optionally rebinned depth bins for modeled GD loci and flanks. |
| `bin_mappings.tsv.gz` | Mapping from model array index to cluster, interval, chromosome, start, and end. Required by `call`. |
| `locus_intervals.tsv.gz` | Interval coordinates for each retained locus. |
| `gd_entry_intervals.tsv.gz` | Mapping from each GD_ID and breakpoint pair to the one or more modeled sub-intervals it covers. |
| `normalization_metadata.tsv` | Per-sample raw-count medians and the shared reference low-resolution bin size used by the count-anchored spatial variance model. Required by `infer`. |
| `ploidy_estimates.tsv` | Per-sample, per-contig median depth and rounded ploidy. |
| `gd_table_filtered.tsv` | GD table restricted to loci that survived preprocessing. Recommended for downstream `call`, `plot`, and `eval`. |
| `preprocessed_baf.tsv.gz` | Optional BAF records filtered to retained regions. |
| `preprocessed_baf_summary.tsv.gz` | Optional bin-by-sample BAF summaries used by `infer`, including occupancy-adjusted `baf_effective_n_sites` / `baf_effective_variance` when ploidy context is available. |
| `preprocess_log.txt` | Command and diagnostic log. |

### `infer` Outputs

| File | Description |
| --- | --- |
| `cn_posteriors.tsv.gz` | One row per bin/sample with depth, total-CN posterior columns such as `prob_cn_0`, pair-state posterior columns such as `prob_pair_0_1`, MAP state columns, and optional raw plus occupancy-adjusted BAF summaries. |
| `sample_posteriors.tsv.gz` | Sample-level MAP parameters including `sample_var_map`, shared `length_scale_var_map`, and BAF temperature/variance scale when present. |
| `bin_posteriors.tsv.gz` | Bin-level MAP bias, compatibility `bin_var_map` values (currently zero under the spatial model), total-CN priors, and pair-state priors. |
| `normalization_metadata.tsv` | Copy of the normalization metadata used to anchor the depth variance model. Written in both direct-input and preprocessed modes. |
| `bin_mappings.tsv.gz` | Written in direct-input mode; reused from `preprocess` in preprocessed mode. |
| `locus_intervals.tsv.gz` | Written in direct-input mode. |
| `gd_entry_intervals.tsv.gz` | Written in direct-input mode. |

### `call` Outputs

| File | Description |
| --- | --- |
| `gd_cnv_calls.tsv.gz` | Sample/GD_ID call table with coordinates, event type, breakpoint labels, interval evidence, carrier flag, best-match flag, null anomaly annotations, calling method, and confidence/QUAL scores. |
| `event_marginals.tsv.gz` | Per-bin event marginal probabilities and QUAL scores for deletion and duplication evidence. |
| `call_log.txt` | Command and calling log. |

Important `gd_cnv_calls.tsv.gz` columns include `sample`, `cluster`, `GD_ID`,
`chrom`, `start`, `end`, `svtype`, `BP1`, `BP2`, `n_bins`, `sample_ploidy`,
`interval_coverage`, `reciprocal_overlap`, `min_interval_confidence`,
`min_flank_non_event_confidence`, `is_carrier`, `is_best_match`,
`null_anomaly_score`, `is_null_anomalous`, `confidence_score`, `qual_score`,
and `calling_method`.

### `plot` Outputs

| File or directory | Description |
| --- | --- |
| `carrier_summary.png` | Summary of carrier counts by locus and event type. |
| `confidence_distribution.png` | Distribution of call confidence scores. |
| `locus_plots/*.png` | Per-locus overview plots. |
| `sample_plots/<cluster>/*.png` | Individual sample plots when requested or selected. |
| `carrier_plots.pdf` | Multi-page PDF of carrier plots. |
| `true_positives.pdf`, `false_positives.pdf`, `false_negatives.pdf`, `anomalous_discrepancies.pdf` | Evaluation PDFs plus an anomalous-sample review PDF; the anomalous PDF is written whenever flagged anomalous samples are present, even if `--eval-report` is omitted. |

### `eval` Outputs

| File | Description |
| --- | --- |
| `truth_evaluation_report.tsv` | Per-GD_ID truth/prediction comparison with truth carriers, predicted carriers, TP, FP, FN, sensitivity, precision, false-positive/false-negative sample lists, and anomalous discrepancy sample lists. |
| `eval_log.txt` | Command and evaluation log. |

The evaluator accepts two truth-table formats:

- BED-style curated truth table with `#chrom`, `start`, `end`, `name`, `svtype`,
  `samples`, `NAHR_GD`, and `NAHR_GD_atypical` columns.
- Synthesize-format truth table with `sample_id` and `GD_ID` columns.

### `aggregate` Outputs

| File | Description |
| --- | --- |
| `aggregate_report.pdf` | Cohort-level PDF with ploidy-style front matter, batch inventory, summary metrics, carrier case index, locus burden, confident and non-confident case detail pages with evidence plots, optional evaluation summary, missing optional artifacts, and field guide. |
| `aggregate_summary.tsv` | Compact metric/value table with batch, sample, GD_ID, carrier, and optional evaluation totals. |
| `aggregate_inventory.tsv` | One row per input work directory with sample, locus, call, carrier, evaluation, and optional artifact counts. |
| `aggregate_calls.tsv` | All input `gd_cnv_calls.tsv.gz` rows with batch labels, sample keys, normalized confidence values, and carrier categories. |
| `aggregate_cases.tsv` | Reportable carrier calls after applying `is_carrier` and, when present, `is_best_match`; carriers are split into high- and low-confidence categories. |
| `aggregate_locus_summary.tsv` | Per-batch, per-GD_ID carrier burden and confidence summaries. |
| `aggregate_eval.tsv` | Concatenated optional `truth_evaluation_report.tsv` rows with batch columns. |
| `aggregate_missing_artifacts.tsv` | Optional aggregate inputs not found or unreadable in each work directory. |
| `aggregate_log.txt` | Command and aggregate log. |

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

Install development dependencies, including coverage tooling, with:

```bash
python -m pip install -e '.[dev]'
```

Run tests from the repository root with the source tree on `PYTHONPATH`:

```bash
PYTHONPATH=src pytest
```

If you are working in the project virtual environment, the equivalent command
is:

```bash
.venv/bin/python -m pytest
```

Coverage reporting is enabled by default through `pytest-cov`, so either test
command prints a terminal coverage summary for `gatk_sv_gd` and enforces the
current minimum coverage threshold.

To generate the HTML coverage report after a test run, use:

```bash
python -m coverage html
```

The HTML report is written to `build/coverage-html/index.html`.

With runtime dependencies installed, run CLI smoke tests without installing the
console script with:

```bash
PYTHONPATH=src python3 -m gatk_sv_gd.cli --help
PYTHONPATH=src python3 -m gatk_sv_gd.cli preprocess --help
PYTHONPATH=src python3 -m gatk_sv_gd.cli infer --help
PYTHONPATH=src python3 -m gatk_sv_gd.cli call --help
PYTHONPATH=src python3 -m gatk_sv_gd.cli aggregate --help
```