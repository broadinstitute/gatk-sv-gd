# gatk-sv-gd

Genomic disorder copy-number variant detection from binned read-depth data.

`gatk-sv-gd` is a Python command-line package designed to detect recurrent genomic disorder (GD) deletions and duplications at known loci. Operating as part of the broader [GATK-SV](https://github.com/broadinstitute/gatk-sv) structural variant discovery ecosystem, this package specifically targets complex loci where low-copy repeats, segmental duplications, multiple recurrent breakpoint pairs, and cohort-level depth overdispersion limit the sensitivity and specificity of standard structural-variant calling algorithms.

The package implements a structured, multi-stage workflow:

```text
preprocess -> infer -> call -> plot -> eval -> aggregate
```

Two ancillary subcommands are also provided: `extract` (pulls putative GD events from VCFs) and `integrate` (integrates GD calls back into a GATK-SV final VCF). `integrate` requires `bcftools` to be available on `PATH` at runtime for output sorting.

---

## Installation

The package requires Python $\ge$ 3.9. Install the core library and dependencies directly from the repository root:

```bash
python -m pip install .
```

For developers using modern environments (`uv` or `pip` with PEP 660 support), editable development installations can be initiated via:

```bash
uv pip install -e '.[dev]'
# or
python -m pip install -e '.[dev]'
```

### Primary Dependencies

* **Core Numerics & Data Architecture:** `numpy`, `pandas`
* **Genomic Format Parsers:** `pysam`, `intervaltree`
* **Probabilistic Framework:** `torch`, `pyro-ppl`
* **Visualization Layer:** `matplotlib`

---

## Upstream Pipeline: From BAM/CRAM to Required Inputs

To bridge raw sequencing inputs (BAM/CRAM) to the `gatk-sv-gd` pipeline, users must first execute foundational data aggregation layers of the core [GATK-SV](https://github.com/broadinstitute/gatk-sv) repository. `gatk-sv-gd` does not process individual alignment files directly; it operates on unified multi-sample matrices derived from cohort-scale preprocessing.

The explicit upstream steps required to generate compatible matrices are:

* **Read-Depth Count Matrix Generation:** Alignment datasets (BAM/CRAM) are processed through standard GATK-SV read-count aggregation pipelines. Sequence coverage is binned into fixed genomic windows across all samples in the cohort, yielding a coordinated matrix of raw read counts containing `Chr`, `Start`, and `End` positions.
* **B-Allele Frequency (BAF) Table Generation:** Active germline SNP sites are evaluated across the cohort. Their respective allelic fractions are calculated and formatted using the specialized GATK-SV tool `SiteDepthtoBAF`. This tool outputs a point-based tabular file tracking chromosome, position, allelic ratio, and sample identity, which must then be block-gzipped and tabix-indexed to act as the BAF input layer.

---

## Input Table Architectures

### 1. Read-Depth / Normalized Coverage Matrix

A tab-separated bin matrix containing coordinated genomic intervals and depth metrics across the evaluated cohort.

* **Required Metadata Columns:** `Chr`, `Start`, `End`
* **Data Columns:** One column per sample name populated with raw read counts or normalized coverage values. Preprocessing rescales these values so that an autosomal median of 2.0 reflects a baseline diploid copy number.

### 2. Genomic Disorder Locus Definitions (`--gd-table`)

A TSV defining the structural interval configurations to evaluate. Required fields are:

* `chr`: Target chromosome or contig name.
* `start_GRCh38` / `end_GRCh38`: Structural coordinates of the canonical variant segment.
* `GD_ID`: Unique stable identifier for the genomic disorder entry.
* `svtype`: Expected call class (`DEL` or `DUP`).
* `NAHR`: Set to `yes` to enable NAHR interval modeling.
* `terminal`: Flags terminal-locus status (`yes` or `no`).
* `cluster`: Locus cluster key; blank rows are grouped automatically by coordinates.
* `BP1` / `BP2`: Labeled breakpoint markers establishing the physical bounds.

### 3. Exclusion Profiles (BED Formats)

Standard three-column (`chrom`, `start`, `end`) plain text or gzip-compressed files. Used to provide interval coordinates for segmental duplications, centromeres, reference assembly gaps, or problematic flanking regions.

---

## Reference Resources

All hg38 reference files required to run `gatk-sv-gd` are listed in [`resources.json`](resources.json) and are publicly available in Google Cloud Storage. No authentication is required.

| Key | Description | `run_gd.sh` flag |
|---|---|---|
| `ref_fasta` | Indexed reference FASTA (required for GC fraction computation) | `--ref-fasta` |
| `gd_table` | Genomic disorder locus definitions | `--gd-table` |
| `segdup_bed` | Segmental duplication intervals | `--segdup-bed`, `--flank-exclusion-interval` |
| `centromere_intervals` | Centromere intervals | `--centromere-bed` |
| `acrocentric_intervals` | Acrocentric arm intervals | `--acrocentric-arm-bed` |
| `gaps_bed` | Reference assembly gap intervals | `--gaps-bed` |
| `par_intervals` | Pseudoautosomal region intervals | `--par-bed` |
| `custom_mask` | GD-specific depth mask | `--custom-mask-bed` |
| `inclusion_intervals` | Hard-inclusion intervals (bypass quality filter) | `--hard-inclusion-bed` |
| `gtf` | Gene annotations (Gencode v47, protein-coding) | `--gtf` |

### Downloading resources

Download all resources to a local directory with `gsutil`:

```bash
RESOURCES_DIR="gd_resources"
mkdir -p "${RESOURCES_DIR}"

gsutil cp \
  gs://gatk-sv-resources-public/hg38/v0/sv-resources/resources/v1/gd/GenomicDisorderRegions_hg38_2025-12-05.with_bp.tsv \
  gs://gatk-sv-resources-public/hg38/v0/sv-resources/resources/v1/gd/hg38_GD_custom_mask.bed \
  gs://gatk-sv-resources-public/hg38/v0/sv-resources/resources/v1/gd/hg38_GD_inclusion_intervals.bed \
  gs://gatk-sv-resources-public/hg38/v0/sv-resources/resources/v1/gd/hg38_SD.bed.gz \
  gs://gatk-sv-resources-public/hg38/v0/sv-resources/resources/v1/gd/hg38_acrocentric_arms.bed \
  gs://gatk-sv-resources-public/hg38/v0/sv-resources/resources/v1/gd/hg38_centromeres.bed \
  gs://gatk-sv-resources-public/hg38/v0/sv-resources/resources/v1/gd/hg38_gap.bed \
  gs://gatk-sv-resources-public/hg38/v0/sv-resources/resources/v1/hg38.par.bed \
  gs://gatk-sv-resources-public/hg38/v0/sv-resources/resources/v1/gencode.v47.basic.protein_coding.canonical.gtf \
  "${RESOURCES_DIR}/"
```

---

## Workflow Execution & Commands

### End-to-End Local Execution

For streamlined evaluation on smaller cohorts, the repository bundles `run_gd.sh`, which executes the entire sequence locally, outputting directory trees for each individual phase under a designated workspace.

After downloading the reference resources (see [Reference Resources](#reference-resources) above), run:

```bash
RESOURCES_DIR="gd_resources"  # directory where resources were downloaded

./run_gd.sh \
  --work-dir gd_work \
  --input-depth counts.tsv.gz \
  --ref-fasta /path/to/reference.fa \
  --gd-table "${RESOURCES_DIR}/GenomicDisorderRegions_hg38_2025-12-05.with_bp.tsv" \
  --segdup-bed "${RESOURCES_DIR}/hg38_SD.bed.gz" \
  --flank-exclusion-interval "${RESOURCES_DIR}/hg38_SD.bed.gz" \
  --centromere-bed "${RESOURCES_DIR}/hg38_centromeres.bed" \
  --acrocentric-arm-bed "${RESOURCES_DIR}/hg38_acrocentric_arms.bed" \
  --gaps-bed "${RESOURCES_DIR}/hg38_gap.bed" \
  --par-bed "${RESOURCES_DIR}/hg38.par.bed" \
  --custom-mask-bed "${RESOURCES_DIR}/hg38_GD_custom_mask.bed" \
  --hard-inclusion-bed "${RESOURCES_DIR}/hg38_GD_inclusion_intervals.bed" \
  --gtf "${RESOURCES_DIR}/gencode.v47.basic.protein_coding.canonical.gtf" \
  --baf-table all_samples.baf.txt.gz \
  --high-res-counts highres.rd.txt.gz
```

Stage-specific parameters can be customized and appended to this wrapper script using the `--preprocess-args`, `--infer-args`, `--call-args`, `--eval-args`, and `--plot-args` flags.

### Cached Pipeline Implementation

For large multi-sample cohorts, execute data filtering and interval extraction once, and then trigger parallelized modeling passes downstream from the cached preprocessing outputs.

#### Step 1: Preprocess Data

Normalizes coverage inputs, maps chromosomal ploidies, filters poor-performing bins using a ploidy-aware median/MAD mask, computes optimal dynamic flank sizes, and handles high-resolution interval substitutions.

```bash
RESOURCES_DIR="gd_resources"  # directory where resources were downloaded

gatk-sv-gd preprocess \
  --input counts.tsv.gz \
  --ref-fasta /path/to/reference.fa \
  --gd-table "${RESOURCES_DIR}/GenomicDisorderRegions_hg38_2025-12-05.with_bp.tsv" \
  --exclusion-intervals \
    "${RESOURCES_DIR}/hg38_SD.bed.gz" \
    "${RESOURCES_DIR}/hg38_centromeres.bed" \
    "${RESOURCES_DIR}/hg38_acrocentric_arms.bed" \
    "${RESOURCES_DIR}/hg38_GD_custom_mask.bed" \
  --flank-exclusion-intervals "${RESOURCES_DIR}/hg38_SD.bed.gz" \
  --hard-inclusion-intervals "${RESOURCES_DIR}/hg38_GD_inclusion_intervals.bed" \
  --par-intervals "${RESOURCES_DIR}/hg38.par.bed" \
  --baf-table all_samples.baf.txt.gz \
  --high-res-counts highres.rd.txt.gz \
  --output-dir preprocessed
```

*Note: If specific target bins must bypass quality and exclusion filtering to guarantee spatial continuity, pass their coordinates via `--hard-inclusion-intervals`. If chrX bins are present, `--par-intervals` must be provided to accurately isolate pseudoautosomal segments.*

#### Step 2: Model Inference

Fits continuous variational profiles and generates exact multi-state posterior tables across the preprocessed intervals.

```bash
gatk-sv-gd infer \
  --preprocessed-dir preprocessed \
  --output-dir inference \
  --device cpu
```

#### Step 3: Genotype & Call Carriers

Processes pair-state posteriors into explicit event calls.

```bash
gatk-sv-gd call \
  --cn-posteriors inference/cn_posteriors.tsv.gz \
  --bin-mappings preprocessed/bin_mappings.tsv.gz \
  --gd-table preprocessed/gd_table_filtered.tsv \
  --ploidy-table preprocessed/ploidy_estimates.tsv \
  --output-dir calls
```

##### Supported Calling Modes

* **`posterior-marginal` (Default):** Direct scoring from pair-state posterior marginals. Converts posterior mass into signed, Phred-like QUAL values using a correlation-adjusted effective independent bin penalty to neutralize counting inflation. It enforces a flank non-event check to discard broad, cross-locus megabase alterations.

#### Step 4: Visualize Regional Diagnostics

Renders summaries of call distributions, QUAL spreads, and high-fidelity multi-panel locus tracks.

```bash
gatk-sv-gd plot \
  --calls calls/gd_cnv_calls.tsv.gz \
  --cn-posteriors inference/cn_posteriors.tsv.gz \
  --sample-posteriors inference/sample_posteriors.tsv.gz \
  --gd-table preprocessed/gd_table_filtered.tsv \
  --ploidy-table preprocessed/ploidy_estimates.tsv \
  --event-marginals calls/event_marginals.tsv.gz \
  --gtf "${RESOURCES_DIR}/gencode.v47.basic.protein_coding.canonical.gtf" \
  --segdup-bed "${RESOURCES_DIR}/hg38_SD.bed.gz" \
  --gaps-bed "${RESOURCES_DIR}/hg38_gap.bed" \
  --output-dir plots
```

#### Step 5: Cohort Aggregation and Reporting

Compiles multiple discrete run folders into a global cohort PDF report and machine-readable data structures. This aggregates burden metrics, sorts carriers into absolute confidence tiers, and tracks structural sample evidence.

```bash
gatk-sv-gd aggregate gd_work_a gd_work_b \
  --output-dir aggregate \
  --min-confidence 0.5
```

---

## Benchmarking & Empirical Validation

To support algorithmic verification and calibration, `gatk-sv-gd` includes built-in tools for benchmarking pipeline performance.

### Synthetic Event Spiking

The `synthesize` engine models true genetic signals by spiking artificial copy-number events directly into raw low-resolution, high-resolution, or BAF source matrices. It handles complex biological confounders by inserting random localized flank-bleed events, whole-chromosome trisomies, and sex-chromosome anomalies relative to an empirical ploidy map.

```bash
gatk-sv-gd synthesize \
  --lo-res-counts lowres.rd.txt.gz \
  --hi-res-counts highres.rd.txt.gz \
  --baf-table all_samples.baf.txt.gz \
  --ploidy-table preprocessed/ploidy_estimates.tsv \
  --gd-table gd_table.tsv \
  --gd-probability 0.5 \
  --seed 42 \
  --output-dir synthetic
```

### Truth-Set Evaluation

The `eval` module scores called carrier tables against a truth table and reports sensitivity, precision, TP/FP/FN counts, and sample-level discrepancies overall and per `GD_ID`. Two truth table formats are accepted and auto-detected from the header.

#### Format 1: BED-style (manually curated)

Tab-separated, one row per carrier event. A leading `#` on the header is stripped automatically.

| Column | Description |
|---|---|
| `chrom` | Chromosome (e.g. `chr15`) |
| `start` | 0-based start coordinate |
| `end` | End coordinate |
| `name` | GD identifier — matched against `GD_ID` in the calls table |
| `svtype` | `DEL` or `DUP` |
| `samples` | Comma-separated list of carrier sample IDs |
| `NAHR_GD` | `True` / `False` — only `True` rows are evaluated |
| `NAHR_GD_atypical` | `True` / `False` — `True` rows are excluded before scoring |

Example:

```
#chrom	start	end	name	svtype	samples	NAHR_GD	NAHR_GD_atypical
chr15	22800000	28500000	15q11.2_BP1-BP2_DEL	DEL	SAMPLE_A,SAMPLE_B	True	False
chr15	28500000	30100000	15q11.2_BP2-BP3_DUP	DUP	SAMPLE_C	True	False
```

#### Format 2: Synthesize format (output of `gatk-sv-gd synthesize`)

Tab-separated, one row per carrier sample. All rows are used as-is with no NAHR filtering.

| Column | Description |
|---|---|
| `sample_id` | Sample identifier |
| `GD_ID` | GD identifier — matched against `GD_ID` in the calls table |

Example:

```
sample_id	GD_ID
SAMPLE_A	15q11.2_BP1-BP2_DEL
SAMPLE_B	15q11.2_BP1-BP2_DEL
SAMPLE_C	15q11.2_BP2-BP3_DUP
```

```bash
gatk-sv-gd eval \
  --calls calls/gd_cnv_calls.tsv.gz \
  --truth-table truth_table.tsv \
  --gd-table preprocessed/gd_table_filtered.tsv \
  --ploidy-table preprocessed/ploidy_estimates.tsv \
  --output-dir evaluation
```

---

## Scientific & Methodological Framework

### Objective & Locus Modeling

Recurrent genomic disorders are copy-number changes typically mediated by non-allelic homologous recombination (NAHR) between flanking segmental duplications or low-copy repeats. A single disease-associated locus frequently presents multiple distinct, overlapping canonical architectures defined by variable breakpoint configurations (e.g., BP1–BP2, BP1–BP3, and BP2–BP3).

Rather than aggregating an entire locus into a single undifferentiated region, `gatk-sv-gd` models these individual breakpoint-defined intervals directly. The primary decision target is to evaluate sample-level carrier status for each specified genomic disorder entry, mathematically partitioning canonical GD events from larger, non-recurrent copy-number adjustments that propagate into flanking sequences.

### Data-Generating Model

The package optimizes a hierarchical Bayesian network implemented via the Pyro probabilistic programming framework and PyTorch. The hidden layer tracks six unordered diploid pair states across per-haplotype copy numbers 0, 1, and 2:

```text
(0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
```

For a genomic bin $b$ and sample $s$, the latent copy state $z_{b,s}$ is drawn from a shared bin-specific prior. Let $d_{b,s}$ denote the observed read depth. It is modeled conditionally as:

$$\mathbb{E}[d_{b,s}] = c(z_{b,s}) \times \beta_b$$

$$d_{b,s} \sim \mathcal{N}\left(\mathbb{E}[d_{b,s}],\ \sigma^2_{b,s}\right)$$

The conditional variance $\sigma^2_{b,s}$ relies on a count-anchored spatial model that links sampling noise to an empirical baseline:

1. **Poisson Baseline:** Calculated explicitly from preprocessing-time per-sample raw autosomal count medians to lock the variance floor to physical sequencing depth rather than allowing it to float as a free parameter.
2. **Spatial Excess Overdispersion:** Modeled as a function of sample overdispersion multiplied by a continuous-AR(1) spatial aggregation factor $f(L; \ell)$. This formulation preserves full excess variance for narrow bins and asymptotically scales inversely with length for wide bins, capturing the empirical averaging out of adjacent counting noise over extended physical intervals.

### Informative BAF Integration

When available, minor B-allele frequency (BAF) tables are integrated into the network. Observed minor-allele balances are evaluated against the expected fractions for each pair state via a centered log-likelihood matrix. To prevent highly localized or noisy SNP distributions from destabilizing the core depth likelihood, BAF variances are scaled upstream by regional SNP density and downweighted globally via an optimized temperature parameter.

### Inference Strategy

Continuous latent variables (such as bin-specific bias, sample overdispersion, and physical correlation length scales) are fit utilizing Stochastic Variational Inference (SVI) via an Adam optimizer and an exponential learning-rate schedule.

Once these continuous map parameters are estimated, the hidden discrete pair-state posteriors are computed exactly and analytically via Bayes' rule over the six-state table. This analytical update path avoids the statistical noise and computational overhead of Monte Carlo sampling. To guarantee robustness against extreme technical outliers, the exact inference layer incorporates a neutral outer null-state prior. Bins that deviate severely from off-model parameters dump their posterior mass safely into this null state, yielding a pignistic neutralization that protects adjacent called-state confidence intervals from catastrophic local penalties.