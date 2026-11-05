# LoCovE

LoCovE estimates local reference coverage from sampled k-mer counts without
mapping reads. It reports depth and k-mer breadth within reference segments.
Shared k-mers contribute to multiple references, so coverage is pooled rather
than assigned to a single source. K-mer breadth is not base breadth.

## Build and run

Install Rust and Cargo, then build:

```sh
cargo build --release --locked
```

Run the included artificial example (no downloads needed):

```sh
mkdir -p demo/output/coverage
target/release/locove build --out demo/output/reference.idx \
    --record-map-out demo/output/records.json demo/reference.fa
target/release/locove run --index demo/output/reference.idx \
    --record-map demo/output/records.json --out demo/output/coverage \
    --threads 1 demo/reads.fq
```

For your own data, replace the reference with one FASTA file per genome and the
reads with FASTQ or gzip-compressed FASTQ files. Multiple read files, including
R1/R2 files, are combined into one sample. Reuse the index across samples and use
new output paths for each run. Existing outputs are not overwritten.

Index defaults are `--k 21 --c 20 --g 300`: k-mer length, subsampling factor, and
approximate segment length. `--threads` specifies query workers; with more than
one worker, an additional input thread is used. For example, `--threads 7` uses
an eight-thread pipeline. `--threads 1` runs serially.

## Outputs

The output directory contains `segments.npz` (coordinates and coverage arrays),
`record-map.json` (reference names), `report.json` (parameters and sample
statistics), and `phases.json` (stage timings).

In `segments.npz`, each array has one entry per segment:

- `seg_start`, `seg_end`: zero-based record coordinates, with the end excluded.
- `seg_record`: record index in `record-map.json`.
- `n_anchors`: number of retained sampled k-mers.
- `breadth_anchor`: fraction observed in reads.
- `depth_raw_mean`: average raw k-mer count.
- `depth_geom_per_read`: depth adjusted for read length.
- `depth_geom_per_read_novelty`: depth additionally adjusted for sequencing errors,
  used for the main manuscript coverage estimates.

The `depth_geom_global` variants instead use a sample-wide read-length adjustment.
Segments without retained k-mers have undefined estimates, not zero coverage.
The sequencing-error adjustment uses absence from the catalog, which can also
reflect missing reference sequence rather than sequencing error.

This basic release contains the coverage estimator. The optimized EM abundance
workflow used in the manuscript is not packaged here. The `experimental-em`
feature contains older development code, not that timed implementation.

## License

[MIT](LICENSE).
