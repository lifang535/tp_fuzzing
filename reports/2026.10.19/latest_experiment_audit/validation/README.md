# Validation results

Validation was performed on **2026-10-19**. Case labels and example filenames were normalized to the final report numbering after execution; numerical outputs, compiler diagnostics, exit codes and timestamps are unchanged. Original captures are retained in the local audit evidence archive.

- [Local environment](environment.json): RTX 4060 Laptop, TileLang 0.1.15, Triton 3.8.0, Python 3.10.20 and CUDA tools 12.9.86.
- [Four local compiler examples](local_runs.json): process exit status, duration and output path for each example. See the matching `.py.txt` files for numerical observations and diagnostics.
- [Fuzzer#4 paired controls](fusion_runs.json): default contraction exits 1 with the saved oracle failure; no-FMA exits 0.
- [TileLang#26 compilation controls](tl26_cuda124.json): six CPU-only compilations on the server's CUDA 12.4.131 installation, with full diagnostics and the installed atomic header's SHA-256.
- [Publication file hashes](publication_hashes.json): SHA-256 values for the final reports, published data, reproduction scripts and validation files other than this hash manifest.

The compiler-example scripts print their findings and normally exit 0 even when the compiler behavior is wrong. The bf16 pow exception is intentionally caught so its two successful controls can run. Use the observations, rather than a zero process status alone, to interpret a reproduction.

The real-bug catalogue remains the fixed **17:35** audit snapshot. These small validation runs verify selected examples; they do not add post-snapshot failures to the 43-mechanism count.
