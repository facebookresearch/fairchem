# fairchem-core standalone scripts

This directory is the single home for standalone, checkout-run scripts
for `fairchem-core`: run them from a repository checkout, e.g.
`python packages/fairchem-core/scripts/preprocess_ef.py ...`. Nothing
here is shipped in the `fairchem-core` wheel.

The rule for where a script lives:

- **Run from a checkout by a human?** Put it here. Benchmark and
  reproduction scripts go under `benchmarks/` (e.g. `benchmarks/omol/`).
- **Imported by shipped code, or run from an installed environment**
  (`python -m ...`)? It is library surface, not a script: it belongs in
  `src/fairchem/core/` (utilities in `src/fairchem/core/common/`,
  supported entry points in `src/fairchem/core/scripts/`).

Other distributions follow the same pattern with a `scripts/`
directory at their own package root.
