# Pre-Cached-Pivoted-QR Benchmark

This is the baseline before replacing streamed-Givens damped QR with cached,
column-pivoted Householder QR.

Command:

```sh
python3 scripts/benchmark.py \
  --solver-work \
  --iterations 30 \
  --build-dir build-external-ninja \
  --output-dir /tmp/levmar-work-timing
```

The benchmark used Clang Release builds with `-O3 -march=native -fno-fast-math`
and `-ffp-contract=off`. Timings are average elapsed milliseconds from
`/tmp/levmar-work-timing/nist-solver-work.csv`.

| Case | levmar | Ceres | CMinpack |
|---|---:|---:|---:|
| Gauss1/start2 analytic | 0.277 ms | 0.150 ms | 0.107 ms |
| MGH10/start1 analytic | 2.969 ms | 0.469 ms | 0.161 ms |
| MGH17/start1 analytic | 3.168 ms | 1.837 ms | 0.967 ms |
| Rat43/start1 analytic | 0.312 ms | 0.073 ms | 0.038 ms |

Key levmar work counts:

| Case | Iterations | Linear solves | Accepted | Rejected |
|---|---:|---:|---:|---:|
| Gauss1/start2 analytic | 6 | 6 | 6 | 0 |
| MGH10/start1 analytic | 435 | 3,539 | 339 | 95 |
| MGH17/start1 analytic | 87 | 958 | 56 | 30 |
| Rat43/start1 analytic | 26 | 271 | 18 | 7 |

MGH10/start1 is the primary cached-QR target: 435 nonlinear iterations require
3,539 streamed-Givens QR solves. The replacement should factor the tall scaled
Jacobian once per evaluated model and make lambda trials `O(n^2)`.
