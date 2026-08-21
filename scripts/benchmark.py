#!/usr/bin/env python3
"""Build, run, and report the NIST conformance benchmark."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
PORTABLE_FLAGS = "-O3 -march=native -fno-fast-math -ffp-contract=off"
TIMING_METRICS = (
    "analytic_residual_and_jacobian",
    "autodiff_residual_and_jacobian",
    "forward_difference_residual_and_jacobian",
    "central_difference_residual_and_jacobian",
)
SUBSET_METRICS = tuple(f"subset_{metric}" for metric in TIMING_METRICS)


def run(command, env_file=None, capture_output=False):
    print("+", " ".join(command))
    kwargs = {"check": True, "cwd": ROOT, "text": True}
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
    if env_file is None:
        return subprocess.run(command, **kwargs)
    return subprocess.run(
        ["bash", "-c", 'set -e; source "$1"; shift; exec "$@"', "bash",
         str(env_file), *command], **kwargs)


def read_csv(path):
    try:
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError as error:
        raise RuntimeError(f"cannot read {path}: {error}") from error


def format_ms(value):
    return f"{float(value):.4f}"


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def timing_rows(timings, metrics):
    return [[row["name"], row["metric"], row["iterations"],
              format_ms(row["dynamic_mean_ms"]),
              format_ms(row["dynamic_stddev_ms"]),
              format_ms(row["static_mean_ms"]),
              format_ms(row["static_stddev_ms"])]
             for row in timings if row["metric"] in metrics]


def summary_timing_rows(timings, metrics):
    summary = {row["metric"]: row for row in timings
               if row["scope"] == "summary"}
    analytic = summary[metrics[0]]
    rows = []
    for metric in metrics:
        row = summary[metric]
        dynamic = float(row["dynamic_mean_ms"])
        static = float(row["static_mean_ms"])
        rows.append([
            metric.removesuffix("_residual_and_jacobian").replace("_", " "),
            f"{dynamic:.4f} +/- {float(row['dynamic_stddev_ms']):.4f} ms",
            f"{dynamic / float(analytic['dynamic_mean_ms']):.2f}x",
            f"{static:.4f} +/- {float(row['static_stddev_ms']):.4f} ms",
            f"{static / float(analytic['static_mean_ms']):.2f}x",
        ])
    return rows


def accuracy_rows(validation, per_problem):
    selected = validation if per_problem else [row for row in validation
                                                if row["row_type"] == "summary"]
    result = []
    for row in selected:
        for metric in ("residual", "analytic", "autodiff",
                       "autodiff_vs_analytic", "forward_difference",
                       "central_difference"):
            if metric + "_count" not in row or int(row[metric + "_count"]) == 0:
                continue
            result.append([row["name"], metric, row[metric + "_count"],
                           row[metric + "_max_abs"], row[metric + "_max_rel"],
                           row[metric + "_rms_abs"], row[metric + "_rms_rel"]])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="clang++")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-benchmark")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "benchmark-results")
    parser.add_argument("--per-problem", action="store_true")
    parser.add_argument("--env", type=Path, help="shell setup file to source")
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.env and not args.env.is_file():
        parser.error(f"--env file does not exist: {args.env}")
    if not args.compiler:
        parser.error("--compiler must not be empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = args.output_dir / "nist-validation.csv"
    benchmark_path = args.output_dir / "nist-benchmark.csv"
    report_path = args.output_dir / "nist-report.md"

    try:
        run(["cmake", "-S", str(ROOT), "-B", str(args.build_dir),
             "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CXX_COMPILER={args.compiler}",
             f"-DLEVMAR_BENCHMARK_COMPILE_OPTIONS={PORTABLE_FLAGS}"], args.env)
        run(["cmake", "--build", str(args.build_dir), "--target",
             "levmar_nist_runner"], args.env)
        runner = args.build_dir / "levmar_nist_runner"
        if not runner.is_file():
            raise RuntimeError(f"benchmark runner was not built: {runner}")
        run([str(runner), str(ROOT / "conformance/nist_nls/corpus"),
             str(validation_path), str(args.iterations), str(benchmark_path)],
            args.env, capture_output=True)
        validation = read_csv(validation_path)
        timings = read_csv(benchmark_path)
        if not validation or not timings:
            raise RuntimeError("runner produced an empty CSV report")
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stdout:
            print(error.stdout, file=sys.stderr, end="")
        return 1

    timing_headers = ["Problem", "Method", "Iterations", "Dynamic mean ms",
                      "Dynamic stddev ms", "Static mean ms", "Static stddev ms"]
    summary_timing_headers = ["Method", "Dynamic", "vs analytic", "Static",
                              "vs analytic"]
    accuracy_headers = ["Problem", "Check", "Count", "Max abs", "Max rel",
                        "RMS abs", "RMS rel"]
    if args.per_problem:
        full_report = markdown_table(timing_headers,
                                     timing_rows(timings, TIMING_METRICS))
        subset_report = markdown_table(timing_headers,
                                       timing_rows(timings, SUBSET_METRICS))
    else:
        full_report = markdown_table(summary_timing_headers,
                                     summary_timing_rows(timings, TIMING_METRICS))
        subset_report = markdown_table(summary_timing_headers,
                                       summary_timing_rows(timings, SUBSET_METRICS))
    accuracy = accuracy_rows(validation, args.per_problem)
    report = "\n\n".join([
        "# NIST Benchmark Report",
        "## Residual + Jacobian Timing\n" + full_report,
        "## Numerical Subset Timing\n" + subset_report,
        "## Accuracy\n" + markdown_table(accuracy_headers, accuracy),
    ]) + "\n"
    report_path.write_text(report)
    print(report)
    print(f"reports written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
