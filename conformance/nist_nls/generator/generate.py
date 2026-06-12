from pathlib import Path
import shutil

from datasets import load_problems
from format import write_csv, write_meta
from models import MODELS
from validate import validate_problem


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"


def residuals(problem, beta):
    model, _ = MODELS[problem["model_id"]]
    return [model(beta, row) for row in problem["data"]]


def jacobian(problem, beta):
    _, jac = MODELS[problem["model_id"]]
    return [jac(beta, row) for row in problem["data"]]


def generate_problem(problem):
    out = CORPUS / problem["id"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    residual_sets = {}
    jacobian_sets = {}
    for label, beta in problem["params"].items():
        residual_sets[label] = residuals(problem, beta)
        jacobian_sets[label] = jacobian(problem, beta)

    validate_problem(problem, residual_sets, jacobian_sets)

    write_meta(
        out / "meta.txt",
        [
            ("name", problem["name"]),
            ("source", "nist_strd_nls"),
            ("source_url", problem["source_url"]),
            ("model_id", problem["model_id"]),
            ("model_class", problem["model_class"]),
            ("difficulty", problem["difficulty"]),
            ("m", problem["m"]),
            ("n", problem["n"]),
            ("predictor_count", problem["predictor_count"]),
            ("residual_definition", "model_value-observed_value"),
            ("certified_rss", f"{problem['certified_rss']:.16e}"),
            ("labels", ",".join(problem["params"].keys())),
            (
                "numerical_derivatives_recommended",
                "false" if problem["model_class"] == "rational" else "true",
            ),
        ],
    )

    data_header = [
        "x" if problem["predictor_count"] == 1 else f"x{i + 1}"
        for i in range(problem["predictor_count"])
    ] + ["y"]
    write_csv(out / "data.csv", data_header, problem["data"])
    write_csv(
        out / "params.csv",
        ["label"] + [f"b{i + 1}" for i in range(problem["n"])],
        [[label] + beta for label, beta in problem["params"].items()],
    )

    jac_header = [f"j{i + 1}" for i in range(problem["n"])]
    for label in problem["params"]:
        write_csv(out / f"residuals_{label}.csv", ["residual"], [[r] for r in residual_sets[label]])
        write_csv(out / f"jacobian_{label}.csv", jac_header, jacobian_sets[label])


def main():
    CORPUS.mkdir(parents=True, exist_ok=True)
    for problem in load_problems():
        generate_problem(problem)


if __name__ == "__main__":
    main()
