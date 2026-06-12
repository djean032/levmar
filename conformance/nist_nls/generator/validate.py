def close_enough(actual, expected, atol=1e-9, rtol=1e-9):
    return abs(actual - expected) <= atol + rtol * abs(expected)


def validate_problem(problem, residuals, jacobians):
    m = problem["m"]
    n = problem["n"]

    if len(problem["data"]) != m:
        raise ValueError(f"{problem['name']}: data row count does not match m")

    for label, beta in problem["params"].items():
        if len(beta) != n:
            raise ValueError(f"{problem['name']} {label}: parameter count mismatch")
        if len(residuals[label]) != m:
            raise ValueError(f"{problem['name']} {label}: residual count mismatch")
        if len(jacobians[label]) != m:
            raise ValueError(f"{problem['name']} {label}: jacobian row count mismatch")
        for row in jacobians[label]:
            if len(row) != n:
                raise ValueError(f"{problem['name']} {label}: jacobian column count mismatch")

    certified_rss = sum(r * r for r in residuals["certified"])
    if not close_enough(certified_rss, problem["certified_rss"], atol=1e-8, rtol=1e-8):
        raise ValueError(
            f"{problem['name']}: certified RSS mismatch "
            f"got {certified_rss:.16e}, expected {problem['certified_rss']:.16e}"
        )
