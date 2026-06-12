# NIST NLS Conformance Corpus Generator

This generator emits a checked-in corpus for residual and Jacobian conformance
tests across the C++, C99, and Fortran implementations.

Conventions:

- Residual definition: `r = model_value - observed_value`
- Some models, such as `Nelson`, use transformed observed values internally
  (for example `log(y)`) while preserving the same residual convention.
- Jacobian entries: `J[i,j] = d r_i / d beta_j`
- Jacobian CSV headers: `j1,j2,...`
- Floating-point output: 16-digit scientific notation
- Numerical derivatives are not recommended for rational models and are excluded
  from the conformance checks.

Regenerate the corpus from the repository root with:

```sh
python3 conformance/nist_nls/generator/generate.py
```

The generator fetches the NIST `.dat` files listed in the StRD NLS corpus,
parses starting values, certified values, certified RSS, and observations, and
combines them with local model formulas and analytic Jacobians to emit the
checked-in corpus.
