import math


def predictor(row, index=0):
    return row[index]


def observed(row):
    return row[-1]


def exp_term(amplitude, rate, x):
    e = math.exp(-rate * x)
    return amplitude * e, e


def monomolecular(beta, row):
    x = predictor(row)
    y = observed(row)
    b1, b2 = beta
    e = math.exp(-b2 * x)
    return b1 * (1.0 - e) - y


def monomolecular_jacobian(beta, row):
    x = predictor(row)
    b1, b2 = beta
    e = math.exp(-b2 * x)
    return [1.0 - e, b1 * x * e]


def chwirut(beta, row):
    x = predictor(row)
    y = observed(row)
    b1, b2, b3 = beta
    e = math.exp(-b1 * x)
    d = b2 + b3 * x
    return e / d - y


def chwirut_jacobian(beta, row):
    x = predictor(row)
    b1, b2, b3 = beta
    e = math.exp(-b1 * x)
    d = b2 + b3 * x
    d2 = d * d
    return [-x * e / d, -e / d2, -x * e / d2]


def triple_exponential(beta, row):
    x = predictor(row)
    y = observed(row)
    value = 0.0
    for i in range(0, 6, 2):
        value += beta[i] * math.exp(-beta[i + 1] * x)
    return value - y


def triple_exponential_jacobian(beta, row):
    x = predictor(row)
    values = []
    for i in range(0, 6, 2):
        e = math.exp(-beta[i + 1] * x)
        values.append(e)
        values.append(-beta[i] * x * e)
    return values


def gauss_mixture(beta, row):
    x = predictor(row)
    y = observed(row)
    b1, b2, b3, b4, b5, b6, b7, b8 = beta
    e1 = math.exp(-b2 * x)
    d1 = x - b4
    d2 = x - b7
    e2 = math.exp(-(d1 * d1) / (b5 * b5))
    e3 = math.exp(-(d2 * d2) / (b8 * b8))
    return b1 * e1 + b3 * e2 + b6 * e3 - y


def gauss_mixture_jacobian(beta, row):
    x = predictor(row)
    b1, b2, b3, b4, b5, b6, b7, b8 = beta
    e1 = math.exp(-b2 * x)
    d1 = x - b4
    d2 = x - b7
    s1 = b5 * b5
    s2 = b8 * b8
    e2 = math.exp(-(d1 * d1) / s1)
    e3 = math.exp(-(d2 * d2) / s2)
    return [
        e1,
        -b1 * x * e1,
        e2,
        b3 * e2 * 2.0 * d1 / s1,
        b3 * e2 * 2.0 * d1 * d1 / (b5 * b5 * b5),
        e3,
        b6 * e3 * 2.0 * d2 / s2,
        b6 * e3 * 2.0 * d2 * d2 / (b8 * b8 * b8),
    ]


def danwood(beta, row):
    x = predictor(row)
    y = observed(row)
    return beta[0] * math.pow(x, beta[1]) - y


def danwood_jacobian(beta, row):
    x = predictor(row)
    x_pow = math.pow(x, beta[1])
    return [x_pow, beta[0] * x_pow * math.log(x)]


def misra1b(beta, row):
    x = predictor(row)
    y = observed(row)
    b1, b2 = beta
    t = 1.0 + b2 * x / 2.0
    return b1 * (1.0 - math.pow(t, -2.0)) - y


def misra1b_jacobian(beta, row):
    x = predictor(row)
    b1, b2 = beta
    t = 1.0 + b2 * x / 2.0
    return [1.0 - math.pow(t, -2.0), b1 * x * math.pow(t, -3.0)]


def rational_quadratic(beta, row):
    x = predictor(row)
    y = observed(row)
    x2 = x * x
    n = beta[0] + beta[1] * x + beta[2] * x2
    d = 1.0 + beta[3] * x + beta[4] * x2
    return n / d - y


def rational_quadratic_jacobian(beta, row):
    x = predictor(row)
    x2 = x * x
    n = beta[0] + beta[1] * x + beta[2] * x2
    d = 1.0 + beta[3] * x + beta[4] * x2
    d2 = d * d
    return [1.0 / d, x / d, x2 / d, -n * x / d2, -n * x2 / d2]


def rational_cubic(beta, row):
    x = predictor(row)
    y = observed(row)
    x2 = x * x
    x3 = x2 * x
    n = beta[0] + beta[1] * x + beta[2] * x2 + beta[3] * x3
    d = 1.0 + beta[4] * x + beta[5] * x2 + beta[6] * x3
    return n / d - y


def rational_cubic_jacobian(beta, row):
    x = predictor(row)
    x2 = x * x
    x3 = x2 * x
    n = beta[0] + beta[1] * x + beta[2] * x2 + beta[3] * x3
    d = 1.0 + beta[4] * x + beta[5] * x2 + beta[6] * x3
    d2 = d * d
    return [
        1.0 / d,
        x / d,
        x2 / d,
        x3 / d,
        -n * x / d2,
        -n * x2 / d2,
        -n * x3 / d2,
    ]


def nelson_log(beta, row):
    x1 = predictor(row, 0)
    x2 = predictor(row, 1)
    y = observed(row)
    return beta[0] - beta[1] * x1 * math.exp(-beta[2] * x2) - math.log(y)


def nelson_log_jacobian(beta, row):
    x1 = predictor(row, 0)
    x2 = predictor(row, 1)
    e = math.exp(-beta[2] * x2)
    return [1.0, -x1 * e, beta[1] * x1 * x2 * e]


def mgh17(beta, row):
    x = predictor(row)
    y = observed(row)
    return beta[0] + beta[1] * math.exp(-beta[3] * x) + beta[2] * math.exp(-beta[4] * x) - y


def mgh17_jacobian(beta, row):
    x = predictor(row)
    e1 = math.exp(-beta[3] * x)
    e2 = math.exp(-beta[4] * x)
    return [1.0, e1, e2, -beta[1] * x * e1, -beta[2] * x * e2]


def misra1c(beta, row):
    x = predictor(row)
    y = observed(row)
    b1, b2 = beta
    t = 1.0 + 2.0 * b2 * x
    return b1 * (1.0 - 1.0 / math.sqrt(t)) - y


def misra1c_jacobian(beta, row):
    x = predictor(row)
    b1, b2 = beta
    t = 1.0 + 2.0 * b2 * x
    return [1.0 - 1.0 / math.sqrt(t), b1 * x * math.pow(t, -1.5)]


def misra1d(beta, row):
    x = predictor(row)
    y = observed(row)
    b1, b2 = beta
    d = 1.0 + b2 * x
    return (b1 * b2 * x) / d - y


def misra1d_jacobian(beta, row):
    x = predictor(row)
    b1, b2 = beta
    d = 1.0 + b2 * x
    d2 = d * d
    return [b2 * x / d, b1 * x / d2]


def roszman1(beta, row):
    x = predictor(row)
    y = observed(row)
    return beta[0] - beta[1] * x - math.atan(beta[2] / (x - beta[3])) / math.pi - y


def roszman1_jacobian(beta, row):
    x = predictor(row)
    u = beta[2] / (x - beta[3])
    common = -1.0 / (math.pi * (1.0 + u * u))
    return [1.0, -x, common / (x - beta[3]), common * beta[2] / ((x - beta[3]) * (x - beta[3]))]


def enso(beta, row):
    x = predictor(row)
    y = observed(row)
    annual = 2.0 * math.pi * x / 12.0
    p4 = 2.0 * math.pi * x / beta[3]
    p7 = 2.0 * math.pi * x / beta[6]
    value = (
        beta[0]
        + beta[1] * math.cos(annual)
        + beta[2] * math.sin(annual)
        + beta[4] * math.cos(p4)
        + beta[5] * math.sin(p4)
        + beta[7] * math.cos(p7)
        + beta[8] * math.sin(p7)
    )
    return value - y


def enso_jacobian(beta, row):
    x = predictor(row)
    annual = 2.0 * math.pi * x / 12.0
    p4 = 2.0 * math.pi * x / beta[3]
    p7 = 2.0 * math.pi * x / beta[6]
    d4 = 2.0 * math.pi * x / (beta[3] * beta[3])
    d7 = 2.0 * math.pi * x / (beta[6] * beta[6])
    return [
        1.0,
        math.cos(annual),
        math.sin(annual),
        d4 * (beta[4] * math.sin(p4) - beta[5] * math.cos(p4)),
        math.cos(p4),
        math.sin(p4),
        d7 * (beta[7] * math.sin(p7) - beta[8] * math.cos(p7)),
        math.cos(p7),
        math.sin(p7),
    ]


def mgh09(beta, row):
    x = predictor(row)
    y = observed(row)
    n = x * x + beta[1] * x
    d = x * x + beta[2] * x + beta[3]
    return beta[0] * n / d - y


def mgh09_jacobian(beta, row):
    x = predictor(row)
    n = x * x + beta[1] * x
    d = x * x + beta[2] * x + beta[3]
    d2 = d * d
    return [n / d, beta[0] * x / d, -beta[0] * n * x / d2, -beta[0] * n / d2]


def rat42(beta, row):
    x = predictor(row)
    y = observed(row)
    e = math.exp(beta[1] - beta[2] * x)
    return beta[0] / (1.0 + e) - y


def rat42_jacobian(beta, row):
    x = predictor(row)
    e = math.exp(beta[1] - beta[2] * x)
    d = 1.0 + e
    d2 = d * d
    return [1.0 / d, -beta[0] * e / d2, beta[0] * x * e / d2]


def mgh10(beta, row):
    x = predictor(row)
    y = observed(row)
    e = math.exp(beta[1] / (x + beta[2]))
    return beta[0] * e - y


def mgh10_jacobian(beta, row):
    x = predictor(row)
    q = x + beta[2]
    e = math.exp(beta[1] / q)
    return [e, beta[0] * e / q, -beta[0] * e * beta[1] / (q * q)]


def eckerle4(beta, row):
    x = predictor(row)
    y = observed(row)
    d = beta[1]
    delta = x - beta[2]
    e = math.exp(-(delta * delta) / (2.0 * d * d))
    return (beta[0] / d) * e - y


def eckerle4_jacobian(beta, row):
    x = predictor(row)
    b1, b2, b3 = beta
    delta = x - b3
    e = math.exp(-(delta * delta) / (2.0 * b2 * b2))
    return [
        e / b2,
        b1 * e * (delta * delta / (b2 * b2 * b2 * b2) - 1.0 / (b2 * b2)),
        b1 * e * delta / (b2 * b2 * b2),
    ]


def rat43(beta, row):
    x = predictor(row)
    y = observed(row)
    e = math.exp(beta[1] - beta[2] * x)
    t = 1.0 + e
    p = math.pow(t, -1.0 / beta[3])
    return beta[0] * p - y


def rat43_jacobian(beta, row):
    x = predictor(row)
    b1, b2, b3, b4 = beta
    e = math.exp(b2 - b3 * x)
    t = 1.0 + e
    p = math.pow(t, -1.0 / b4)
    yhat = b1 * p
    return [
        p,
        -yhat * e / (b4 * t),
        yhat * x * e / (b4 * t),
        yhat * math.log(t) / (b4 * b4),
    ]


def bennett5(beta, row):
    x = predictor(row)
    y = observed(row)
    return beta[0] * math.pow(beta[1] + x, -1.0 / beta[2]) - y


def bennett5_jacobian(beta, row):
    x = predictor(row)
    b1, b2, b3 = beta
    t = b2 + x
    p = math.pow(t, -1.0 / b3)
    return [p, -b1 * p / (b3 * t), b1 * p * math.log(t) / (b3 * b3)]


MODELS = {
    "monomolecular": (monomolecular, monomolecular_jacobian),
    "chwirut": (chwirut, chwirut_jacobian),
    "triple_exponential": (triple_exponential, triple_exponential_jacobian),
    "gauss_mixture": (gauss_mixture, gauss_mixture_jacobian),
    "danwood": (danwood, danwood_jacobian),
    "misra1b": (misra1b, misra1b_jacobian),
    "rational_quadratic": (rational_quadratic, rational_quadratic_jacobian),
    "rational_cubic": (rational_cubic, rational_cubic_jacobian),
    "nelson_log": (nelson_log, nelson_log_jacobian),
    "mgh17": (mgh17, mgh17_jacobian),
    "misra1c": (misra1c, misra1c_jacobian),
    "misra1d": (misra1d, misra1d_jacobian),
    "roszman1": (roszman1, roszman1_jacobian),
    "enso": (enso, enso_jacobian),
    "mgh09": (mgh09, mgh09_jacobian),
    "rat42": (rat42, rat42_jacobian),
    "mgh10": (mgh10, mgh10_jacobian),
    "eckerle4": (eckerle4, eckerle4_jacobian),
    "rat43": (rat43, rat43_jacobian),
    "bennett5": (bennett5, bennett5_jacobian),
}
