from collections import OrderedDict
import re
import urllib.request


NIST_BASE = "https://www.itl.nist.gov/div898/strd/nls/data"
NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][-+]?\d+)?")


PROBLEM_SPECS = [
    {
        "id": "misra1a",
        "name": "Misra1a",
        "source_url": f"{NIST_BASE}/misra1a.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Misra1a.dat",
        "model_id": "monomolecular",
        "model_class": "exponential",
        "difficulty": "lower",
        "n": 2,
        "predictor_count": 1,
    },
    {
        "id": "chwirut2",
        "name": "Chwirut2",
        "source_url": f"{NIST_BASE}/chwirut2.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Chwirut2.dat",
        "model_id": "chwirut",
        "model_class": "exponential",
        "difficulty": "lower",
        "n": 3,
        "predictor_count": 1,
    },
    {
        "id": "chwirut1",
        "name": "Chwirut1",
        "source_url": f"{NIST_BASE}/chwirut1.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Chwirut1.dat",
        "model_id": "chwirut",
        "model_class": "exponential",
        "difficulty": "lower",
        "n": 3,
        "predictor_count": 1,
    },
    {
        "id": "lanczos3",
        "name": "Lanczos3",
        "source_url": f"{NIST_BASE}/lanczos3.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Lanczos3.dat",
        "model_id": "triple_exponential",
        "model_class": "exponential",
        "difficulty": "lower",
        "n": 6,
        "predictor_count": 1,
    },
    {
        "id": "gauss1",
        "name": "Gauss1",
        "source_url": f"{NIST_BASE}/gauss1.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Gauss1.dat",
        "model_id": "gauss_mixture",
        "model_class": "exponential",
        "difficulty": "lower",
        "n": 8,
        "predictor_count": 1,
    },
    {
        "id": "gauss2",
        "name": "Gauss2",
        "source_url": f"{NIST_BASE}/gauss2.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Gauss2.dat",
        "model_id": "gauss_mixture",
        "model_class": "exponential",
        "difficulty": "lower",
        "n": 8,
        "predictor_count": 1,
    },
    {
        "id": "danwood",
        "name": "DanWood",
        "source_url": f"{NIST_BASE}/daniel_wood.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/DanWood.dat",
        "model_id": "danwood",
        "model_class": "miscellaneous",
        "difficulty": "lower",
        "n": 2,
        "predictor_count": 1,
    },
    {
        "id": "misra1b",
        "name": "Misra1b",
        "source_url": f"{NIST_BASE}/misra1b.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Misra1b.dat",
        "model_id": "misra1b",
        "model_class": "miscellaneous",
        "difficulty": "lower",
        "n": 2,
        "predictor_count": 1,
    },
    {
        "id": "kirby2",
        "name": "Kirby2",
        "source_url": f"{NIST_BASE}/kirby2.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Kirby2.dat",
        "model_id": "rational_quadratic",
        "model_class": "rational",
        "difficulty": "average",
        "n": 5,
        "predictor_count": 1,
    },
    {
        "id": "hahn1",
        "name": "Hahn1",
        "source_url": f"{NIST_BASE}/hahn1.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Hahn1.dat",
        "model_id": "rational_cubic",
        "model_class": "rational",
        "difficulty": "average",
        "n": 7,
        "predictor_count": 1,
    },
    {
        "id": "nelson",
        "name": "Nelson",
        "source_url": f"{NIST_BASE}/nelson.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Nelson.dat",
        "model_id": "nelson_log",
        "model_class": "exponential",
        "difficulty": "average",
        "n": 3,
        "predictor_count": 2,
    },
    {
        "id": "mgh17",
        "name": "MGH17",
        "source_url": f"{NIST_BASE}/mgh17.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/MGH17.dat",
        "model_id": "mgh17",
        "model_class": "exponential",
        "difficulty": "average",
        "n": 5,
        "predictor_count": 1,
    },
    {
        "id": "lanczos1",
        "name": "Lanczos1",
        "source_url": f"{NIST_BASE}/lanczos1.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Lanczos1.dat",
        "model_id": "triple_exponential",
        "model_class": "exponential",
        "difficulty": "average",
        "n": 6,
        "predictor_count": 1,
    },
    {
        "id": "lanczos2",
        "name": "Lanczos2",
        "source_url": f"{NIST_BASE}/lanczos2.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Lanczos2.dat",
        "model_id": "triple_exponential",
        "model_class": "exponential",
        "difficulty": "average",
        "n": 6,
        "predictor_count": 1,
    },
    {
        "id": "gauss3",
        "name": "Gauss3",
        "source_url": f"{NIST_BASE}/gauss3.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Gauss3.dat",
        "model_id": "gauss_mixture",
        "model_class": "exponential",
        "difficulty": "average",
        "n": 8,
        "predictor_count": 1,
    },
    {
        "id": "misra1c",
        "name": "Misra1c",
        "source_url": f"{NIST_BASE}/misra1c.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Misra1c.dat",
        "model_id": "misra1c",
        "model_class": "miscellaneous",
        "difficulty": "average",
        "n": 2,
        "predictor_count": 1,
    },
    {
        "id": "misra1d",
        "name": "Misra1d",
        "source_url": f"{NIST_BASE}/misra1d.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Misra1d.dat",
        "model_id": "misra1d",
        "model_class": "miscellaneous",
        "difficulty": "average",
        "n": 2,
        "predictor_count": 1,
    },
    {
        "id": "roszman1",
        "name": "Roszman1",
        "source_url": f"{NIST_BASE}/roszman1.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Roszman1.dat",
        "model_id": "roszman1",
        "model_class": "miscellaneous",
        "difficulty": "average",
        "n": 4,
        "predictor_count": 1,
    },
    {
        "id": "enso",
        "name": "ENSO",
        "source_url": f"{NIST_BASE}/enso.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/ENSO.dat",
        "model_id": "enso",
        "model_class": "miscellaneous",
        "difficulty": "average",
        "n": 9,
        "predictor_count": 1,
    },
    {
        "id": "mgh09",
        "name": "MGH09",
        "source_url": f"{NIST_BASE}/mgh09.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/MGH09.dat",
        "model_id": "mgh09",
        "model_class": "rational",
        "difficulty": "higher",
        "n": 4,
        "predictor_count": 1,
    },
    {
        "id": "thurber",
        "name": "Thurber",
        "source_url": f"{NIST_BASE}/thurber.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Thurber.dat",
        "model_id": "rational_cubic",
        "model_class": "rational",
        "difficulty": "higher",
        "n": 7,
        "predictor_count": 1,
    },
    {
        "id": "boxbod",
        "name": "BoxBOD",
        "source_url": f"{NIST_BASE}/boxbod.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/BoxBOD.dat",
        "model_id": "monomolecular",
        "model_class": "exponential",
        "difficulty": "higher",
        "n": 2,
        "predictor_count": 1,
    },
    {
        "id": "rat42",
        "name": "Rat42",
        "source_url": f"{NIST_BASE}/ratkowsky2.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Rat42.dat",
        "model_id": "rat42",
        "model_class": "exponential",
        "difficulty": "higher",
        "n": 3,
        "predictor_count": 1,
    },
    {
        "id": "mgh10",
        "name": "MGH10",
        "source_url": f"{NIST_BASE}/mgh10.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/MGH10.dat",
        "model_id": "mgh10",
        "model_class": "exponential",
        "difficulty": "higher",
        "n": 3,
        "predictor_count": 1,
    },
    {
        "id": "eckerle4",
        "name": "Eckerle4",
        "source_url": f"{NIST_BASE}/eckerle4.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Eckerle4.dat",
        "model_id": "eckerle4",
        "model_class": "exponential",
        "difficulty": "higher",
        "n": 3,
        "predictor_count": 1,
    },
    {
        "id": "rat43",
        "name": "Rat43",
        "source_url": f"{NIST_BASE}/ratkowsky3.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Rat43.dat",
        "model_id": "rat43",
        "model_class": "exponential",
        "difficulty": "higher",
        "n": 4,
        "predictor_count": 1,
    },
    {
        "id": "bennett5",
        "name": "Bennett5",
        "source_url": f"{NIST_BASE}/bennett5.shtml",
        "data_file_url": f"{NIST_BASE}/LINKS/DATA/Bennett5.dat",
        "model_id": "bennett5",
        "model_class": "miscellaneous",
        "difficulty": "higher",
        "n": 3,
        "predictor_count": 1,
    },
]


def fetch_text(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8")


def parse_parameter_sets(text, count):
    params = OrderedDict((("start1", []), ("start2", []), ("certified", [])))
    found = 0
    for line in text.splitlines():
        if not line.lstrip().startswith("b"):
            continue
        values = [float(match) for match in NUMBER_RE.findall(line)]
        if len(values) < 4:
            continue
        params["start1"].append(values[1])
        params["start2"].append(values[2])
        params["certified"].append(values[3])
        found += 1
        if found == count:
            break

    if found != count:
        raise ValueError(f"Expected {count} parameter rows, found {found}")

    return params


def parse_certified_rss(text):
    for line in text.splitlines():
        if line.startswith("Residual Sum of Squares"):
            values = [float(match) for match in NUMBER_RE.findall(line)]
            if not values:
                break
            return values[0]
    raise ValueError("Could not parse certified RSS")


def parse_data_rows(text, predictor_count):
    rows = []
    reading = False
    expected_count = predictor_count + 1

    for line in text.splitlines():
        stripped = line.strip()
        if not reading:
            if (
                stripped.startswith("Data:")
                and "y" in stripped
                and "x" in stripped
                and "=" not in stripped
                and "Response" not in stripped
            ):
                reading = True
            continue

        values = [float(match) for match in NUMBER_RE.findall(line)]
        if not values:
            continue
        if len(values) != expected_count:
            continue
        y = values[0]
        predictors = values[1:]
        rows.append(predictors + [y])

    if not rows:
        raise ValueError("Could not parse data rows")

    return rows


def load_problem(spec):
    text = fetch_text(spec["data_file_url"])
    params = parse_parameter_sets(text, spec["n"])
    data = parse_data_rows(text, spec["predictor_count"])

    return {
        **spec,
        "m": len(data),
        "params": params,
        "certified_rss": parse_certified_rss(text),
        "data": data,
    }


def load_problems():
    return [load_problem(spec) for spec in PROBLEM_SPECS]
