import csv


def float_text(value):
    return f"{value:.16e}"


def write_meta(path, items):
    with path.open("w", encoding="utf-8", newline="") as file:
        for key, value in items:
            file.write(f"{key}={value}\n")


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in rows:
            writer.writerow([
                float_text(value) if isinstance(value, float) else value
                for value in row
            ])
