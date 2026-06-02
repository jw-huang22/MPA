import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def format_number(value):
    if isinstance(value, float):
        return f"{value:.2e}"
    return value


def rank_sort_value(rank):
    try:
        return int(rank)
    except (TypeError, ValueError):
        return -1


def summarize(input_path, merge_datasets=False, group_by_rank="auto"):
    groups = defaultdict(lambda: {"adp_sum": 0.0, "adp_count": 0, "ratio_sum": 0.0, "ratio_count": 0, "adp_over_ratio_sum": 0.0, "adp_over_ratio_count": 0})
    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)
        has_rank = reader.fieldnames is not None and "rank_r" in reader.fieldnames
        if group_by_rank == "auto":
            include_rank = has_rank
        else:
            include_rank = group_by_rank == "always" and has_rank
        for row in reader:
            if merge_datasets:
                key = (row["model"], row["obfus"])
            else:
                key = (row["model"], row["dataset"], row["obfus"])
            if include_rank:
                key = (*key, row.get("rank_r", ""))
            adp = parse_float(row.get("adp"))
            ratio = parse_float(row.get("k_over_nm"))
            # if adp is not None and not 0 <= adp <= 1:
            #     adp = None
            if adp is not None:
                groups[key]["adp_sum"] += adp
                groups[key]["adp_count"] += 1
            if ratio is not None:
                groups[key]["ratio_sum"] += ratio
                groups[key]["ratio_count"] += 1
            if adp is not None and ratio is not None and ratio != 0:
                groups[key]["adp_over_ratio_sum"] += adp / ratio
                groups[key]["adp_over_ratio_count"] += 1
    rows = []
    def sort_key(item):
        key, _ = item
        rank = key[-1] if include_rank else None
        prefix = key[:-1] if include_rank else key
        return (*prefix, rank_sort_value(rank))

    for key, stat in sorted(groups.items(), key=sort_key):
        if merge_datasets:
            if include_rank:
                model, obfus, rank_r = key
            else:
                model, obfus = key
                rank_r = None
            dataset = "all"
        else:
            if include_rank:
                model, dataset, obfus, rank_r = key
            else:
                model, dataset, obfus = key
                rank_r = None
        adp_mean = stat["adp_sum"] / stat["adp_count"] if stat["adp_count"] else ""
        ratio_mean = stat["ratio_sum"] / stat["ratio_count"] if stat["ratio_count"] else ""
        adp_over_ratio_mean = stat["adp_over_ratio_sum"] / stat["adp_over_ratio_count"] if stat["adp_over_ratio_count"] else ""
        out = {
            "model": model,
            "dataset": dataset,
            "obfus": obfus,
            "num_adp_layers": stat["adp_count"],
            "num_ratio_layers": stat["ratio_count"],
            "adp_mean": format_number(adp_mean),
            "k_over_nm_mean": format_number(ratio_mean),
            "adp_over_k_over_nm_mean": format_number(adp_over_ratio_mean),
            "adp_mean_over_k_over_nm_mean": format_number(adp_mean / ratio_mean) if ratio_mean not in (0, "", None) else "",
        }
        if include_rank:
            out["rank_r"] = rank_r
        rows.append(out)
    return rows


def write_rows(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "dataset",
        "obfus",
    ]
    if any("rank_r" in row for row in rows):
        fieldnames.append("rank_r")
    fieldnames.extend(
        [
            "num_adp_layers",
            "num_ratio_layers",
            "adp_mean",
            "k_over_nm_mean",
            "adp_over_k_over_nm_mean",
            "adp_mean_over_k_over_nm_mean",
        ]
    )
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    if not rows:
        print("No ADP rows found.")
        return
    headers = ["model", "dataset", "obfus"]
    if any("rank_r" in row for row in rows):
        headers.append("rank_r")
    headers.extend(["num_adp_layers", "adp_mean", "k_over_nm_mean", "adp_over_k_over_nm_mean", "adp_mean_over_k_over_nm_mean"])
    widths = {h: len(h) for h in headers}
    printable = []
    for row in rows:
        item = {}
        for h in headers:
            value = str(row[h])
            item[h] = value
            widths[h] = max(widths[h], len(value))
        printable.append(item)
    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in printable:
        print("  ".join(row[h].ljust(widths[h]) for h in headers))


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize ADP CSV by model and obfuscation method.")
    parser.add_argument("--input", default="results/adp_results/adp.csv")
    parser.add_argument("--output", default="results/adp_results/adp_summary_by_model_obfus.csv")
    parser.add_argument("--merge_datasets", action="store_true", help="Group only by model and obfuscation method.")
    parser.add_argument(
        "--group_by_rank",
        choices=["auto", "always", "never"],
        default="auto",
        help="Group by rank_r when the input CSV contains it.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = summarize(input_path, merge_datasets=args.merge_datasets, group_by_rank=args.group_by_rank)
    write_rows(output_path, rows)
    print_table(rows)
    print(f"\nWrote summary to {output_path}")


if __name__ == "__main__":
    main()
