#!/usr/bin/env python3
"""Compute pooled PSNR and capture-group uncertainty from stored fold reports."""

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


CONFIGS = (("residual", 16), ("direct", 12), ("residual", 21), ("direct", 16))
SEEDS = (42, 43)


def pooled_psnr(rows: list[dict], field: str) -> float:
    total_error = 0.0
    total_channels = 0
    for row in rows:
        channels = row["width"] * row["height"] * 3
        total_error += 10 ** (-row[field] / 10) * channels
        total_channels += channels
    return -10 * math.log10(total_error / total_channels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("work_root", type=Path)
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text())["images"]
    if len(records) != 174:
        raise ValueError("training manifest has changed")
    baseline_rows = []
    model_rows = defaultdict(list)
    paired = defaultdict(list)
    for fold in range(5):
        baseline = json.loads((args.baseline_root / f"qse-fold{fold}-baselines.json").read_text())
        baseline_rows.extend(baseline["images"])
        for seed in SEEDS:
            for kind, width in CONFIGS:
                report = json.loads((args.work_root / f"fold{fold}-seed{seed}-{kind}{width}.json").read_text())
                model_rows[(kind, width)].extend(report["images"])
            for size, res_width, direct_width in (("~500", 16, 12), ("~650", 21, 16)):
                res = json.loads((args.work_root / f"fold{fold}-seed{seed}-residual{res_width}.json").read_text())
                direct = json.loads((args.work_root / f"fold{fold}-seed{seed}-direct{direct_width}.json").read_text())
                res_by_index = {int(Path(row["path"]).name.split("-", 1)[0]): row for row in res["images"]}
                for row in direct["images"]:
                    index = int(Path(row["path"]).name.split("-", 1)[0])
                    if records[index]["fold"] != fold:
                        raise ValueError("fold assignment does not match report")
                    delta = row["model_psnr_db"] - res_by_index[index]["model_psnr_db"]
                    paired[size].append((records[index]["group"], delta))
    if len(baseline_rows) != 174 or any(len(rows) != 348 for rows in model_rows.values()):
        raise ValueError("report counts are incomplete")
    result = {
        "baseline_pooled_psnr_db": {
            "nearest": pooled_psnr(baseline_rows, "nearest_psnr_db"),
            "bicubic": pooled_psnr(baseline_rows, "bicubic_psnr_db"),
        },
        "models": [{"kind": kind, "hidden": width,
                    "pooled_psnr_db": pooled_psnr(model_rows[(kind, width)], "model_psnr_db"),
                    "mean_image_psnr_db": statistics.mean(row["model_psnr_db"] for row in model_rows[(kind, width)])}
                   for kind, width in CONFIGS],
        "matched_pairs": [],
    }
    rng = random.Random(20260924)
    for size in ("~500", "~650"):
        groups = defaultdict(list)
        for group, delta in paired[size]:
            groups[group].append(delta)
        group_ids = sorted(groups)
        samples = []
        for _ in range(10_000):
            selected = [groups[group_ids[rng.randrange(len(group_ids))]]
                        for _ in group_ids]
            samples.append(statistics.mean(delta for group in selected for delta in group))
        samples.sort()
        result["matched_pairs"].append({
            "size": size,
            "mean_direct_minus_residual_db": statistics.mean(delta for _, delta in paired[size]),
            "bootstrap_group_95pct_db": [samples[249], samples[9749]],
            "capture_groups": len(groups),
            "groups_direct_better": sum(statistics.mean(groups[group]) > 0 for group in group_ids),
            "image_seed_comparisons": len(paired[size]),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
