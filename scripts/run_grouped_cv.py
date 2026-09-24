#!/usr/bin/env python3
"""Run the frozen five-fold, two-seed, parameter-matched image comparison."""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path


CONFIGS = (("residual", 16), ("direct", 12), ("residual", 21), ("direct", 16))
SEEDS = (42, 43)
EPOCHS = 24
TARGET_PIXELS_PER_EPOCH = 800_000
QUALITY = 70
LEARNING_RATE = 0.0003


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("fold_root", type=Path)
    parser.add_argument("work_root", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = manifest["images"]
    if len(records) != 174 or manifest["folds"] != 5:
        raise ValueError("this protocol requires the frozen 174-image, five-fold manifest")
    for record in records:
        path = args.data_root / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError(f"image changed: {path}")
    if len({record["sha256"] for record in records}) != len(records):
        raise ValueError("duplicate source image")
    args.work_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for fold in range(5):
        train_dir = args.fold_root / f"fold-{fold}" / "train"
        eval_dir = args.fold_root / f"fold-{fold}" / "eval"
        expected = sum(record["fold"] == fold for record in records)
        if len(list(eval_dir.iterdir())) != expected or len(list(train_dir.iterdir())) != 174 - expected:
            raise ValueError("staged fold size differs from manifest")
        for seed in SEEDS:
            for kind, hidden in CONFIGS:
                name = f"fold{fold}-seed{seed}-{kind}{hidden}"
                model = args.work_root / f"{name}.qsm"
                report = args.work_root / f"{name}.json"
                if not model.exists():
                    print(f"TRAIN {name}", flush=True)
                    subprocess.run(
                        [str(args.binary), "train-matched", kind, str(hidden), str(train_dir),
                         str(model), str(EPOCHS), str(TARGET_PIXELS_PER_EPOCH), str(QUALITY),
                         str(seed), str(LEARNING_RATE)],
                        check=True,
                        stdout=(args.work_root / f"{name}.log").open("w"),
                    )
                if not report.exists():
                    print(f"EVAL  {name}", flush=True)
                    subprocess.run(
                        [str(args.binary), "eval-matched", str(model), str(eval_dir), str(report),
                         str(QUALITY)], check=True, stdout=subprocess.DEVNULL,
                    )
                value = json.loads(report.read_text())
                if value["kind"] != kind or value["hidden"] != hidden or len(value["images"]) != expected:
                    raise ValueError(f"invalid report: {report}")
                reports.append({"fold": fold, "seed": seed, "kind": kind, "hidden": hidden,
                                "data": value})
                print(f"DONE  {name} {value['model_psnr_db']:.3f} dB", flush=True)
    summaries = []
    for kind, hidden in CONFIGS:
        subset = [item for item in reports if (item["kind"], item["hidden"]) == (kind, hidden)]
        per_image = [entry for item in subset for entry in item["data"]["images"]]
        summaries.append({
            "kind": kind, "hidden": hidden,
            "parameters": subset[0]["data"]["parameters"],
            "model_bytes": subset[0]["data"]["model_bytes"],
            "mean_image_psnr_db": statistics.mean(x["model_psnr_db"] for x in per_image),
            "mean_image_bicubic_psnr_db": statistics.mean(x["bicubic_psnr_db"] for x in per_image),
            "mean_image_gain_db": statistics.mean(x["model_psnr_db"] - x["bicubic_psnr_db"] for x in per_image),
            "fold_seed_psnr_db": [item["data"]["model_psnr_db"] for item in subset],
        })
    pairs = []
    for size, residual_width, direct_width in (("~500", 16, 12), ("~650", 21, 16)):
        differences = []
        units = []
        for fold in range(5):
            for seed in SEEDS:
                res = next(item for item in reports if (item["fold"], item["seed"], item["kind"], item["hidden"]) == (fold, seed, "residual", residual_width))
                direct = next(item for item in reports if (item["fold"], item["seed"], item["kind"], item["hidden"]) == (fold, seed, "direct", direct_width))
                by_path = {Path(x["path"]).name: x for x in res["data"]["images"]}
                ds = [x["model_psnr_db"] - by_path[Path(x["path"]).name]["model_psnr_db"] for x in direct["data"]["images"]]
                differences.extend(ds)
                units.append({"fold": fold, "seed": seed, "mean_direct_minus_residual_db": statistics.mean(ds)})
        pairs.append({"size": size, "residual_width": residual_width, "direct_width": direct_width,
                      "mean_direct_minus_residual_db": statistics.mean(differences),
                      "median_direct_minus_residual_db": statistics.median(differences),
                      "image_seed_comparisons": len(differences), "fold_seed_results": units})
    result = {
        "protocol": {"images": len(records), "folds": 5, "seeds": list(SEEDS),
                     "epochs": EPOCHS, "target_pixels_per_epoch": TARGET_PIXELS_PER_EPOCH,
                     "jpeg_quality": QUALITY, "learning_rate": LEARNING_RATE,
                     "manifest_sha256": digest(args.manifest),
                     "evaluation": "one central crop up to 512x512 per held-out image"},
        "models": summaries, "matched_pairs": pairs,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
