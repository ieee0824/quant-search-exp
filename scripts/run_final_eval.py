#!/usr/bin/env python3
"""Train the frozen candidates on all training images and evaluate final images once."""

import argparse
import hashlib
import json
import statistics
import subprocess
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
    parser.add_argument("train_manifest", type=Path)
    parser.add_argument("final_manifest", type=Path)
    parser.add_argument("cv_result", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("work_root", type=Path)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    cv = json.loads(args.cv_result.read_text())
    if cv["protocol"]["manifest_sha256"] != digest(args.train_manifest):
        raise ValueError("cross-validation used a different training snapshot")
    train = json.loads(args.train_manifest.read_text())["images"]
    final = json.loads(args.final_manifest.read_text())["images"]
    if len(train) != 174 or len(final) != 12:
        raise ValueError("training or final snapshot has changed")
    for record in train:
        path = args.data_root / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError(f"image changed: {path}")
    for record in final:
        path = Path(record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError(f"image changed: {path}")
    if {item["sha256"] for item in train} & {item["sha256"] for item in final}:
        raise ValueError("training/final image overlap")
    stage = args.work_root / "all-train"
    stage.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(train):
        source = (args.data_root / record["path"]).resolve()
        link = stage / f"{index:03d}-{source.name}"
        if link.is_symlink():
            if link.resolve() != source:
                raise ValueError(f"staged link changed: {link}")
        elif link.exists():
            raise ValueError(f"staged path is not a symlink: {link}")
        else:
            link.symlink_to(source)
    args.work_root.mkdir(parents=True, exist_ok=True)
    baseline_path = args.work_root / "final-baselines.json"
    if not baseline_path.exists():
        subprocess.run(
            [str(args.binary), "eval-direct-report", "data/direct_fp16_baseline.qsd",
             "data/final_inbox", str(baseline_path), str(QUALITY)], check=True,
        )
    baseline = json.loads(baseline_path.read_text())
    if len(baseline["images"]) != len(final):
        raise ValueError("baseline final report has wrong image count")
    runs = []
    for seed in SEEDS:
        for kind, hidden in CONFIGS:
            name = f"final-seed{seed}-{kind}{hidden}"
            model = args.work_root / f"{name}.qsm"
            report = args.work_root / f"{name}.json"
            if not model.exists():
                print(f"TRAIN {name}", flush=True)
                with (args.work_root / f"{name}.log").open("w") as stream:
                    subprocess.run(
                        [str(args.binary), "train-matched", kind, str(hidden), str(stage),
                         str(model), str(EPOCHS), str(TARGET_PIXELS_PER_EPOCH), str(QUALITY),
                         str(seed), str(LEARNING_RATE)], check=True, stdout=stream,
                    )
            if not report.exists():
                print(f"EVAL  {name}", flush=True)
                subprocess.run(
                    [str(args.binary), "eval-matched", str(model), "data/final_inbox",
                     str(report), str(QUALITY)], check=True, stdout=subprocess.DEVNULL,
                )
            value = json.loads(report.read_text())
            if (value["kind"], value["hidden"], len(value["images"])) != (kind, hidden, 12):
                raise ValueError(f"invalid final report: {report}")
            runs.append({"seed": seed, "kind": kind, "hidden": hidden, "data": value,
                         "model_sha256": digest(model)})
            print(f"DONE  {name} {value['model_psnr_db']:.3f} dB", flush=True)
    models = []
    for kind, hidden in CONFIGS:
        subset = [run for run in runs if (run["kind"], run["hidden"]) == (kind, hidden)]
        images = [image for run in subset for image in run["data"]["images"]]
        models.append({"kind": kind, "hidden": hidden,
                       "parameters": subset[0]["data"]["parameters"],
                       "model_bytes": subset[0]["data"]["model_bytes"],
                       "mean_image_psnr_db": statistics.mean(x["model_psnr_db"] for x in images),
                       "mean_image_gain_over_bicubic_db": statistics.mean(
                           x["model_psnr_db"] - x["bicubic_psnr_db"] for x in images),
                       "runs": [{"seed": run["seed"], "model_sha256": run["model_sha256"],
                                 "pooled_psnr_db": run["data"]["model_psnr_db"]}
                                for run in subset]})
    pairs = []
    for size, residual_width, direct_width in (("~500", 16, 12), ("~650", 21, 16)):
        differences = []
        for seed in SEEDS:
            res = next(run for run in runs if (run["seed"], run["kind"], run["hidden"]) == (seed, "residual", residual_width))
            direct = next(run for run in runs if (run["seed"], run["kind"], run["hidden"]) == (seed, "direct", direct_width))
            by_name = {Path(x["path"]).name: x for x in res["data"]["images"]}
            differences.extend(x["model_psnr_db"] - by_name[Path(x["path"]).name]["model_psnr_db"]
                               for x in direct["data"]["images"])
        pairs.append({"size": size, "mean_direct_minus_residual_db": statistics.mean(differences),
                      "median_direct_minus_residual_db": statistics.median(differences),
                      "image_seed_comparisons": len(differences)})
    per_image = []
    for base in baseline["images"]:
        name = Path(base["path"]).name
        scores = {}
        for run in runs:
            match = next(x for x in run["data"]["images"] if Path(x["path"]).name == name)
            scores[f"{run['kind']}{run['hidden']}_seed{run['seed']}"] = match["model_psnr_db"]
        per_image.append({"path": name, "width": base["width"], "height": base["height"],
                          "nearest_psnr_db": base["nearest_psnr_db"],
                          "bicubic_psnr_db": base["bicubic_psnr_db"], "model_psnr_db": scores})
    result = {
        "protocol": {"train_images": 174, "final_images": 12, "seeds": list(SEEDS),
                     "epochs": EPOCHS, "target_pixels_per_epoch": TARGET_PIXELS_PER_EPOCH,
                     "jpeg_quality": QUALITY, "learning_rate": LEARNING_RATE,
                     "train_manifest_sha256": digest(args.train_manifest),
                     "final_manifest_sha256": digest(args.final_manifest),
                     "evaluation": "one central crop up to 512x512 per final image"},
        "baselines": {
            "nearest_mean_image_psnr_db": statistics.mean(x["nearest_psnr_db"] for x in baseline["images"]),
            "bicubic_mean_image_psnr_db": statistics.mean(x["bicubic_psnr_db"] for x in baseline["images"]),
        },
        "models": models, "matched_pairs": pairs, "per_image": per_image,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
