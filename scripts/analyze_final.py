#!/usr/bin/env python3
"""Add pixel-pooled PSNR and per-image win counts to the final report."""

import argparse
import json
import math
import statistics
from pathlib import Path


def pooled(rows: list[tuple[float, int]]) -> float:
    total = sum(10 ** (-score / 10) * pixels for score, pixels in rows)
    return -10 * math.log10(total / sum(pixels for _, pixels in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("final_result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.final_result.read_text())
    images = report["per_image"]
    if len(images) != 12 or len({image["path"] for image in images}) != 12:
        raise ValueError("final report must contain 12 unique images")
    result = {
        "note": "Pooled PSNR uses the combined RGB pixel MSE across final central crops.",
        "pooled_psnr_db": {
            "nearest": pooled([(x["nearest_psnr_db"], x.get("width", 512) * x.get("height", 512)) for x in images]),
            "bicubic": pooled([(x["bicubic_psnr_db"], x.get("width", 512) * x.get("height", 512)) for x in images]),
        },
        "models": [],
        "paired_wins": [],
    }
    for model in report["models"]:
        name = f"{model['kind']}{model['hidden']}"
        scores = [(image["model_psnr_db"][f"{name}_seed{seed}"],
                   image.get("width", 512) * image.get("height", 512))
                  for image in images for seed in (42, 43)]
        result["models"].append({"kind": model["kind"], "hidden": model["hidden"],
                                 "pooled_psnr_db": pooled(scores),
                                 "mean_image_psnr_db": statistics.mean(score for score, _ in scores),
                                 "image_seed_beats_bicubic": sum(
                                     image["model_psnr_db"][f"{name}_seed{seed}"] > image["bicubic_psnr_db"]
                                     for image in images for seed in (42, 43))})
    for size, residual, direct in (("~500", "residual16", "direct12"),
                                   ("~650", "residual21", "direct16")):
        differences = [image["model_psnr_db"][f"{direct}_seed{seed}"] -
                       image["model_psnr_db"][f"{residual}_seed{seed}"]
                       for image in images for seed in (42, 43)]
        result["paired_wins"].append({"size": size,
                                      "direct_better_image_seed_count": sum(x > 0 for x in differences),
                                      "comparisons": len(differences),
                                      "mean_direct_minus_residual_db": statistics.mean(differences)})
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
