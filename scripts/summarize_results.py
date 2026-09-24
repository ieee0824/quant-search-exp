#!/usr/bin/env python3
"""Print a compact Markdown comparison table from one or more eval-report JSON files."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    print("| report | split | model | PSNR (dB) | file bytes | bits/parameter | inference ms/image |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: |")
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        split = report["dataset"]["split"]
        if split == "test":
            split = "test (exploratory)"
        for index, model in enumerate(report["models"]):
            score = report["summary"]["models"][index]
            psnr = score["psnr_db"]
            psnr_text = "infinite" if psnr is None else f"{psnr:.3f}"
            print(
                f"| {path.stem} | {split} | {Path(model['path']).name} | "
                f"{psnr_text} | {model['bytes']} | "
                f"{model['effective_bits_per_parameter']:.3f} | "
                f"{score['mean_inference_ms_per_image']:.3f} |"
            )


if __name__ == "__main__":
    main()
