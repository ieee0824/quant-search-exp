#!/usr/bin/env python3
"""Quantize the frozen matched models to both MXFP6 formats and compare PSNR."""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path


CONFIGS = (("residual", 16), ("direct", 12), ("residual", 21), ("direct", 16))
SEEDS = (42, 43)
FORMATS = ("e2m3", "e3m2")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pooled(rows: list[dict], field: str) -> float:
    total_error = 0.0
    total_channels = 0
    for row in rows:
        channels = row["width"] * row["height"] * 3
        total_error += channels * 10 ** (-row[field] / 10)
        total_channels += channels
    return -10 * math.log10(total_error / total_channels)


def run_one(binary: Path, source: Path, evaluation: Path, output_dir: Path,
            name: str, format_name: str, expected: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = output_dir / f"{name}-{format_name}.qsm6"
    report = output_dir / f"{name}-{format_name}.json"
    if not model.exists():
        subprocess.run([str(binary), "quantize-matched-fp6", str(source), str(model), format_name],
                       check=True, stdout=subprocess.DEVNULL)
    if not report.exists():
        subprocess.run([str(binary), "eval-matched", str(model), str(evaluation), str(report), "70"],
                       check=True, stdout=subprocess.DEVNULL)
    value = json.loads(report.read_text())
    if value["precision"] != f"MXFP6-{format_name.upper()}" or len(value["images"]) != expected:
        raise ValueError(f"invalid FP6 report: {report}")
    value["model_sha256"] = digest(model)
    print(f"DONE {name} {format_name} {value['model_psnr_db']:.3f} dB", flush=True)
    return value


def summarize(fp16: list[dict], fp6: list[dict], kind: str, width: int,
              format_name: str) -> dict:
    base_rows = [image for report in fp16 for image in report["images"]]
    quant_rows = [image for report in fp6 for image in report["images"]]
    if len(base_rows) != len(quant_rows):
        raise ValueError("FP16 and FP6 image counts differ")
    deltas = []
    for base_report, quant_report in zip(fp16, fp6, strict=True):
        lookup = {Path(image["path"]).name: image for image in base_report["images"]}
        if set(lookup) != {Path(image["path"]).name for image in quant_report["images"]}:
            raise ValueError("FP16 and FP6 image sets differ")
        deltas.extend(image["model_psnr_db"] - lookup[Path(image["path"]).name]["model_psnr_db"]
                      for image in quant_report["images"])
    return {
        "kind": kind, "hidden": width, "format": format_name,
        "fp6_bytes": fp6[0]["model_bytes"],
        "fp16_pooled_psnr_db": pooled(base_rows, "model_psnr_db"),
        "fp6_pooled_psnr_db": pooled(quant_rows, "model_psnr_db"),
        "fp6_minus_fp16_pooled_db": pooled(quant_rows, "model_psnr_db") - pooled(base_rows, "model_psnr_db"),
        "mean_image_fp6_minus_fp16_db": statistics.mean(deltas),
        "image_seed_comparisons": len(deltas),
        "quantized_better_count": sum(value > 0 for value in deltas),
        "model_sha256": [report["model_sha256"] for report in fp6],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("cv_model_dir", type=Path)
    parser.add_argument("fold_root", type=Path)
    parser.add_argument("final_model_dir", type=Path)
    parser.add_argument("final_report_dir", type=Path)
    parser.add_argument("final_dir", type=Path)
    parser.add_argument("work_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = []
    for split in ("cv", "final"):
        for format_name in FORMATS:
            for kind, width in CONFIGS:
                fp16_reports = []
                fp6_reports = []
                model_names = []
                for fold in (range(5) if split == "cv" else (None,)):
                    for seed in SEEDS:
                        name = (f"fold{fold}-seed{seed}-{kind}{width}" if fold is not None
                                else f"final-seed{seed}-{kind}{width}")
                        source_dir = args.cv_model_dir if fold is not None else args.final_model_dir
                        source = source_dir / f"{name}.qsm"
                        fp16_report = (args.cv_model_dir if fold is not None else args.final_report_dir) / f"{name}.json"
                        if not source.exists() or not fp16_report.exists():
                            raise ValueError(f"missing FP16 source or report for {name}")
                        evaluation = (args.fold_root / f"fold-{fold}" / "eval"
                                      if fold is not None else args.final_dir)
                        expected = len(list(evaluation.iterdir()))
                        fp16_reports.append(json.loads(fp16_report.read_text()))
                        fp6_reports.append(run_one(args.binary, source, evaluation,
                                                   args.work_root / split, name, format_name, expected))
                        model_names.append(name)
                summary = summarize(fp16_reports, fp6_reports, kind, width, format_name)
                summary["split"] = split
                summary["models"] = model_names
                results.append(summary)
    output = {"protocol": {"formats": list(FORMATS), "jpeg_quality": 70,
                           "scale": "E8M0 per 32 connection weights, OCP default",
                           "rounding": "nearest-even", "biases": "FP16",
                           "arithmetic": "dequantized FP32", "binary_sha256": digest(args.binary)},
              "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
