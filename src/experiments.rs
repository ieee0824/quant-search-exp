//! Validation-only model selection experiments for FP6 blocks.

use crate::evaluation::ValidationSet;
use crate::fp6::{BLOCKS, BlockOptions, Fp6Format, Fp6Model, Rounding};
use crate::{Model, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[derive(Serialize)]
struct Trial {
    block: usize,
    format: String,
    scale_offset: i8,
    clip_ratio: f32,
    rounding: String,
    validation_psnr_db: f64,
    accepted: bool,
}

#[derive(Serialize)]
struct SelectionRecord {
    schema_version: u32,
    experiment: String,
    source_model_sha256: String,
    validation_manifest_sha256: String,
    selection_split: String,
    selection_images: usize,
    jpeg_quality: u8,
    format: String,
    block_size: usize,
    baseline_e3m2_psnr_db: Option<f64>,
    baseline_e2m3_psnr_db: Option<f64>,
    baseline_psnr_db: f64,
    selected_psnr_db: f64,
    trials: Vec<Trial>,
    trial_count: usize,
    search_ms: f64,
    forced_mixed_for_comparison: bool,
    selected_e3m2_blocks: usize,
    selected_e2m3_blocks: usize,
    output_model_sha256: String,
    output_file_bytes: u64,
}

fn digest(path: &Path) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

fn write_record(path: &Path, record: &SelectionRecord) -> Result<()> {
    fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
    let mut file = File::create(path)?;
    serde_json::to_writer_pretty(&mut file, record)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn validate_outputs(source: &Path, manifest: &Path, output: &Path, record: &Path) -> Result<()> {
    if source == output
        || source == record
        || manifest == output
        || manifest == record
        || output == record
    {
        return Err("source, manifest, output model, and record paths must differ".into());
    }
    Ok(())
}

pub fn select_mixed_fp6(
    source_path: &Path,
    manifest_path: &Path,
    data_root: &Path,
    output_path: &Path,
    record_path: &Path,
    quality: u8,
) -> Result<()> {
    validate_outputs(source_path, manifest_path, output_path, record_path)?;
    let source = Model::load_fp16(source_path)?;
    let validation = ValidationSet::load(manifest_path, data_root, quality)?;
    let e3 = Fp6Model::uniform(&source, Fp6Format::E3M2)?;
    let e2 = Fp6Model::uniform(&source, Fp6Format::E2M3)?;
    let e3_score = validation.score(&e3);
    let e2_score = validation.score(&e2);
    let (mut selected, mut score) = if e3_score > e2_score {
        (e3, e3_score)
    } else {
        (e2, e2_score)
    };
    let baseline_score = score;
    let start = Instant::now();
    let mut trials = Vec::new();
    let mut best_forced = None;
    for block in 0..BLOCKS {
        let alternate = match selected.block_format(block) {
            Fp6Format::E3M2 => Fp6Format::E2M3,
            Fp6Format::E2M3 => Fp6Format::E3M2,
        };
        let mut candidate = selected.clone();
        candidate.set_block_from_source(&source, block, alternate, BlockOptions::default())?;
        let candidate_score = validation.score(&candidate);
        if best_forced
            .as_ref()
            .is_none_or(|(_, best_score)| candidate_score > *best_score)
        {
            best_forced = Some((block, candidate_score));
        }
        let accepted = candidate_score > score;
        if accepted {
            selected = candidate;
            score = candidate_score;
        }
        trials.push(Trial {
            block,
            format: alternate.name().into(),
            scale_offset: 0,
            clip_ratio: 1.0,
            rounding: "nearest_even".into(),
            validation_psnr_db: candidate_score,
            accepted,
        });
    }
    let mut forced_mixed = false;
    if selected.is_uniform() {
        let (block, _) = best_forced.ok_or("no mixed-format trial")?;
        let alternate = match selected.block_format(block) {
            Fp6Format::E3M2 => Fp6Format::E2M3,
            Fp6Format::E2M3 => Fp6Format::E3M2,
        };
        selected.set_block_from_source(&source, block, alternate, BlockOptions::default())?;
        score = validation.score(&selected);
        forced_mixed = true;
    }
    let search_ms = start.elapsed().as_secs_f64() * 1_000.0;
    fs::create_dir_all(output_path.parent().unwrap_or(Path::new(".")))?;
    selected.save(output_path)?;
    let record = SelectionRecord {
        schema_version: 1,
        experiment: "greedy_block_format_selection".into(),
        source_model_sha256: digest(source_path)?,
        validation_manifest_sha256: validation.manifest_sha256.clone(),
        selection_split: "val".into(),
        selection_images: validation.source_images,
        jpeg_quality: validation.quality(),
        format: selected.format_name().into(),
        block_size: crate::fp6::BLOCK_SIZE,
        baseline_e3m2_psnr_db: Some(e3_score),
        baseline_e2m3_psnr_db: Some(e2_score),
        baseline_psnr_db: baseline_score,
        selected_psnr_db: score,
        trial_count: trials.len(),
        trials,
        search_ms,
        forced_mixed_for_comparison: forced_mixed,
        selected_e3m2_blocks: (0..BLOCKS)
            .filter(|&i| selected.block_format(i) == Fp6Format::E3M2)
            .count(),
        selected_e2m3_blocks: (0..BLOCKS)
            .filter(|&i| selected.block_format(i) == Fp6Format::E2M3)
            .count(),
        output_model_sha256: digest(output_path)?,
        output_file_bytes: fs::metadata(output_path)?.len(),
    };
    write_record(record_path, &record)
}

pub fn search_uniform_fp6(
    source_path: &Path,
    manifest_path: &Path,
    data_root: &Path,
    output_path: &Path,
    record_path: &Path,
    format: Fp6Format,
    quality: u8,
) -> Result<()> {
    validate_outputs(source_path, manifest_path, output_path, record_path)?;
    let source = Model::load_fp16(source_path)?;
    let validation = ValidationSet::load(manifest_path, data_root, quality)?;
    let mut selected = Fp6Model::uniform(&source, format)?;
    let baseline_score = validation.score(&selected);
    let mut score = baseline_score;
    let start = Instant::now();
    let mut trials = Vec::new();
    for block in 0..BLOCKS {
        let mut block_best = None;
        for scale_offset in [-1, 0, 1] {
            for clip_ratio in [0.75, 0.875, 1.0] {
                for rounding in [
                    Rounding::NearestEven,
                    Rounding::TowardZero,
                    Rounding::AwayFromZero,
                ] {
                    if scale_offset == 0 && clip_ratio == 1.0 && rounding == Rounding::NearestEven {
                        continue;
                    }
                    let options = BlockOptions {
                        scale_offset,
                        clip_ratio,
                        rounding,
                    };
                    let mut candidate = selected.clone();
                    candidate.set_block_from_source(&source, block, format, options)?;
                    let candidate_score = validation.score(&candidate);
                    let accepted = candidate_score > score
                        && block_best
                            .as_ref()
                            .is_none_or(|(_, best_score)| candidate_score > *best_score);
                    if accepted {
                        block_best = Some((options, candidate_score));
                    }
                    trials.push(Trial {
                        block,
                        format: format.name().into(),
                        scale_offset,
                        clip_ratio,
                        rounding: match rounding {
                            Rounding::NearestEven => "nearest_even",
                            Rounding::TowardZero => "toward_zero",
                            Rounding::AwayFromZero => "away_from_zero",
                        }
                        .into(),
                        validation_psnr_db: candidate_score,
                        accepted: false,
                    });
                }
            }
        }
        if let Some((options, best_score)) = block_best {
            selected.set_block_from_source(&source, block, format, options)?;
            score = best_score;
            if let Some(trial) = trials.iter_mut().rev().find(|trial| {
                trial.block == block
                    && trial.scale_offset == options.scale_offset
                    && trial.clip_ratio == options.clip_ratio
                    && trial.rounding
                        == match options.rounding {
                            Rounding::NearestEven => "nearest_even",
                            Rounding::TowardZero => "toward_zero",
                            Rounding::AwayFromZero => "away_from_zero",
                        }
            }) {
                trial.accepted = true;
            }
        }
    }
    let search_ms = start.elapsed().as_secs_f64() * 1_000.0;
    if !selected.is_uniform() {
        return Err("search changed the fixed FP6 format".into());
    }
    fs::create_dir_all(output_path.parent().unwrap_or(Path::new(".")))?;
    selected.save(output_path)?;
    let record = SelectionRecord {
        schema_version: 1,
        experiment: "fixed_format_scale_clipping_rounding_search".into(),
        source_model_sha256: digest(source_path)?,
        validation_manifest_sha256: validation.manifest_sha256.clone(),
        selection_split: "val".into(),
        selection_images: validation.source_images,
        jpeg_quality: validation.quality(),
        format: selected.format_name().into(),
        block_size: crate::fp6::BLOCK_SIZE,
        baseline_e3m2_psnr_db: None,
        baseline_e2m3_psnr_db: None,
        baseline_psnr_db: baseline_score,
        selected_psnr_db: score,
        trial_count: trials.len(),
        trials,
        search_ms,
        forced_mixed_for_comparison: false,
        selected_e3m2_blocks: (0..BLOCKS)
            .filter(|&i| selected.block_format(i) == Fp6Format::E3M2)
            .count(),
        selected_e2m3_blocks: (0..BLOCKS)
            .filter(|&i| selected.block_format(i) == Fp6Format::E2M3)
            .count(),
        output_model_sha256: digest(output_path)?,
        output_file_bytes: fs::metadata(output_path)?.len(),
    };
    write_record(record_path, &record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn mixed_selection_and_fixed_format_search_write_reusable_models() {
        let root = std::env::temp_dir().join(format!(
            "quant-search-fp6-experiments-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let data = root.join("data");
        fs::create_dir_all(data.join("val")).unwrap();
        let image_path = data.join("val/01_example.jpg");
        RgbImage::from_fn(24, 20, |x, y| {
            Rgb([
                ((x * 11 + y * 5) % 256) as u8,
                ((x * 3 + y * 13) % 256) as u8,
                ((x * 7 + y * 17) % 256) as u8,
            ])
        })
        .save(&image_path)
        .unwrap();
        let manifest_path = root.join("manifest.json");
        let manifest = serde_json::json!({
            "gallery": "https://example.org/heldout/",
            "page_sha256": "0".repeat(64),
            "images": [{
                "index": 1,
                "source_url": "https://example.org/images/example.jpg",
                "split": "val",
                "path": "val/01_example.jpg",
                "bytes": fs::metadata(&image_path).unwrap().len(),
                "sha256": digest(&image_path).unwrap(),
            }]
        });
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let source_path = root.join("model.qsr");
        Model::new(42).save_fp16(&source_path).unwrap();
        let mixed_path = root.join("mixed.qsf");
        let mixed_record = root.join("mixed.json");
        select_mixed_fp6(
            &source_path,
            &manifest_path,
            &data,
            &mixed_path,
            &mixed_record,
            70,
        )
        .unwrap();
        assert_eq!(
            Fp6Model::load(&mixed_path).unwrap().format_name(),
            "QSF1-MXFP6-MIXED"
        );
        let mixed: serde_json::Value =
            serde_json::from_slice(&fs::read(mixed_record).unwrap()).unwrap();
        assert_eq!(mixed["trial_count"], BLOCKS);
        let searched_path = root.join("searched.qsf");
        let search_record = root.join("search.json");
        search_uniform_fp6(
            &source_path,
            &manifest_path,
            &data,
            &searched_path,
            &search_record,
            Fp6Format::E2M3,
            70,
        )
        .unwrap();
        assert_eq!(
            Fp6Model::load(&searched_path).unwrap().format_name(),
            "QSF1-MXFP6-E2M3"
        );
        assert_eq!(
            fs::metadata(searched_path).unwrap().len(),
            Fp6Model::file_bytes() as u64
        );
        let searched: serde_json::Value =
            serde_json::from_slice(&fs::read(search_record).unwrap()).unwrap();
        assert_eq!(searched["trial_count"], 390);
        assert!(
            searched["selected_psnr_db"].as_f64().unwrap()
                >= searched["baseline_psnr_db"].as_f64().unwrap()
        );
        fs::remove_dir_all(root).unwrap();
    }
}
