//! Reproducible dataset verification and comparison reports.

use crate::{Model, Result, crops_from_image, load_rgb, make_pair, psnr, squared_error};
use image::ExtendedColorType;
use image::codecs::jpeg::JpegEncoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

const GALLERY: &str = "https://blog.ast.moe/gallery/";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    gallery: String,
    page_sha256: String,
    images: Vec<ManifestImage>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ManifestImage {
    index: usize,
    source_url: String,
    split: String,
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Serialize)]
pub struct Verification {
    pub manifest_sha256: String,
    pub images: usize,
    pub train: usize,
    pub val: usize,
    pub test: usize,
    pub final_holdout: usize,
    pub total_bytes: u64,
}

#[derive(Serialize)]
struct ModelIdentity {
    path: String,
    format: String,
    sha256: String,
    bytes: u64,
}

#[derive(Serialize)]
struct ModelImageResult {
    model_index: usize,
    psnr_db: Option<f64>,
    inference_ms: f64,
    output_jpeg_encode_ms: f64,
}

#[derive(Serialize)]
struct ImageResult {
    path: String,
    source_sha256: String,
    width: u32,
    height: u32,
    baseline_psnr_db: Option<f64>,
    source_read_decode_ms: f64,
    synthetic_pair_ms: f64,
    models: Vec<ModelImageResult>,
}

#[derive(Serialize)]
struct ModelSummary {
    model_index: usize,
    psnr_db: Option<f64>,
    mean_inference_ms_per_image: f64,
    mean_output_jpeg_encode_ms_per_image: f64,
}

#[derive(Serialize)]
struct Summary {
    images: usize,
    rgb_channels: u64,
    baseline_psnr_db: Option<f64>,
    models: Vec<ModelSummary>,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    runtime: Runtime,
    dataset: DatasetIdentity,
    settings: Settings,
    models: Vec<ModelIdentity>,
    images: Vec<ImageResult>,
    summary: Summary,
}

#[derive(Serialize)]
struct Runtime {
    package_version: String,
    os: String,
    arch: String,
    neon_enabled: bool,
}

#[derive(Serialize)]
struct DatasetIdentity {
    manifest_path: String,
    data_root: String,
    manifest_sha256: String,
    gallery: String,
    gallery_page_sha256: String,
    split: String,
    evaluation_status: String,
    exclusion_manifest_sha256: Option<String>,
    verification: Verification,
}

#[derive(Serialize)]
struct Settings {
    jpeg_quality: u8,
    output_jpeg_quality: u8,
    timing_runs: usize,
    crop: String,
    psnr: String,
    timing: String,
}

fn digest_file(path: &Path) -> Result<(String, u64)> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut length = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        length += count as u64;
    }
    Ok((format!("{:x}", hash.finalize()), length))
}

fn digest_format(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn split_for_index(index: usize) -> &'static str {
    match index % 6 {
        5 => "val",
        0 => "test",
        _ => "train",
    }
}

fn read_manifest(path: &Path) -> Result<(Manifest, String)> {
    let bytes = fs::read(path)?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    if !manifest.gallery.starts_with("https://") || !digest_format(&manifest.page_sha256) {
        return Err("invalid gallery URL or page SHA-256".into());
    }
    if manifest.images.is_empty() {
        return Err("manifest has no images".into());
    }
    let mut indices = HashSet::new();
    let mut urls = HashSet::new();
    let mut paths = HashSet::new();
    let mut digests = HashSet::new();
    for image in &manifest.images {
        let relative = Path::new(&image.path);
        let components: Vec<_> = relative.components().collect();
        if image.index == 0
            || image.bytes < 4
            || !digest_format(&image.sha256)
            || !matches!(image.split.as_str(), "train" | "val" | "test" | "final")
            || components.len() != 2
            || !components.iter().all(|c| matches!(c, Component::Normal(_)))
            || components[0].as_os_str() != image.split.as_str()
            || !indices.insert(image.index)
            || !urls.insert(&image.source_url)
            || !paths.insert(&image.path)
            || !digests.insert(&image.sha256)
        {
            return Err(format!("invalid or duplicate manifest entry: {}", image.path).into());
        }
        let url_path = image
            .source_url
            .strip_prefix("https://")
            .and_then(|tail| tail.split_once('/'))
            .map(|(_, path)| path)
            .ok_or_else(|| format!("invalid source URL: {}", image.source_url))?;
        let filename = relative.file_name().unwrap().to_string_lossy();
        let expected_name = format!(
            "{:02}_{}",
            image.index,
            url_path.rsplit('/').next().unwrap()
        );
        if url_path.contains(['?', '#'])
            || url_path
                .split('/')
                .any(|part| part == ".." || part.is_empty())
            || !matches!(
                relative
                    .extension()
                    .and_then(|x| x.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase()
                    .as_str(),
                "jpg" | "jpeg"
            )
            || filename != expected_name
        {
            return Err(format!("source URL and path disagree: {}", image.path).into());
        }
        if manifest.gallery == GALLERY
            && (!image.source_url.starts_with("https://blog.ast.moe/images/")
                || image.index > 26
                || image.split != split_for_index(image.index))
        {
            return Err(format!("gallery split or URL is invalid: {}", image.path).into());
        }
    }
    if manifest.gallery == GALLERY
        && (manifest.images.len() != 26 || (1..=26).any(|i| !indices.contains(&i)))
    {
        return Err("gallery manifest must contain indices 1..26 exactly".into());
    }
    Ok((manifest, hash))
}

pub fn verify_manifest_data(manifest_path: &Path, data_root: &Path) -> Result<Verification> {
    let (manifest, manifest_sha256) = read_manifest(manifest_path)?;
    let mut result = Verification {
        manifest_sha256,
        images: manifest.images.len(),
        train: 0,
        val: 0,
        test: 0,
        final_holdout: 0,
        total_bytes: 0,
    };
    for image in &manifest.images {
        let path = data_root.join(&image.path);
        let (actual_hash, actual_bytes) =
            digest_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        if actual_hash != image.sha256 || actual_bytes != image.bytes {
            return Err(format!(
                "image size or SHA-256 differs from manifest: {}",
                path.display()
            )
            .into());
        }
        result.total_bytes += actual_bytes;
        match image.split.as_str() {
            "train" => result.train += 1,
            "val" => result.val += 1,
            "test" => result.test += 1,
            "final" => result.final_holdout += 1,
            _ => unreachable!(),
        }
    }
    Ok(result)
}

fn finite_psnr(error: f64, channels: u64) -> Option<f64> {
    let value = psnr(error / channels as f64);
    value.is_finite().then_some(value)
}

pub struct ReportOptions<'a> {
    pub quality: u8,
    pub runs: usize,
    pub exclusion_manifest: Option<&'a Path>,
}

pub fn evaluate_report(
    manifest_path: &Path,
    data_root: &Path,
    split: &str,
    output: &Path,
    model_paths: &[PathBuf],
    options: ReportOptions<'_>,
) -> Result<()> {
    let ReportOptions {
        quality,
        runs,
        exclusion_manifest,
    } = options;
    if !(1..=100).contains(&quality) || runs == 0 || model_paths.is_empty() {
        return Err("JPEG quality must be 1..100; runs and model count must be positive".into());
    }
    if !matches!(split, "val" | "test" | "final") {
        return Err("split must be val, test or final".into());
    }
    if output == manifest_path
        || exclusion_manifest == Some(output)
        || model_paths.iter().any(|path| path == output)
    {
        return Err("report output must differ from manifest and model inputs".into());
    }
    let (manifest, manifest_sha256) = read_manifest(manifest_path)?;
    let verification = verify_manifest_data(manifest_path, data_root)?;
    let selected: Vec<_> = manifest
        .images
        .iter()
        .filter(|image| image.split == split)
        .collect();
    if selected.is_empty() {
        return Err(format!("manifest has no {split} images").into());
    }
    let exclusion_hash = if split == "final" {
        let path = exclusion_manifest.ok_or("final split requires --exclude-manifest")?;
        let (excluded, hash) = read_manifest(path)?;
        let known_urls: HashSet<_> = excluded.images.iter().map(|x| &x.source_url).collect();
        let known_hashes: HashSet<_> = excluded.images.iter().map(|x| &x.sha256).collect();
        if selected.iter().any(|image| {
            known_urls.contains(&image.source_url) || known_hashes.contains(&image.sha256)
        }) {
            return Err("final holdout overlaps the exclusion manifest".into());
        }
        Some(hash)
    } else {
        if exclusion_manifest.is_some() {
            return Err("--exclude-manifest is only valid for final split".into());
        }
        None
    };
    let mut models = Vec::new();
    let mut identities = Vec::new();
    for path in model_paths {
        let (sha256, bytes) = digest_file(path)?;
        let model = Model::load_fp16(path)?;
        identities.push(ModelIdentity {
            path: path.display().to_string(),
            format: "QSR1-FP16".into(),
            sha256,
            bytes,
        });
        models.push(model);
    }
    let mut results = Vec::new();
    let mut baseline_error = 0.0;
    let mut model_errors = vec![0.0; models.len()];
    let mut inference_totals = vec![0.0; models.len()];
    let mut encode_totals = vec![0.0; models.len()];
    let mut channels = 0_u64;
    for image in selected {
        let path = data_root.join(&image.path);
        let start = Instant::now();
        let source = load_rgb(&path)?;
        let source_read_decode_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let start = Instant::now();
        let crop = crops_from_image(&source, false)?.remove(0);
        let pair = make_pair(&crop, quality)?;
        let synthetic_pair_ms = start.elapsed().as_secs_f64() * 1_000.0;
        let image_channels = pair.target.width() as u64 * pair.target.height() as u64 * 3;
        let image_baseline_error = squared_error(&pair.base, &pair.target);
        baseline_error += image_baseline_error;
        channels += image_channels;
        let mut model_results = Vec::new();
        for (index, model) in models.iter().enumerate() {
            let mut inference_ms = 0.0;
            let mut first_output = None;
            for run in 0..runs {
                let start = Instant::now();
                let rendered = std::hint::black_box(model.enhance(&pair.base));
                inference_ms += start.elapsed().as_secs_f64() * 1_000.0;
                if run == 0 {
                    first_output = Some(rendered);
                }
            }
            inference_ms /= runs as f64;
            let rendered = first_output.unwrap();
            let error = squared_error(&rendered, &pair.target);
            model_errors[index] += error;
            let mut encoded = Vec::new();
            let start = Instant::now();
            JpegEncoder::new_with_quality(&mut encoded, 95).encode(
                rendered.as_raw(),
                rendered.width(),
                rendered.height(),
                ExtendedColorType::Rgb8,
            )?;
            let output_jpeg_encode_ms = start.elapsed().as_secs_f64() * 1_000.0;
            inference_totals[index] += inference_ms;
            encode_totals[index] += output_jpeg_encode_ms;
            model_results.push(ModelImageResult {
                model_index: index,
                psnr_db: finite_psnr(error, image_channels),
                inference_ms,
                output_jpeg_encode_ms,
            });
        }
        results.push(ImageResult {
            path: image.path.clone(),
            source_sha256: image.sha256.clone(),
            width: pair.target.width(),
            height: pair.target.height(),
            baseline_psnr_db: finite_psnr(image_baseline_error, image_channels),
            source_read_decode_ms,
            synthetic_pair_ms,
            models: model_results,
        });
    }
    let report = Report {
        schema_version: 1,
        runtime: Runtime {
            package_version: env!("CARGO_PKG_VERSION").into(),
            os: std::env::consts::OS.into(),
            arch: std::env::consts::ARCH.into(),
            neon_enabled: Model::simd_enabled(),
        },
        dataset: DatasetIdentity {
            manifest_path: manifest_path.display().to_string(),
            data_root: data_root.display().to_string(),
            manifest_sha256,
            gallery: manifest.gallery,
            gallery_page_sha256: manifest.page_sha256,
            split: split.into(),
            evaluation_status: if split == "final" {
                "final_holdout_disjoint_from_exclusion_manifest; prior use requires external provenance"
            } else if split == "test" {
                "exploratory: these test images were inspected during initial training selection"
            } else {
                "candidate_selection_validation"
            }.into(),
            exclusion_manifest_sha256: exclusion_hash,
            verification,
        },
        settings: Settings {
            jpeg_quality: quality,
            output_jpeg_quality: 95,
            timing_runs: runs,
            crop: "center; at most 512x512; even dimensions; synthetic 2x JPEG input".into(),
            psnr: "RGB pixel-weighted MSE; infinite PSNR encoded as null".into(),
            timing: "milliseconds; decode, synthetic pair, DNN inference, and output JPEG encode are separate; inference mean over timing_runs".into(),
        },
        models: identities,
        summary: Summary {
            images: results.len(),
            rgb_channels: channels,
            baseline_psnr_db: finite_psnr(baseline_error, channels),
            models: model_errors.iter().enumerate().map(|(index, error)| ModelSummary {
                model_index: index,
                psnr_db: finite_psnr(*error, channels),
                mean_inference_ms_per_image: inference_totals[index] / results.len() as f64,
                mean_output_jpeg_encode_ms_per_image: encode_totals[index] / results.len() as f64,
            }).collect(),
        },
        images: results,
    };
    let parent = output.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = output.with_extension("json.tmp");
    let mut file = File::create(&temporary)?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(temporary, output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluate;
    use image::{Rgb, RgbImage};

    fn fixture() -> (PathBuf, PathBuf, PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "quant-search-report-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let data = root.join("data");
        fs::create_dir_all(data.join("val")).unwrap();
        let image_path = data.join("val/01_sample.jpg");
        let image = RgbImage::from_fn(24, 20, |x, y| {
            Rgb([((x * 7 + y) % 256) as u8, ((x + y * 11) % 256) as u8, 90])
        });
        image.save(&image_path).unwrap();
        let (hash, bytes) = digest_file(&image_path).unwrap();
        let manifest_path = root.join("manifest.json");
        let manifest = serde_json::json!({
            "gallery": "https://example.org/held-out/",
            "page_sha256": "0".repeat(64),
            "images": [{
                "index": 1,
                "source_url": "https://example.org/images/sample.jpg",
                "split": "val",
                "path": "val/01_sample.jpg",
                "bytes": bytes,
                "sha256": hash
            }]
        });
        fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        let model_path = root.join("model.qsr");
        Model::new(42).save_fp16(&model_path).unwrap();
        let report_path = root.join("report.json");
        (root, manifest_path, model_path, report_path)
    }

    #[test]
    fn checked_in_gallery_manifest_has_reproducible_split() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/gallery_manifest.json");
        let (manifest, _) = read_manifest(&path).unwrap();
        assert_eq!(
            manifest
                .images
                .iter()
                .filter(|x| x.split == "train")
                .count(),
            18
        );
        assert_eq!(
            manifest.images.iter().filter(|x| x.split == "val").count(),
            4
        );
        assert_eq!(
            manifest.images.iter().filter(|x| x.split == "test").count(),
            4
        );
        let mut changed: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        changed["images"][4]["split"] = "train".into();
        let copy = std::env::temp_dir().join(format!(
            "quant-search-invalid-manifest-{}.json",
            std::process::id()
        ));
        fs::write(&copy, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(read_manifest(&copy).is_err());
        fs::remove_file(copy).unwrap();
    }

    #[test]
    fn report_matches_existing_aggregate_and_detects_changed_bytes() {
        let (root, manifest, model, report) = fixture();
        let data = root.join("data");
        let verified = verify_manifest_data(&manifest, &data).unwrap();
        assert_eq!(verified.val, 1);
        evaluate_report(
            &manifest,
            &data,
            "val",
            &report,
            &[model.clone(), model.clone()],
            ReportOptions {
                quality: 70,
                runs: 1,
                exclusion_manifest: None,
            },
        )
        .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&fs::read(&report).unwrap()).unwrap();
        let loaded = Model::load_fp16(&model).unwrap();
        let old = evaluate(&loaded, &data.join("val"), 70).unwrap();
        let actual = value["summary"]["models"][0]["psnr_db"].as_f64().unwrap();
        assert!((actual - old.model_psnr).abs() < 1e-9);
        assert_eq!(value["images"].as_array().unwrap().len(), 1);
        assert_eq!(value["images"][0]["models"].as_array().unwrap().len(), 2);
        assert_eq!(value["models"][0]["bytes"], 1006);
        let image = data.join("val/01_sample.jpg");
        let mut bytes = fs::read(&image).unwrap();
        bytes[10] ^= 1;
        fs::write(image, bytes).unwrap();
        assert!(verify_manifest_data(&manifest, &data).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn final_holdout_requires_disjoint_exclusion_manifest() {
        let (root, manifest, model, report) = fixture();
        let data = root.join("data");
        fs::create_dir_all(data.join("final")).unwrap();
        let final_image = data.join("final/01_sample.jpg");
        fs::copy(data.join("val/01_sample.jpg"), &final_image).unwrap();
        let final_manifest = root.join("final_manifest.json");
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
        value["images"][0]["split"] = "final".into();
        value["images"][0]["path"] = "final/01_sample.jpg".into();
        fs::write(&final_manifest, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(
            evaluate_report(
                &final_manifest,
                &data,
                "final",
                &report,
                std::slice::from_ref(&model),
                ReportOptions {
                    quality: 70,
                    runs: 1,
                    exclusion_manifest: None
                },
            )
            .is_err()
        );
        assert!(
            evaluate_report(
                &final_manifest,
                &data,
                "final",
                &report,
                std::slice::from_ref(&model),
                ReportOptions {
                    quality: 70,
                    runs: 1,
                    exclusion_manifest: Some(&manifest)
                },
            )
            .is_err()
        );
        let image = RgbImage::from_pixel(24, 20, Rgb([21, 45, 81]));
        image.save(&final_image).unwrap();
        let (hash, bytes) = digest_file(&final_image).unwrap();
        value["images"][0]["source_url"] = "https://example.org/images/new.jpg".into();
        value["images"][0]["path"] = "final/01_new.jpg".into();
        value["images"][0]["sha256"] = hash.into();
        value["images"][0]["bytes"] = bytes.into();
        fs::rename(&final_image, data.join("final/01_new.jpg")).unwrap();
        fs::write(&final_manifest, serde_json::to_vec(&value).unwrap()).unwrap();
        evaluate_report(
            &final_manifest,
            &data,
            "final",
            &report,
            &[model],
            ReportOptions {
                quality: 70,
                runs: 1,
                exclusion_manifest: Some(&manifest),
            },
        )
        .unwrap();
        let result: serde_json::Value =
            serde_json::from_slice(&fs::read(&report).unwrap()).unwrap();
        assert_eq!(result["dataset"]["split"], "final");
        fs::remove_dir_all(root).unwrap();
    }
}
