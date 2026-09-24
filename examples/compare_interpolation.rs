//! Compare interpolation baselines using the same synthetic JPEG pairs as eval-report.

use image::codecs::jpeg::JpegEncoder;
use image::imageops::{self, FilterType};
use image::{DynamicImage, ImageFormat, RgbImage};
use quant_search_exp::{Result, image_paths, load_rgb};
use serde_json::json;
use std::env;
use std::path::Path;

const QUALITY: u8 = 70;

fn pair(path: &Path) -> Result<(RgbImage, RgbImage)> {
    let source = load_rgb(path)?;
    let width = source.width().min(512) & !1;
    let height = source.height().min(512) & !1;
    if width < 4 || height < 4 {
        return Err(format!("image is smaller than 4x4: {}", path.display()).into());
    }
    let x = (source.width() - width) / 2;
    let y = (source.height() - height) / 2;
    let target = imageops::crop_imm(&source, x, y, width, height).to_image();
    let low = imageops::resize(&target, width / 2, height / 2, FilterType::CatmullRom);
    let mut encoded = Vec::new();
    DynamicImage::ImageRgb8(low)
        .write_with_encoder(JpegEncoder::new_with_quality(&mut encoded, QUALITY))?;
    let low = image::load_from_memory_with_format(&encoded, ImageFormat::Jpeg)?.to_rgb8();
    Ok((target, low))
}

fn squared_error(actual: &RgbImage, target: &RgbImage) -> f64 {
    actual
        .pixels()
        .zip(target.pixels())
        .flat_map(|(a, b)| a.0.into_iter().zip(b.0))
        .map(|(a, b)| {
            let difference = (a as f64 - b as f64) / 255.0;
            difference * difference
        })
        .sum()
}

fn psnr(error: f64, channels: u64) -> f64 {
    -10.0 * (error / channels as f64).log10()
}

fn main() -> Result<()> {
    let mut splits = Vec::new();
    for directory in env::args().skip(1) {
        let mut nearest_error = 0.0;
        let mut bicubic_error = 0.0;
        let mut channels = 0_u64;
        let mut images = Vec::new();
        for path in image_paths(Path::new(&directory))? {
            let (target, low) = pair(&path)?;
            let nearest =
                imageops::resize(&low, target.width(), target.height(), FilterType::Nearest);
            let bicubic = imageops::resize(
                &low,
                target.width(),
                target.height(),
                FilterType::CatmullRom,
            );
            let n = squared_error(&nearest, &target);
            let b = squared_error(&bicubic, &target);
            let count = target.width() as u64 * target.height() as u64 * 3;
            nearest_error += n;
            bicubic_error += b;
            channels += count;
            images.push(json!({
                "path": path,
                "nearest_psnr_db": psnr(n, count),
                "bicubic_psnr_db": psnr(b, count),
            }));
        }
        splits.push(json!({
            "directory": directory,
            "images": images,
            "summary": {
                "image_count": images.len(),
                "nearest_psnr_db": psnr(nearest_error, channels),
                "bicubic_psnr_db": psnr(bicubic_error, channels),
            },
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "jpeg_quality": QUALITY,
            "crop": "center; at most 512x512; even dimensions; synthetic 2x JPEG input",
            "psnr": "RGB pixel-weighted MSE",
            "splits": splits,
        }))?
    );
    Ok(())
}
