//! Tiny 2x super-resolution model that maps low-resolution pixels directly to output blocks.

use crate::{Result, Rng, crops_from_image, image_paths, load_rgb, psnr, squared_error};
use half::f16;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::{self, FilterType};
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use serde::Serialize;
use std::fs;
use std::path::Path;

const INPUTS: usize = 27;
const HIDDEN: usize = 16;
const OUTPUTS: usize = 12;
const W1: usize = 0;
const B1: usize = W1 + INPUTS * HIDDEN;
const W2: usize = B1 + HIDDEN;
const B2: usize = W2 + HIDDEN * OUTPUTS;
pub const PARAMS: usize = B2 + OUTPUTS;
const BATCH: usize = 64;
const MAGIC: &[u8; 4] = b"QSD1";

struct Pair {
    low: RgbImage,
    target: RgbImage,
}

fn make_pair(target: &RgbImage, quality: u8) -> Result<Pair> {
    if target.width() < 4 || target.height() < 4 || !(1..=100).contains(&quality) {
        return Err("image must be at least 4x4 and JPEG quality must be 1..100".into());
    }
    let width = target.width() & !1;
    let height = target.height() & !1;
    let target = imageops::crop_imm(target, 0, 0, width, height).to_image();
    let low = imageops::resize(&target, width / 2, height / 2, FilterType::CatmullRom);
    let mut encoded = Vec::new();
    DynamicImage::ImageRgb8(low)
        .write_with_encoder(JpegEncoder::new_with_quality(&mut encoded, quality))?;
    let low = image::load_from_memory_with_format(&encoded, ImageFormat::Jpeg)?.to_rgb8();
    Ok(Pair { low, target })
}

fn neighborhood(image: &RgbImage, x: u32, y: u32) -> [f32; INPUTS] {
    let mut values = [0.0; INPUTS];
    let mut index = 0;
    for dy in -1_i64..=1 {
        for dx in -1_i64..=1 {
            let px = (x as i64 + dx).clamp(0, image.width() as i64 - 1) as u32;
            let py = (y as i64 + dy).clamp(0, image.height() as i64 - 1) as u32;
            for &channel in &image.get_pixel(px, py).0 {
                values[index] = channel as f32 / 255.0;
                index += 1;
            }
        }
    }
    values
}

#[derive(Clone)]
pub struct DirectModel {
    weights: [f32; PARAMS],
}

impl DirectModel {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut weights = [0.0; PARAMS];
        for weight in &mut weights[W1..B1] {
            *weight = rng.signed_unit() * (2.0 / INPUTS as f32).sqrt();
        }
        weights[B1..W2].fill(0.01);
        // Start at nearest-neighbor output without using interpolation during inference.
        // The first three hidden units carry the low-resolution center RGB value.
        for channel in 0..3 {
            weights[W1 + channel * INPUTS..W1 + (channel + 1) * INPUTS].fill(0.0);
            weights[W1 + channel * INPUTS + 12 + channel] = 1.0;
            weights[B1 + channel] = 0.0;
            for pixel in 0..4 {
                weights[W2 + (pixel * 3 + channel) * HIDDEN + channel] = 1.0;
            }
        }
        Self { weights }
    }

    fn forward(&self, input: &[f32; INPUTS]) -> ([f32; HIDDEN], [f32; OUTPUTS]) {
        let mut hidden = [0.0; HIDDEN];
        for (h, activation) in hidden.iter_mut().enumerate() {
            let mut sum = self.weights[B1 + h];
            for (i, &value) in input.iter().enumerate() {
                sum += self.weights[W1 + h * INPUTS + i] * value;
            }
            *activation = sum.max(0.0);
        }
        let mut output = [0.0; OUTPUTS];
        for (o, value) in output.iter_mut().enumerate() {
            let mut sum = self.weights[B2 + o];
            for (h, &activation) in hidden.iter().enumerate() {
                sum += self.weights[W2 + o * HIDDEN + h] * activation;
            }
            *value = sum;
        }
        (hidden, output)
    }

    pub fn upscale(&self, low: &RgbImage) -> Result<RgbImage> {
        let width = low.width().checked_mul(2).ok_or("image width overflows")?;
        let height = low
            .height()
            .checked_mul(2)
            .ok_or("image height overflows")?;
        let mut high = RgbImage::new(width, height);
        for y in 0..low.height() {
            for x in 0..low.width() {
                let (_, output) = self.forward(&neighborhood(low, x, y));
                for dy in 0..2 {
                    for dx in 0..2 {
                        let pixel = (dy * 2 + dx) as usize;
                        high.put_pixel(
                            2 * x + dx,
                            2 * y + dy,
                            Rgb(std::array::from_fn(|channel| {
                                (output[pixel * 3 + channel].clamp(0.0, 1.0) * 255.0).round() as u8
                            })),
                        );
                    }
                }
            }
        }
        Ok(high)
    }

    pub fn save_fp16(&self, path: &Path) -> Result<()> {
        if self
            .weights
            .iter()
            .any(|weight| !weight.is_finite() || weight.abs() > f16::MAX.to_f32())
        {
            return Err("model contains a weight that cannot be stored as FP16".into());
        }
        let mut bytes = Vec::with_capacity(8 + 2 * PARAMS);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(PARAMS as u32).to_le_bytes());
        for &weight in &self.weights {
            bytes.extend_from_slice(&f16::from_f32(weight).to_le_bytes());
        }
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load_fp16(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() != 8 + 2 * PARAMS
            || &bytes[..4] != MAGIC
            || u32::from_le_bytes(bytes[4..8].try_into()?) as usize != PARAMS
        {
            return Err("invalid QSD1 direct model".into());
        }
        let mut weights = [0.0; PARAMS];
        for (i, weight) in weights.iter_mut().enumerate() {
            *weight = f16::from_le_bytes(bytes[8 + i * 2..10 + i * 2].try_into()?).to_f32();
            if !weight.is_finite() {
                return Err("model contains a non-finite weight".into());
            }
        }
        Ok(Self { weights })
    }
}

struct Adam {
    mean: [f32; PARAMS],
    variance: [f32; PARAMS],
    step: i32,
}

impl Adam {
    fn new() -> Self {
        Self {
            mean: [0.0; PARAMS],
            variance: [0.0; PARAMS],
            step: 0,
        }
    }

    fn apply(
        &mut self,
        model: &mut DirectModel,
        gradient: &[f32; PARAMS],
        count: usize,
        rate: f32,
    ) {
        self.step += 1;
        let correction1 = 1.0 - 0.9_f32.powi(self.step);
        let correction2 = 1.0 - 0.999_f32.powi(self.step);
        for (i, &value) in gradient.iter().enumerate() {
            let g = value / count as f32;
            self.mean[i] = 0.9 * self.mean[i] + 0.1 * g;
            self.variance[i] = 0.999 * self.variance[i] + 0.001 * g * g;
            model.weights[i] -= rate * (self.mean[i] / correction1)
                / ((self.variance[i] / correction2).sqrt() + 1e-8);
        }
    }
}

fn accumulate_gradient(
    model: &DirectModel,
    pair: &Pair,
    x: u32,
    y: u32,
    gradient: &mut [f32; PARAMS],
) -> f32 {
    let input = neighborhood(&pair.low, x, y);
    let (hidden, output) = model.forward(&input);
    let mut delta = [0.0; OUTPUTS];
    let mut loss = 0.0;
    for pixel in 0..4 {
        let dx = (pixel % 2) as u32;
        let dy = (pixel / 2) as u32;
        let target = pair.target.get_pixel(2 * x + dx, 2 * y + dy);
        for channel in 0..3 {
            let o = pixel * 3 + channel;
            let error = output[o] - target[channel] as f32 / 255.0;
            loss += error * error / OUTPUTS as f32;
            delta[o] = 2.0 * error / OUTPUTS as f32;
            gradient[B2 + o] += delta[o];
            for (h, &activation) in hidden.iter().enumerate() {
                gradient[W2 + o * HIDDEN + h] += delta[o] * activation;
            }
        }
    }
    for (h, &activation) in hidden.iter().enumerate() {
        if activation == 0.0 {
            continue;
        }
        let mut upstream = 0.0;
        for (o, &derivative) in delta.iter().enumerate() {
            upstream += model.weights[W2 + o * HIDDEN + h] * derivative;
        }
        gradient[B1 + h] += upstream;
        for (i, &value) in input.iter().enumerate() {
            gradient[W1 + h * INPUTS + i] += upstream * value;
        }
    }
    loss
}

pub fn train(
    directory: &Path,
    model_path: &Path,
    epochs: usize,
    samples_per_epoch: usize,
    quality: u8,
    seed: u64,
    learning_rate: f32,
) -> Result<()> {
    if epochs == 0
        || samples_per_epoch == 0
        || !(1..=100).contains(&quality)
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
    {
        return Err("invalid training settings".into());
    }
    let mut pairs = Vec::new();
    for path in image_paths(directory)? {
        let image = load_rgb(&path)?;
        for crop in crops_from_image(&image, true)? {
            pairs.push(make_pair(&crop, quality)?);
        }
        println!("loaded {}", path.display());
    }
    let mut model = DirectModel::new(seed);
    let mut rng = Rng::new(seed ^ 0x9e37_79b9_7f4a_7c15);
    let mut optimizer = Adam::new();
    for epoch in 0..epochs {
        let mut loss = 0.0_f64;
        let mut remaining = samples_per_epoch;
        while remaining > 0 {
            let count = remaining.min(BATCH);
            let mut gradient = [0.0; PARAMS];
            for _ in 0..count {
                let pair = &pairs[rng.index(pairs.len())];
                let x = rng.index(pair.low.width() as usize) as u32;
                let y = rng.index(pair.low.height() as usize) as u32;
                loss += accumulate_gradient(&model, pair, x, y, &mut gradient) as f64;
            }
            optimizer.apply(&mut model, &gradient, count, learning_rate);
            remaining -= count;
        }
        println!(
            "epoch {}/{}: sampled MSE {:.8}",
            epoch + 1,
            epochs,
            loss / samples_per_epoch as f64
        );
    }
    model.save_fp16(model_path)?;
    println!(
        "saved direct FP16 model: {} ({PARAMS} parameters)",
        model_path.display()
    );
    Ok(())
}

#[derive(Serialize)]
pub struct ImageMetrics {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub nearest_psnr_db: f64,
    pub bicubic_psnr_db: f64,
    pub direct_psnr_db: f64,
}

#[derive(Serialize)]
pub struct Summary {
    pub images: usize,
    pub nearest_psnr_db: f64,
    pub bicubic_psnr_db: f64,
    pub direct_psnr_db: f64,
}

#[derive(Serialize)]
pub struct Report {
    pub model_path: String,
    pub model_bytes: u64,
    pub model_parameters: usize,
    pub directory: String,
    pub jpeg_quality: u8,
    pub images: Vec<ImageMetrics>,
    pub summary: Summary,
}

pub fn evaluate(model_path: &Path, directory: &Path, quality: u8) -> Result<Report> {
    let model = DirectModel::load_fp16(model_path)?;
    if !(1..=100).contains(&quality) {
        return Err("JPEG quality must be 1..100".into());
    }
    let mut images = Vec::new();
    let mut nearest_error = 0.0;
    let mut bicubic_error = 0.0;
    let mut direct_error = 0.0;
    let mut channels = 0_u64;
    for path in image_paths(directory)? {
        let source = load_rgb(&path)?;
        let crop = crops_from_image(&source, false)?.remove(0);
        let pair = make_pair(&crop, quality)?;
        let direct = model.upscale(&pair.low)?;
        let nearest = imageops::resize(
            &pair.low,
            pair.target.width(),
            pair.target.height(),
            FilterType::Nearest,
        );
        let bicubic = imageops::resize(
            &pair.low,
            pair.target.width(),
            pair.target.height(),
            FilterType::CatmullRom,
        );
        let count = pair.target.width() as u64 * pair.target.height() as u64 * 3;
        let n = squared_error(&nearest, &pair.target);
        let b = squared_error(&bicubic, &pair.target);
        let d = squared_error(&direct, &pair.target);
        nearest_error += n;
        bicubic_error += b;
        direct_error += d;
        channels += count;
        images.push(ImageMetrics {
            path: path.display().to_string(),
            width: pair.target.width(),
            height: pair.target.height(),
            nearest_psnr_db: psnr(n / count as f64),
            bicubic_psnr_db: psnr(b / count as f64),
            direct_psnr_db: psnr(d / count as f64),
        });
    }
    Ok(Report {
        model_path: model_path.display().to_string(),
        model_bytes: fs::metadata(model_path)?.len(),
        model_parameters: PARAMS,
        directory: directory.display().to_string(),
        jpeg_quality: quality,
        summary: Summary {
            images: images.len(),
            nearest_psnr_db: psnr(nearest_error / channels as f64),
            bicubic_psnr_db: psnr(bicubic_error / channels as f64),
            direct_psnr_db: psnr(direct_error / channels as f64),
        },
        images,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_output_has_double_dimensions_and_roundtrips() {
        let low = RgbImage::from_pixel(4, 3, Rgb([120, 60, 20]));
        let model = DirectModel::new(42);
        let rendered = model.upscale(&low).unwrap();
        assert_eq!(rendered.dimensions(), (8, 6));
        assert!(rendered.pixels().all(|pixel| pixel.0 == [120, 60, 20]));
        let path = std::env::temp_dir().join(format!("qsd1-test-{}.qsd", std::process::id()));
        model.save_fp16(&path).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), 8 + 2 * PARAMS as u64);
        let loaded = DirectModel::load_fp16(&path).unwrap();
        assert_eq!(loaded.upscale(&low).unwrap().dimensions(), (8, 6));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn gradient_reduces_a_known_block_error() {
        let pair = Pair {
            low: RgbImage::from_pixel(2, 2, Rgb([100, 50, 10])),
            target: RgbImage::from_pixel(4, 4, Rgb([120, 60, 20])),
        };
        let mut model = DirectModel::new(42);
        let before = squared_error(&model.upscale(&pair.low).unwrap(), &pair.target);
        let mut optimizer = Adam::new();
        for _ in 0..100 {
            let mut gradient = [0.0; PARAMS];
            accumulate_gradient(&model, &pair, 0, 0, &mut gradient);
            optimizer.apply(&mut model, &gradient, 1, 0.001);
        }
        let after = squared_error(&model.upscale(&pair.low).unwrap(), &pair.target);
        assert!(after < before / 2.0, "before={before}, after={after}");
    }
}
