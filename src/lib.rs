use half::f16;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::{self, FilterType};
use image::{DynamicImage, ImageFormat, ImageReader, Rgb, RgbImage};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub mod evaluation;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const INPUTS: usize = 27; // 3x3 RGB neighborhood
const HIDDEN: usize = 16;
const B1: usize = INPUTS * HIDDEN;
const W2: usize = B1 + HIDDEN;
const B2: usize = W2 + 3 * HIDDEN;
const PARAMS: usize = B2 + 3;
const BATCH: usize = 64;

fn dot_scalar(weights: &[f32], input: &[f32], bias: f32) -> f32 {
    let mut result = bias;
    for (&weight, &value) in weights.iter().zip(input) {
        result += weight * value;
    }
    result
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn dot_neon(weights: &[f32], input: &[f32], bias: f32) -> f32 {
    use core::arch::aarch64::{vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    debug_assert_eq!(weights.len(), input.len());
    // This function is compiled only when NEON is enabled for the target.
    let mut vector = unsafe { vdupq_n_f32(0.0) };
    let lanes = weights.len() / 4 * 4;
    for i in (0..lanes).step_by(4) {
        // Each load reads four elements inside the checked slice bounds.
        unsafe {
            let w = vld1q_f32(weights.as_ptr().add(i));
            let x = vld1q_f32(input.as_ptr().add(i));
            vector = vfmaq_f32(vector, w, x);
        }
    }
    let mut result = bias + unsafe { vaddvq_f32(vector) };
    for i in lanes..weights.len() {
        result += weights[i] * input[i];
    }
    result
}

#[derive(Clone)]
pub struct Model {
    weights: [f32; PARAMS],
}

pub struct Pair {
    pub base: RgbImage,
    pub target: RgbImage,
}

pub struct Metrics {
    pub images: usize,
    pub baseline_psnr: f64,
    pub model_psnr: f64,
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn index(&mut self, upper: usize) -> usize {
        (self.next() % upper as u64) as usize
    }
    fn signed_unit(&mut self) -> f32 {
        (self.next() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
    }
}

impl Model {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut weights = [0.0; PARAMS];
        for weight in &mut weights[..B1] {
            *weight = rng.signed_unit() * (2.0 / INPUTS as f32).sqrt();
        }
        weights[B1..W2].fill(0.01);
        Self { weights }
    }

    // The model predicts an RGB correction to the bicubic pixel.
    fn forward_with<D>(&self, input: &[f32; INPUTS], dot: D) -> ([f32; HIDDEN], [f32; 3])
    where
        D: Fn(&[f32], &[f32], f32) -> f32,
    {
        let mut hidden = [0.0; HIDDEN];
        for (h, activation) in hidden.iter_mut().enumerate() {
            *activation = dot(
                &self.weights[h * INPUTS..(h + 1) * INPUTS],
                input,
                self.weights[B1 + h],
            )
            .max(0.0);
        }
        let mut output = [0.0; 3];
        for (c, channel) in output.iter_mut().enumerate() {
            *channel = dot(
                &self.weights[W2 + c * HIDDEN..W2 + (c + 1) * HIDDEN],
                &hidden,
                self.weights[B2 + c],
            );
        }
        (hidden, output)
    }

    fn forward(&self, input: &[f32; INPUTS]) -> ([f32; HIDDEN], [f32; 3]) {
        self.forward_with(input, dot_scalar)
    }

    pub fn simd_enabled() -> bool {
        cfg!(all(target_arch = "aarch64", target_feature = "neon"))
    }

    pub fn enhance(&self, base: &RgbImage) -> RgbImage {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            self.enhance_with(base, |input| self.forward_with(input, dot_neon).1)
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            self.enhance_scalar(base)
        }
    }

    pub fn enhance_scalar(&self, base: &RgbImage) -> RgbImage {
        self.enhance_with(base, |input| self.forward(input).1)
    }

    fn enhance_with<F>(&self, base: &RgbImage, forward: F) -> RgbImage
    where
        F: Fn(&[f32; INPUTS]) -> [f32; 3],
    {
        let mut output = RgbImage::new(base.width(), base.height());
        for y in 0..base.height() {
            for x in 0..base.width() {
                let correction = forward(&neighborhood(base, x, y));
                let center = base.get_pixel(x, y);
                let rgb = std::array::from_fn(|c| {
                    ((center[c] as f32 / 255.0 + correction[c]).clamp(0.0, 1.0) * 255.0).round()
                        as u8
                });
                output.put_pixel(x, y, Rgb(rgb));
            }
        }
        output
    }

    pub fn save_fp16(&self, path: &Path) -> Result<()> {
        if self
            .weights
            .iter()
            .any(|w| !w.is_finite() || w.abs() > f16::MAX.to_f32())
        {
            return Err("model contains a weight that cannot be stored as FP16".into());
        }
        let mut file = File::create(path)?;
        file.write_all(b"QSR1")?;
        file.write_all(&(PARAMS as u32).to_le_bytes())?;
        for &weight in &self.weights {
            file.write_all(&f16::from_f32(weight).to_le_bytes())?;
        }
        Ok(())
    }

    pub fn load_fp16(path: &Path) -> Result<Self> {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        if bytes.len() != 8 + PARAMS * 2 || &bytes[..4] != b"QSR1" {
            return Err("invalid QSR1 model file".into());
        }
        if u32::from_le_bytes(bytes[4..8].try_into()?) as usize != PARAMS {
            return Err("model architecture does not match this program".into());
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

pub fn bicubic_2x(image: &RgbImage) -> Result<RgbImage> {
    let width = image
        .width()
        .checked_mul(2)
        .ok_or("image width overflows")?;
    let height = image
        .height()
        .checked_mul(2)
        .ok_or("image height overflows")?;
    Ok(imageops::resize(
        image,
        width,
        height,
        FilterType::CatmullRom,
    ))
}

pub fn make_pair(target: &RgbImage, quality: u8) -> Result<Pair> {
    if target.width() < 4 || target.height() < 4 || !(1..=100).contains(&quality) {
        return Err("image must be at least 4x4 and JPEG quality must be 1..100".into());
    }
    let (width, height) = (target.width() & !1, target.height() & !1);
    let target = imageops::crop_imm(target, 0, 0, width, height).to_image();
    let low = imageops::resize(&target, width / 2, height / 2, FilterType::CatmullRom);
    let mut encoded = Vec::new();
    DynamicImage::ImageRgb8(low)
        .write_with_encoder(JpegEncoder::new_with_quality(&mut encoded, quality))?;
    let low = image::load_from_memory_with_format(&encoded, ImageFormat::Jpeg)?.to_rgb8();
    Ok(Pair {
        base: bicubic_2x(&low)?,
        target,
    })
}

pub fn image_paths(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png") {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(format!("no JPEG or PNG images in {}", directory.display()).into());
    }
    Ok(paths)
}

pub fn load_rgb(path: &Path) -> Result<RgbImage> {
    Ok(ImageReader::open(path)?.decode()?.to_rgb8())
}

fn image_crops(path: &Path, training: bool) -> Result<Vec<RgbImage>> {
    let image = load_rgb(path)?;
    crops_from_image(&image, training)
}

fn crops_from_image(image: &RgbImage, training: bool) -> Result<Vec<RgbImage>> {
    if image.width() < 4 || image.height() < 4 {
        return Err("image is smaller than 4x4".into());
    }
    let width = image.width().min(512) & !1;
    let height = image.height().min(512) & !1;
    let last_x = image.width() - width;
    let last_y = image.height() - height;
    let center = (last_x / 2, last_y / 2);
    let mut positions = if training {
        vec![(0, 0), center, (last_x, last_y)]
    } else {
        vec![center]
    };
    positions.dedup();
    Ok(positions
        .into_iter()
        .map(|(x, y)| imageops::crop_imm(image, x, y, width, height).to_image())
        .collect())
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

    fn apply(&mut self, model: &mut Model, gradient: &[f32; PARAMS], count: usize, rate: f32) {
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
    model: &Model,
    pair: &Pair,
    x: u32,
    y: u32,
    gradient: &mut [f32; PARAMS],
) -> f32 {
    let input = neighborhood(&pair.base, x, y);
    let (hidden, output) = model.forward(&input);
    let base = pair.base.get_pixel(x, y);
    let target = pair.target.get_pixel(x, y);
    let mut delta = [0.0; 3];
    let mut loss = 0.0;
    for (c, delta_c) in delta.iter_mut().enumerate() {
        let wanted = (target[c] as f32 - base[c] as f32) / 255.0;
        let error = output[c] - wanted;
        loss += error * error / 3.0;
        *delta_c = 2.0 * error / 3.0;
        gradient[B2 + c] += *delta_c;
        for (h, activation) in hidden.iter().enumerate() {
            gradient[W2 + c * HIDDEN + h] += *delta_c * activation;
        }
    }
    for h in 0..HIDDEN {
        if hidden[h] == 0.0 {
            continue;
        }
        let mut upstream = 0.0;
        for (c, &delta_c) in delta.iter().enumerate() {
            upstream += model.weights[W2 + c * HIDDEN + h] * delta_c;
        }
        gradient[B1 + h] += upstream;
        for (i, pixel) in input.iter().enumerate() {
            gradient[h * INPUTS + i] += upstream * pixel;
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
        return Err(
            "epochs, samples and learning rate must be positive; JPEG quality must be 1..100"
                .into(),
        );
    }
    let mut pairs = Vec::new();
    for path in image_paths(directory)? {
        let crops = image_crops(&path, true)?;
        println!("loaded {} ({} crops)", path.display(), crops.len());
        for crop in crops {
            pairs.push(make_pair(&crop, quality)?);
        }
    }
    let mut model = Model::new(seed);
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
                let x = rng.index(pair.target.width() as usize) as u32;
                let y = rng.index(pair.target.height() as usize) as u32;
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
        "saved FP16 model: {} ({} parameters)",
        model_path.display(),
        PARAMS
    );
    Ok(())
}

fn squared_error(a: &RgbImage, b: &RgbImage) -> f64 {
    a.pixels()
        .zip(b.pixels())
        .map(|(left, right)| {
            left.0
                .iter()
                .zip(right.0.iter())
                .map(|(&x, &y)| {
                    let difference = (x as f64 - y as f64) / 255.0;
                    difference * difference
                })
                .sum::<f64>()
        })
        .sum()
}

fn psnr(mse: f64) -> f64 {
    if mse == 0.0 {
        f64::INFINITY
    } else {
        -10.0 * mse.log10()
    }
}

pub fn evaluate(model: &Model, directory: &Path, quality: u8) -> Result<Metrics> {
    if !(1..=100).contains(&quality) {
        return Err("JPEG quality must be 1..100".into());
    }
    let paths = image_paths(directory)?;
    let mut baseline_error = 0.0;
    let mut model_error = 0.0;
    let mut channels = 0_u64;
    for path in &paths {
        let crop = image_crops(path, false)?.remove(0);
        let pair = make_pair(&crop, quality)?;
        let output = model.enhance(&pair.base);
        baseline_error += squared_error(&pair.base, &pair.target);
        model_error += squared_error(&output, &pair.target);
        channels += pair.target.width() as u64 * pair.target.height() as u64 * 3;
    }
    Ok(Metrics {
        images: paths.len(),
        baseline_psnr: psnr(baseline_error / channels as f64),
        model_psnr: psnr(model_error / channels as f64),
    })
}

pub fn upscale(model: &Model, input: &Path, output: &Path) -> Result<()> {
    if input == output {
        return Err("input and output paths must differ".into());
    }
    let extension = output.extension().and_then(|s| s.to_str()).unwrap_or("");
    if !matches!(extension.to_ascii_lowercase().as_str(), "jpg" | "jpeg") {
        return Err("output path must end in .jpg or .jpeg".into());
    }
    let low = load_rgb(input)?;
    let enhanced = model.enhance(&bicubic_2x(&low)?);
    let mut file = File::create(output)?;
    DynamicImage::ImageRgb8(enhanced)
        .write_with_encoder(JpegEncoder::new_with_quality(&mut file, 95))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_images_have_the_target_dimensions() {
        let target = RgbImage::from_fn(12, 10, |x, y| Rgb([(x * 20) as u8, (y * 20) as u8, 90]));
        let pair = make_pair(&target, 70).unwrap();
        assert_eq!(pair.base.dimensions(), (12, 10));
        assert_eq!(pair.target.dimensions(), (12, 10));
    }

    #[test]
    fn fp16_model_roundtrip_preserves_inference() {
        let path =
            std::env::temp_dir().join(format!("quant-search-exp-{}.qsr", std::process::id()));
        let mut model = Model::new(42);
        model.weights[W2] = 0.125;
        model.save_fp16(&path).unwrap();
        let loaded = Model::load_fp16(&path).unwrap();
        fs::remove_file(path).unwrap();
        for (original, restored) in model.weights.iter().zip(loaded.weights.iter()) {
            assert_eq!(*restored, f16::from_f32(*original).to_f32());
        }
        let image = RgbImage::from_pixel(2, 2, Rgb([100, 120, 140]));
        assert_eq!(model.enhance(&image), loaded.enhance(&image));
    }

    #[test]
    fn simd_and_scalar_inference_agree_to_one_color_level() {
        let mut model = Model::new(7);
        for (i, weight) in model.weights[W2..].iter_mut().enumerate() {
            *weight = (i as f32 * 0.17).sin() * 0.05;
        }
        let image = RgbImage::from_fn(65, 63, |x, y| {
            Rgb([
                ((x * 7 + y * 3) % 256) as u8,
                ((x * 11 + y * 5) % 256) as u8,
                ((x * 13 + y * 17) % 256) as u8,
            ])
        });
        let scalar = model.enhance_scalar(&image);
        let simd = model.enhance(&image);
        for (left, right) in scalar.pixels().zip(simd.pixels()) {
            for (&a, &b) in left.0.iter().zip(right.0.iter()) {
                assert!(a.abs_diff(b) <= 1, "scalar {a}, SIMD {b}");
            }
        }
    }

    #[test]
    fn gradient_reduces_a_constant_residual() {
        let base = RgbImage::from_pixel(4, 4, Rgb([100, 100, 100]));
        let target = RgbImage::from_pixel(4, 4, Rgb([110, 110, 110]));
        let pair = Pair { base, target };
        let mut model = Model::new(1);
        let mut adam = Adam::new();
        let initial = {
            let mut gradient = [0.0; PARAMS];
            accumulate_gradient(&model, &pair, 1, 1, &mut gradient)
        };
        for _ in 0..100 {
            let mut gradient = [0.0; PARAMS];
            accumulate_gradient(&model, &pair, 1, 1, &mut gradient);
            adam.apply(&mut model, &gradient, 1, 0.003);
        }
        let mut gradient = [0.0; PARAMS];
        let final_loss = accumulate_gradient(&model, &pair, 1, 1, &mut gradient);
        assert!(final_loss < initial * 0.1, "{initial} -> {final_loss}");
    }

    #[test]
    fn jpeg_training_evaluation_and_upscaling_work_together() {
        let root = std::env::temp_dir().join(format!(
            "quant-search-exp-smoke-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let train_dir = root.join("train");
        let test_dir = root.join("test");
        fs::create_dir_all(&train_dir).unwrap();
        fs::create_dir_all(&test_dir).unwrap();
        for (directory, phase) in [(&train_dir, 0), (&test_dir, 3)] {
            let image = RgbImage::from_fn(24, 24, |x, y| {
                Rgb([
                    ((x * 9 + phase) % 256) as u8,
                    ((y * 9 + phase) % 256) as u8,
                    (((x + y) * 5 + phase) % 256) as u8,
                ])
            });
            image.save(directory.join("example.png")).unwrap();
        }
        let model_path = root.join("model.qsr");
        train(&train_dir, &model_path, 5, 1_000, 70, 42, 0.001).unwrap();
        let model = Model::load_fp16(&model_path).unwrap();
        let metrics = evaluate(&model, &test_dir, 70).unwrap();
        assert_eq!(metrics.images, 1);
        assert!(metrics.baseline_psnr.is_finite());
        assert!(metrics.model_psnr.is_finite());
        assert!(
            metrics.model_psnr > metrics.baseline_psnr,
            "bicubic {} dB, model {} dB",
            metrics.baseline_psnr,
            metrics.model_psnr
        );
        let input_path = root.join("input.jpg");
        let input = RgbImage::from_pixel(12, 12, Rgb([100, 120, 140]));
        DynamicImage::ImageRgb8(input)
            .write_with_encoder(JpegEncoder::new_with_quality(
                File::create(&input_path).unwrap(),
                70,
            ))
            .unwrap();
        let output_path = root.join("output.jpg");
        upscale(&model, &input_path, &output_path).unwrap();
        assert_eq!(load_rgb(&output_path).unwrap().dimensions(), (24, 24));
        fs::remove_dir_all(root).unwrap();
    }
}
