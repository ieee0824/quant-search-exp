//! Parameter-matched residual/direct models for grouped cross-validation.

use crate::fp6::{
    BLOCK_SIZE, Fp6Format, Rounding, decode_element, decode_scale, encode_default_scale,
    encode_element, pack_six, unpack_six,
};
use crate::{
    Result, Rng, bicubic_2x, crops_from_image, image_paths, load_rgb, psnr, squared_error,
};
use half::f16;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::{self, FilterType};
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use serde::Serialize;
use std::fs;
use std::path::Path;

const INPUTS: usize = 27;
const MAX_HIDDEN: usize = 64;
const MAX_OUTPUTS: usize = 12;
const MAGIC: &[u8; 4] = b"QSM1";
const FP6_MAGIC: &[u8; 4] = b"QSM6";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Residual,
    Direct,
}

impl Kind {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "residual" => Ok(Self::Residual),
            "direct" => Ok(Self::Direct),
            _ => Err("kind must be residual or direct".into()),
        }
    }

    fn outputs(self) -> usize {
        match self {
            Self::Residual => 3,
            Self::Direct => 12,
        }
    }

    fn code(self) -> u8 {
        match self {
            Self::Residual => 0,
            Self::Direct => 1,
        }
    }
}

struct Pair {
    low: RgbImage,
    base: RgbImage,
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
    let base = bicubic_2x(&low)?;
    Ok(Pair { low, base, target })
}

fn neighborhood(image: &RgbImage, x: u32, y: u32) -> [f32; INPUTS] {
    let mut input = [0.0; INPUTS];
    let mut index = 0;
    for dy in -1_i64..=1 {
        for dx in -1_i64..=1 {
            let px = (x as i64 + dx).clamp(0, image.width() as i64 - 1) as u32;
            let py = (y as i64 + dy).clamp(0, image.height() as i64 - 1) as u32;
            for &channel in &image.get_pixel(px, py).0 {
                input[index] = channel as f32 / 255.0;
                index += 1;
            }
        }
    }
    input
}

#[derive(Clone)]
pub struct MatchedModel {
    kind: Kind,
    hidden: usize,
    weights: Vec<f32>,
    precision: &'static str,
}

impl MatchedModel {
    fn b1(&self) -> usize {
        INPUTS * self.hidden
    }
    fn w2(&self) -> usize {
        self.b1() + self.hidden
    }
    fn b2(&self) -> usize {
        self.w2() + self.hidden * self.kind.outputs()
    }
    pub fn parameters(&self) -> usize {
        self.b2() + self.kind.outputs()
    }

    pub fn new(kind: Kind, hidden: usize, seed: u64) -> Result<Self> {
        if !(3..=MAX_HIDDEN).contains(&hidden) {
            return Err(format!("hidden width must be 3..={MAX_HIDDEN}").into());
        }
        let params = INPUTS * hidden + hidden + hidden * kind.outputs() + kind.outputs();
        let mut model = Self {
            kind,
            hidden,
            weights: vec![0.0; params],
            precision: "FP16",
        };
        let mut rng = Rng::new(seed);
        let b1 = model.b1();
        for weight in &mut model.weights[..b1] {
            *weight = rng.signed_unit() * (2.0 / INPUTS as f32).sqrt();
        }
        let w2 = model.w2();
        model.weights[b1..w2].fill(0.01);
        if kind == Kind::Direct {
            for channel in 0..3 {
                model.weights[channel * INPUTS..(channel + 1) * INPUTS].fill(0.0);
                model.weights[channel * INPUTS + 12 + channel] = 1.0;
                model.weights[b1 + channel] = 0.0;
                for pixel in 0..4 {
                    model.weights[w2 + (pixel * 3 + channel) * hidden + channel] = 1.0;
                }
            }
        }
        Ok(model)
    }

    fn forward(&self, input: &[f32; INPUTS]) -> ([f32; MAX_HIDDEN], [f32; MAX_OUTPUTS]) {
        let mut hidden = [0.0; MAX_HIDDEN];
        let b1 = self.b1();
        let w2 = self.w2();
        let b2 = self.b2();
        for (h, value) in hidden.iter_mut().enumerate().take(self.hidden) {
            let mut sum = self.weights[b1 + h];
            for (i, &input_value) in input.iter().enumerate() {
                sum += self.weights[h * INPUTS + i] * input_value;
            }
            *value = sum.max(0.0);
        }
        let mut output = [0.0; MAX_OUTPUTS];
        for (o, value) in output.iter_mut().enumerate().take(self.kind.outputs()) {
            let mut sum = self.weights[b2 + o];
            for (h, &activation) in hidden.iter().enumerate().take(self.hidden) {
                sum += self.weights[w2 + o * self.hidden + h] * activation;
            }
            *value = sum;
        }
        (hidden, output)
    }

    fn render(&self, pair: &Pair) -> RgbImage {
        let mut high = RgbImage::new(pair.target.width(), pair.target.height());
        match self.kind {
            Kind::Residual => {
                for y in 0..high.height() {
                    for x in 0..high.width() {
                        let (_, correction) = self.forward(&neighborhood(&pair.base, x, y));
                        let base = pair.base.get_pixel(x, y);
                        high.put_pixel(
                            x,
                            y,
                            Rgb(std::array::from_fn(|channel| {
                                ((base[channel] as f32 / 255.0 + correction[channel])
                                    .clamp(0.0, 1.0)
                                    * 255.0)
                                    .round() as u8
                            })),
                        );
                    }
                }
            }
            Kind::Direct => {
                for y in 0..pair.low.height() {
                    for x in 0..pair.low.width() {
                        let (_, output) = self.forward(&neighborhood(&pair.low, x, y));
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let pixel = (dy * 2 + dx) as usize;
                                high.put_pixel(
                                    2 * x + dx,
                                    2 * y + dy,
                                    Rgb(std::array::from_fn(|channel| {
                                        (output[pixel * 3 + channel].clamp(0.0, 1.0) * 255.0)
                                            .round() as u8
                                    })),
                                );
                            }
                        }
                    }
                }
            }
        }
        high
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if self
            .weights
            .iter()
            .any(|weight| !weight.is_finite() || weight.abs() > f16::MAX.to_f32())
        {
            return Err("weight cannot be represented as FP16".into());
        }
        let mut bytes = Vec::with_capacity(12 + 2 * self.parameters());
        bytes.extend_from_slice(MAGIC);
        bytes.push(self.kind.code());
        bytes.push(self.hidden as u8);
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&(self.parameters() as u32).to_le_bytes());
        for &weight in &self.weights {
            bytes.extend_from_slice(&f16::from_f32(weight).to_le_bytes());
        }
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() >= 4 && &bytes[..4] == FP6_MAGIC {
            return Self::load_fp6_bytes(&bytes);
        }
        if bytes.len() < 12 || &bytes[..4] != MAGIC || bytes[6..8] != [0, 0] {
            return Err("invalid QSM1 model".into());
        }
        let kind = match bytes[4] {
            0 => Kind::Residual,
            1 => Kind::Direct,
            _ => return Err("invalid QSM1 kind".into()),
        };
        let mut model = Self::new(kind, bytes[5] as usize, 1)?;
        if bytes.len() != 12 + 2 * model.parameters()
            || u32::from_le_bytes(bytes[8..12].try_into()?) as usize != model.parameters()
        {
            return Err("invalid QSM1 parameter count".into());
        }
        for (i, weight) in model.weights.iter_mut().enumerate() {
            *weight = f16::from_le_bytes(bytes[12 + 2 * i..14 + 2 * i].try_into()?).to_f32();
            if !weight.is_finite() {
                return Err("non-finite QSM1 weight".into());
            }
        }
        Ok(model)
    }

    fn connection_weights(&self) -> usize {
        self.b1() + self.hidden * self.kind.outputs()
    }

    fn connection_weight(&self, index: usize) -> f32 {
        if index < self.b1() {
            self.weights[index]
        } else {
            self.weights[self.w2() + index - self.b1()]
        }
    }

    fn set_connection_weight(&mut self, index: usize, value: f32) {
        let position = if index < self.b1() {
            index
        } else {
            self.w2() + index - self.b1()
        };
        self.weights[position] = value;
    }

    fn bias(&self, index: usize) -> f32 {
        if index < self.hidden {
            self.weights[self.b1() + index]
        } else {
            self.weights[self.b2() + index - self.hidden]
        }
    }

    fn set_bias(&mut self, index: usize, value: f32) {
        let position = if index < self.hidden {
            self.b1() + index
        } else {
            self.b2() + index - self.hidden
        };
        self.weights[position] = value;
    }

    pub fn save_fp6(&self, path: &Path, format: Fp6Format) -> Result<()> {
        if self.precision != "FP16" {
            return Err("FP6 source must be an FP16 QSM1 model".into());
        }
        let count = self.connection_weights();
        let biases = self.hidden + self.kind.outputs();
        let blocks = count.div_ceil(BLOCK_SIZE);
        let mut scales = Vec::with_capacity(blocks);
        let mut codes = Vec::with_capacity(count);
        for block in 0..blocks {
            let start = block * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(count);
            let original: Vec<_> = (start..end)
                .map(|index| self.connection_weight(index))
                .collect();
            let scale_code = encode_default_scale(&original, format, 0)?;
            let scale = decode_scale(scale_code)?;
            scales.push(scale_code);
            for value in original {
                codes.push(encode_element(
                    value / scale,
                    format,
                    Rounding::NearestEven,
                )?);
            }
        }
        let packed = pack_six(&codes)?;
        let mut bytes = Vec::with_capacity(16 + blocks + packed.len() + biases * 2);
        bytes.extend_from_slice(FP6_MAGIC);
        bytes.push(1);
        bytes.push(self.kind.code());
        bytes.push(self.hidden as u8);
        bytes.push(match format {
            Fp6Format::E3M2 => 0,
            Fp6Format::E2M3 => 1,
        });
        bytes.extend_from_slice(&(count as u16).to_le_bytes());
        bytes.extend_from_slice(&(biases as u16).to_le_bytes());
        bytes.extend_from_slice(&(blocks as u16).to_le_bytes());
        bytes.extend_from_slice(&0_u16.to_le_bytes());
        bytes.extend_from_slice(&scales);
        bytes.extend_from_slice(&packed);
        for index in 0..biases {
            let value = self.bias(index);
            if !value.is_finite() || value.abs() > f16::MAX.to_f32() {
                return Err("bias cannot be stored as FP16".into());
            }
            bytes.extend_from_slice(&f16::from_f32(value).to_le_bytes());
        }
        fs::write(path, bytes)?;
        Ok(())
    }

    fn load_fp6_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 || bytes[4] != 1 || bytes[14..16] != [0, 0] {
            return Err("invalid QSM6 header".into());
        }
        let kind = match bytes[5] {
            0 => Kind::Residual,
            1 => Kind::Direct,
            _ => return Err("invalid QSM6 kind".into()),
        };
        let format = match bytes[7] {
            0 => Fp6Format::E3M2,
            1 => Fp6Format::E2M3,
            _ => return Err("invalid QSM6 format".into()),
        };
        let mut model = Self::new(kind, bytes[6] as usize, 1)?;
        let count = model.connection_weights();
        let biases = model.hidden + kind.outputs();
        let blocks = count.div_ceil(BLOCK_SIZE);
        let packed_len = (count * 6).div_ceil(8);
        if u16::from_le_bytes(bytes[8..10].try_into()?) as usize != count
            || u16::from_le_bytes(bytes[10..12].try_into()?) as usize != biases
            || u16::from_le_bytes(bytes[12..14].try_into()?) as usize != blocks
            || bytes.len() != 16 + blocks + packed_len + biases * 2
        {
            return Err("invalid QSM6 size".into());
        }
        let scales = &bytes[16..16 + blocks];
        let codes = unpack_six(&bytes[16 + blocks..16 + blocks + packed_len], count)?;
        for (index, code) in codes.into_iter().enumerate() {
            let value = decode_element(code, format) * decode_scale(scales[index / BLOCK_SIZE])?;
            if !value.is_finite() {
                return Err("non-finite QSM6 weight".into());
            }
            model.set_connection_weight(index, value);
        }
        let offset = 16 + blocks + packed_len;
        for index in 0..biases {
            let value =
                f16::from_le_bytes(bytes[offset + index * 2..offset + index * 2 + 2].try_into()?)
                    .to_f32();
            if !value.is_finite() {
                return Err("non-finite QSM6 bias".into());
            }
            model.set_bias(index, value);
        }
        model.precision = match format {
            Fp6Format::E3M2 => "MXFP6-E3M2",
            Fp6Format::E2M3 => "MXFP6-E2M3",
        };
        Ok(model)
    }
}

pub fn quantize_fp6(source: &Path, destination: &Path, format: Fp6Format) -> Result<()> {
    if source == destination {
        return Err("source and destination must differ".into());
    }
    MatchedModel::load(source)?.save_fp6(destination, format)
}

struct Adam {
    mean: Vec<f32>,
    variance: Vec<f32>,
    step: i32,
}

impl Adam {
    fn new(count: usize) -> Self {
        Self {
            mean: vec![0.0; count],
            variance: vec![0.0; count],
            step: 0,
        }
    }

    fn apply(&mut self, model: &mut MatchedModel, gradient: &[f32], count: usize, rate: f32) {
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
    model: &MatchedModel,
    pair: &Pair,
    x: u32,
    y: u32,
    gradient: &mut [f32],
) -> f32 {
    let input = match model.kind {
        Kind::Residual => neighborhood(&pair.base, x, y),
        Kind::Direct => neighborhood(&pair.low, x, y),
    };
    let (hidden, output) = model.forward(&input);
    let mut delta = [0.0; MAX_OUTPUTS];
    let outputs = model.kind.outputs();
    let mut loss = 0.0;
    let w2 = model.w2();
    let b2 = model.b2();
    for o in 0..outputs {
        let wanted = match model.kind {
            Kind::Residual => {
                let target = pair.target.get_pixel(x, y)[o];
                let base = pair.base.get_pixel(x, y)[o];
                (target as f32 - base as f32) / 255.0
            }
            Kind::Direct => {
                let pixel = o / 3;
                let channel = o % 3;
                let target = pair
                    .target
                    .get_pixel(2 * x + (pixel % 2) as u32, 2 * y + (pixel / 2) as u32)[channel];
                target as f32 / 255.0
            }
        };
        let error = output[o] - wanted;
        loss += error * error / outputs as f32;
        delta[o] = 2.0 * error / outputs as f32;
        gradient[b2 + o] += delta[o];
        for (h, &activation) in hidden.iter().enumerate().take(model.hidden) {
            gradient[w2 + o * model.hidden + h] += delta[o] * activation;
        }
    }
    for (h, &activation) in hidden.iter().enumerate().take(model.hidden) {
        if activation == 0.0 {
            continue;
        }
        let mut upstream = 0.0;
        for (o, &derivative) in delta.iter().enumerate().take(outputs) {
            upstream += model.weights[w2 + o * model.hidden + h] * derivative;
        }
        gradient[model.b1() + h] += upstream;
        for (i, &value) in input.iter().enumerate() {
            gradient[h * INPUTS + i] += upstream * value;
        }
    }
    loss
}

pub struct TrainOptions {
    pub kind: Kind,
    pub hidden: usize,
    pub epochs: usize,
    pub target_pixels_per_epoch: usize,
    pub quality: u8,
    pub seed: u64,
    pub learning_rate: f32,
}

pub fn train(directory: &Path, model_path: &Path, options: TrainOptions) -> Result<()> {
    let TrainOptions {
        kind,
        hidden,
        epochs,
        target_pixels_per_epoch,
        quality,
        seed,
        learning_rate,
    } = options;
    if epochs == 0
        || target_pixels_per_epoch == 0
        || !target_pixels_per_epoch.is_multiple_of(4)
        || !(1..=100).contains(&quality)
        || !learning_rate.is_finite()
        || learning_rate <= 0.0
    {
        return Err("invalid matched training settings".into());
    }
    let mut pairs = Vec::new();
    for path in image_paths(directory)? {
        let source = load_rgb(&path)?;
        for crop in crops_from_image(&source, true)? {
            pairs.push(make_pair(&crop, quality)?);
        }
    }
    println!(
        "loaded {} crops from {} images",
        pairs.len(),
        image_paths(directory)?.len()
    );
    let mut model = MatchedModel::new(kind, hidden, seed)?;
    let mut optimizer = Adam::new(model.parameters());
    let mut rng = Rng::new(seed ^ 0x9e37_79b9_7f4a_7c15);
    let samples = match kind {
        Kind::Residual => target_pixels_per_epoch,
        Kind::Direct => target_pixels_per_epoch / 4,
    };
    let batch = match kind {
        Kind::Residual => 256,
        Kind::Direct => 64,
    };
    for epoch in 0..epochs {
        let mut loss = 0.0_f64;
        let mut remaining = samples;
        while remaining > 0 {
            let count = remaining.min(batch);
            let mut gradient = vec![0.0; model.parameters()];
            for _ in 0..count {
                let pair = &pairs[rng.index(pairs.len())];
                let image = match kind {
                    Kind::Residual => &pair.target,
                    Kind::Direct => &pair.low,
                };
                let x = rng.index(image.width() as usize) as u32;
                let y = rng.index(image.height() as usize) as u32;
                loss += accumulate_gradient(&model, pair, x, y, &mut gradient) as f64;
            }
            optimizer.apply(&mut model, &gradient, count, learning_rate);
            remaining -= count;
        }
        println!(
            "epoch {}/{}: sampled MSE {:.8}",
            epoch + 1,
            epochs,
            loss / samples as f64
        );
    }
    model.save(model_path)?;
    println!(
        "saved {:?} hidden={} params={} model={}",
        kind,
        hidden,
        model.parameters(),
        model_path.display()
    );
    Ok(())
}

#[derive(Serialize)]
pub struct ImageMetrics {
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub bicubic_psnr_db: f64,
    pub model_psnr_db: f64,
}

#[derive(Serialize)]
pub struct Report {
    pub kind: Kind,
    pub precision: &'static str,
    pub hidden: usize,
    pub parameters: usize,
    pub model_bytes: u64,
    pub directory: String,
    pub quality: u8,
    pub images: Vec<ImageMetrics>,
    pub bicubic_psnr_db: f64,
    pub model_psnr_db: f64,
}

pub fn evaluate(model_path: &Path, directory: &Path, quality: u8) -> Result<Report> {
    if !(1..=100).contains(&quality) {
        return Err("JPEG quality must be 1..100".into());
    }
    let model = MatchedModel::load(model_path)?;
    let mut images = Vec::new();
    let mut base_error = 0.0;
    let mut model_error = 0.0;
    let mut channels = 0_u64;
    for path in image_paths(directory)? {
        let source = load_rgb(&path)?;
        let crop = crops_from_image(&source, false)?.remove(0);
        let pair = make_pair(&crop, quality)?;
        let rendered = model.render(&pair);
        let b = squared_error(&pair.base, &pair.target);
        let m = squared_error(&rendered, &pair.target);
        let count = pair.target.width() as u64 * pair.target.height() as u64 * 3;
        base_error += b;
        model_error += m;
        channels += count;
        images.push(ImageMetrics {
            path: path.display().to_string(),
            width: pair.target.width(),
            height: pair.target.height(),
            bicubic_psnr_db: psnr(b / count as f64),
            model_psnr_db: psnr(m / count as f64),
        });
    }
    Ok(Report {
        kind: model.kind,
        precision: model.precision,
        hidden: model.hidden,
        parameters: model.parameters(),
        model_bytes: fs::metadata(model_path)?.len(),
        directory: directory.display().to_string(),
        quality,
        bicubic_psnr_db: psnr(base_error / channels as f64),
        model_psnr_db: psnr(model_error / channels as f64),
        images,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_sizes_and_initial_outputs() {
        let residual = MatchedModel::new(Kind::Residual, 21, 42).unwrap();
        let direct = MatchedModel::new(Kind::Direct, 16, 42).unwrap();
        assert_eq!(residual.parameters(), 654);
        assert_eq!(direct.parameters(), 652);
        assert_eq!(
            MatchedModel::new(Kind::Residual, 16, 42)
                .unwrap()
                .parameters(),
            499
        );
        assert_eq!(
            MatchedModel::new(Kind::Direct, 12, 42)
                .unwrap()
                .parameters(),
            492
        );
        let target = RgbImage::from_pixel(4, 4, Rgb([120, 60, 20]));
        let pair = make_pair(&target, 70).unwrap();
        assert_eq!(direct.render(&pair).dimensions(), target.dimensions());
        assert_eq!(residual.render(&pair), pair.base);
        assert_eq!(
            direct.render(&pair),
            imageops::resize(&pair.low, 4, 4, FilterType::Nearest)
        );
    }

    #[test]
    fn fp6_roundtrip_preserves_structure_and_rejects_corruption() {
        let path = std::env::temp_dir().join(format!(
            "qse-matched-fp6-{}-{}.qsm6",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let model = MatchedModel::new(Kind::Direct, 16, 42).unwrap();
        model.save_fp6(&path, Fp6Format::E2M3).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), 560);
        let loaded = MatchedModel::load(&path).unwrap();
        assert_eq!(loaded.parameters(), 652);
        assert_eq!(loaded.precision, "MXFP6-E2M3");
        assert!(loaded.weights.iter().all(|value| value.is_finite()));
        let target = RgbImage::from_pixel(4, 4, Rgb([120, 60, 20]));
        let pair = make_pair(&target, 70).unwrap();
        assert_eq!(loaded.render(&pair), model.render(&pair));
        let mut bytes = fs::read(&path).unwrap();
        bytes[16] = 255;
        fs::write(&path, bytes).unwrap();
        assert!(MatchedModel::load(&path).is_err());
        fs::remove_file(path).unwrap();
    }
}
