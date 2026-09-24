//! OCP MXFP6 element encodings in a versioned, project-specific model container.
//!
//! Numeric E3M2/E2M3 and E8M0 semantics follow OCP MX v1.0. The QSF1 byte
//! layout and little-endian six-bit packing are defined by this project, not OCP.

use crate::{B1, B2, Model, PARAMS, Result, W2};
use half::f16;
use image::RgbImage;
use std::fs;
use std::path::Path;

pub const BLOCK_SIZE: usize = 32;
pub const QUANT_WEIGHTS: usize = B1 + (B2 - W2); // 432 + 48 = 480
pub const BIASES: usize = PARAMS - QUANT_WEIGHTS; // 16 + 3 = 19
pub const BLOCKS: usize = QUANT_WEIGHTS.div_ceil(BLOCK_SIZE);
const PACKED_BYTES: usize = (QUANT_WEIGHTS * 6).div_ceil(8);
const TAG_BYTES: usize = BLOCKS.div_ceil(8);
const HEADER_BYTES: usize = 16;
const FILE_BYTES: usize = HEADER_BYTES + BLOCKS + TAG_BYTES + PACKED_BYTES + BIASES * 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fp6Format {
    E3M2,
    E2M3,
}

impl Fp6Format {
    pub fn name(self) -> &'static str {
        match self {
            Self::E3M2 => "E3M2",
            Self::E2M3 => "E2M3",
        }
    }

    fn exponent_bits(self) -> u32 {
        match self {
            Self::E3M2 => 3,
            Self::E2M3 => 2,
        }
    }

    fn mantissa_bits(self) -> u32 {
        5 - self.exponent_bits()
    }

    fn bias(self) -> i32 {
        match self {
            Self::E3M2 => 3,
            Self::E2M3 => 1,
        }
    }

    fn max_power_exponent(self) -> i32 {
        match self {
            Self::E3M2 => 4, // 16 is the largest represented power of two.
            Self::E2M3 => 2, // 4 is the largest represented power of two.
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rounding {
    NearestEven,
    TowardZero,
    AwayFromZero,
}

#[derive(Clone, Copy, Debug)]
pub struct BlockOptions {
    pub scale_offset: i8,
    pub clip_ratio: f32,
    pub rounding: Rounding,
}

impl Default for BlockOptions {
    fn default() -> Self {
        Self {
            scale_offset: 0,
            clip_ratio: 1.0,
            rounding: Rounding::NearestEven,
        }
    }
}

pub fn decode_element(code: u8, format: Fp6Format) -> f32 {
    let mantissa_bits = format.mantissa_bits();
    let exponent = ((code & 31) >> mantissa_bits) as i32;
    let mantissa = (code & ((1 << mantissa_bits) - 1)) as f32;
    let magnitude = if exponent == 0 {
        2.0_f32.powi(1 - format.bias()) * mantissa / (1 << mantissa_bits) as f32
    } else {
        2.0_f32.powi(exponent - format.bias()) * (1.0 + mantissa / (1 << mantissa_bits) as f32)
    };
    if code & 32 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

pub fn encode_element(value: f32, format: Fp6Format, rounding: Rounding) -> Result<u8> {
    if value.is_nan() {
        return Err("NaN cannot be encoded as FP6".into());
    }
    let sign = if value.is_sign_negative() { 32 } else { 0 };
    let magnitude = value.abs() as f64;
    if magnitude >= decode_element(31, format) as f64 {
        return Ok(sign | 31);
    }
    let selected = match rounding {
        Rounding::NearestEven => {
            let mut best = 0_u8;
            let mut distance = f64::INFINITY;
            for code in 0..32_u8 {
                let error = (magnitude - decode_element(code, format) as f64).abs();
                if error < distance || (error == distance && code & 1 == 0) {
                    best = code;
                    distance = error;
                }
            }
            best
        }
        Rounding::TowardZero => (0..32_u8)
            .rev()
            .find(|&code| decode_element(code, format) as f64 <= magnitude)
            .unwrap_or(0),
        Rounding::AwayFromZero => (0..32_u8)
            .find(|&code| decode_element(code, format) as f64 >= magnitude)
            .unwrap_or(31),
    };
    Ok(sign | selected)
}

pub fn pack_six(values: &[u8]) -> Result<Vec<u8>> {
    let mut bytes = vec![0_u8; (values.len() * 6).div_ceil(8)];
    for (index, &code) in values.iter().enumerate() {
        if code >= 64 {
            return Err("FP6 code exceeds six bits".into());
        }
        let bit = index * 6;
        let byte = bit / 8;
        let shift = bit % 8;
        bytes[byte] |= code << shift;
        if shift > 2 {
            bytes[byte + 1] |= code >> (8 - shift);
        }
    }
    Ok(bytes)
}

pub fn unpack_six(bytes: &[u8], count: usize) -> Result<Vec<u8>> {
    if bytes.len() != (count * 6).div_ceil(8) {
        return Err("packed FP6 length does not match value count".into());
    }
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let bit = index * 6;
        let byte = bit / 8;
        let shift = bit % 8;
        let word = bytes[byte] as u16 | ((bytes.get(byte + 1).copied().unwrap_or(0) as u16) << 8);
        values.push(((word >> shift) & 63) as u8);
    }
    if !(count * 6).is_multiple_of(8) {
        let used = count * 6 % 8;
        if bytes.last().unwrap() >> used != 0 {
            return Err("nonzero FP6 padding bits".into());
        }
    }
    Ok(values)
}

pub fn decode_scale(exponent: u8) -> Result<f32> {
    if exponent == 255 {
        return Err("E8M0 255 is reserved for NaN".into());
    }
    Ok(2.0_f32.powi(exponent as i32 - 127))
}

fn encode_default_scale(weights: &[f32], format: Fp6Format, offset: i8) -> Result<u8> {
    if weights.iter().any(|x| !x.is_finite()) {
        return Err("non-finite model weight".into());
    }
    let max_abs = weights
        .iter()
        .fold(0.0_f32, |max, value| max.max(value.abs()));
    // Section 6.3 of OCP MX v1.0: largest power of two <= amax, divided by
    // the largest power of two representable by the chosen element format.
    let base = if max_abs == 0.0 {
        0
    } else {
        max_abs.log2().floor() as i32 - format.max_power_exponent()
    };
    Ok(((base + offset as i32).clamp(-127, 127) + 127) as u8)
}

fn source_weight(model: &Model, index: usize) -> f32 {
    if index < B1 {
        model.weights[index]
    } else {
        model.weights[W2 + index - B1]
    }
}

fn source_bias(model: &Model, index: usize) -> f32 {
    if index < W2 - B1 {
        model.weights[B1 + index]
    } else {
        model.weights[B2 + index - (W2 - B1)]
    }
}

#[derive(Clone)]
pub struct Fp6Model {
    formats: [Fp6Format; BLOCKS],
    scales: [u8; BLOCKS],
    packed: Vec<u8>,
    biases: [u16; BIASES],
}

impl Fp6Model {
    pub fn uniform(source: &Model, format: Fp6Format) -> Result<Self> {
        let mut result = Self {
            formats: [format; BLOCKS],
            scales: [127; BLOCKS],
            packed: vec![0; PACKED_BYTES],
            biases: [0; BIASES],
        };
        for index in 0..BIASES {
            let value = source_bias(source, index);
            if !value.is_finite() || value.abs() > f16::MAX.to_f32() {
                return Err("bias cannot be stored as FP16".into());
            }
            result.biases[index] = f16::from_f32(value).to_bits();
        }
        for block in 0..BLOCKS {
            result.set_block_from_source(source, block, format, BlockOptions::default())?;
        }
        Ok(result)
    }

    pub fn set_block_from_source(
        &mut self,
        source: &Model,
        block: usize,
        format: Fp6Format,
        options: BlockOptions,
    ) -> Result<()> {
        if block >= BLOCKS
            || !options.clip_ratio.is_finite()
            || !(0.0..=1.0).contains(&options.clip_ratio)
            || options.clip_ratio == 0.0
        {
            return Err("invalid FP6 block or clip ratio".into());
        }
        let start = block * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(QUANT_WEIGHTS);
        let original: Vec<_> = (start..end)
            .map(|index| source_weight(source, index))
            .collect();
        let scale_code = encode_default_scale(&original, format, options.scale_offset)?;
        let scale = decode_scale(scale_code)?;
        let max_abs = original.iter().fold(0.0_f32, |max, x| max.max(x.abs()));
        let clipping = max_abs * options.clip_ratio;
        let mut codes = unpack_six(&self.packed, QUANT_WEIGHTS)?;
        for (index, &weight) in original.iter().enumerate() {
            let clipped = weight.clamp(-clipping, clipping);
            codes[start + index] = encode_element(clipped / scale, format, options.rounding)?;
        }
        self.packed = pack_six(&codes)?;
        self.scales[block] = scale_code;
        self.formats[block] = format;
        self.to_model()?;
        Ok(())
    }

    pub fn block_format(&self, block: usize) -> Fp6Format {
        self.formats[block]
    }

    pub fn block_scale(&self, block: usize) -> u8 {
        self.scales[block]
    }

    pub fn block_codes(&self, block: usize) -> Result<Vec<u8>> {
        if block >= BLOCKS {
            return Err("invalid FP6 block".into());
        }
        let codes = unpack_six(&self.packed, QUANT_WEIGHTS)?;
        Ok(codes[block * BLOCK_SIZE..((block + 1) * BLOCK_SIZE).min(QUANT_WEIGHTS)].to_vec())
    }

    pub fn is_uniform(&self) -> bool {
        self.formats.iter().all(|format| *format == self.formats[0])
    }

    pub fn format_name(&self) -> &'static str {
        if self.is_uniform() {
            match self.formats[0] {
                Fp6Format::E3M2 => "QSF1-MXFP6-E3M2",
                Fp6Format::E2M3 => "QSF1-MXFP6-E2M3",
            }
        } else {
            "QSF1-MXFP6-MIXED"
        }
    }

    pub fn to_model(&self) -> Result<Model> {
        let codes = unpack_six(&self.packed, QUANT_WEIGHTS)?;
        let mut weights = [0.0_f32; PARAMS];
        for (index, &code) in codes.iter().enumerate() {
            let block = index / BLOCK_SIZE;
            let value =
                decode_element(code, self.formats[block]) * decode_scale(self.scales[block])?;
            if !value.is_finite() {
                return Err("FP6 value and scale overflow FP32".into());
            }
            let destination = if index < B1 { index } else { W2 + index - B1 };
            weights[destination] = value;
        }
        for (index, &bits) in self.biases.iter().enumerate() {
            let value = f16::from_bits(bits).to_f32();
            if !value.is_finite() {
                return Err("non-finite FP16 bias".into());
            }
            let destination = if index < W2 - B1 {
                B1 + index
            } else {
                B2 + index - (W2 - B1)
            };
            weights[destination] = value;
        }
        Ok(Model { weights })
    }

    /// Unpacking and E8M0 scale application occur inside each timed inference.
    pub fn enhance(&self, base: &RgbImage) -> RgbImage {
        self.to_model().expect("validated FP6 model").enhance(base)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut bytes = Vec::with_capacity(FILE_BYTES);
        bytes.extend_from_slice(b"QSF1");
        bytes.push(1); // version
        bytes.push(BLOCK_SIZE as u8);
        bytes.extend_from_slice(&(QUANT_WEIGHTS as u16).to_le_bytes());
        bytes.extend_from_slice(&(BIASES as u16).to_le_bytes());
        bytes.extend_from_slice(&(BLOCKS as u16).to_le_bytes());
        bytes.extend_from_slice(&(PACKED_BYTES as u16).to_le_bytes());
        bytes.push(TAG_BYTES as u8);
        bytes.push(0); // reserved
        bytes.extend_from_slice(&self.scales);
        let mut tags = [0_u8; TAG_BYTES];
        for (index, &format) in self.formats.iter().enumerate() {
            if format == Fp6Format::E2M3 {
                tags[index / 8] |= 1 << (index % 8);
            }
        }
        bytes.extend_from_slice(&tags);
        bytes.extend_from_slice(&self.packed);
        for &bias in &self.biases {
            bytes.extend_from_slice(&bias.to_le_bytes());
        }
        debug_assert_eq!(bytes.len(), FILE_BYTES);
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() != FILE_BYTES
            || &bytes[..4] != b"QSF1"
            || bytes[4] != 1
            || bytes[5] as usize != BLOCK_SIZE
            || u16::from_le_bytes(bytes[6..8].try_into()?) as usize != QUANT_WEIGHTS
            || u16::from_le_bytes(bytes[8..10].try_into()?) as usize != BIASES
            || u16::from_le_bytes(bytes[10..12].try_into()?) as usize != BLOCKS
            || u16::from_le_bytes(bytes[12..14].try_into()?) as usize != PACKED_BYTES
            || bytes[14] as usize != TAG_BYTES
            || bytes[15] != 0
        {
            return Err("invalid QSF1 header or file length".into());
        }
        let mut scales = [0_u8; BLOCKS];
        scales.copy_from_slice(&bytes[HEADER_BYTES..HEADER_BYTES + BLOCKS]);
        if scales.contains(&255) {
            return Err("QSF1 contains reserved E8M0 scale".into());
        }
        let tag_start = HEADER_BYTES + BLOCKS;
        let tags = &bytes[tag_start..tag_start + TAG_BYTES];
        if tags[TAG_BYTES - 1] >> (BLOCKS % 8) != 0 {
            return Err("nonzero QSF1 format-tag padding".into());
        }
        let formats = std::array::from_fn(|index| {
            if tags[index / 8] & (1 << (index % 8)) == 0 {
                Fp6Format::E3M2
            } else {
                Fp6Format::E2M3
            }
        });
        let packed_start = tag_start + TAG_BYTES;
        let packed = bytes[packed_start..packed_start + PACKED_BYTES].to_vec();
        unpack_six(&packed, QUANT_WEIGHTS)?;
        let mut biases = [0_u16; BIASES];
        for (index, bias) in biases.iter_mut().enumerate() {
            let start = packed_start + PACKED_BYTES + index * 2;
            *bias = u16::from_le_bytes(bytes[start..start + 2].try_into()?);
        }
        let model = Self {
            formats,
            scales,
            packed,
            biases,
        };
        model.to_model()?;
        Ok(model)
    }

    pub fn file_bytes() -> usize {
        FILE_BYTES
    }

    pub fn weight_stream_bytes() -> usize {
        BLOCKS + TAG_BYTES + PACKED_BYTES
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn all_64_patterns_decode_to_ocp_values() {
        let e3m2 = [
            0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5, 0.625, 0.75, 0.875, 1.0,
            1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0,
            20.0, 24.0, 28.0,
        ];
        let e2m3 = [
            0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625,
            1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
            7.0, 7.5,
        ];
        for (format, expected) in [(Fp6Format::E3M2, e3m2), (Fp6Format::E2M3, e2m3)] {
            for (code, &value) in expected.iter().enumerate() {
                assert_eq!(decode_element(code as u8, format), value);
                assert_eq!(decode_element((code + 32) as u8, format), -value);
            }
        }
        assert!(decode_element(32, Fp6Format::E3M2).is_sign_negative());
    }

    #[test]
    fn rounding_saturation_subnormals_and_sign_are_explicit() {
        assert_eq!(
            encode_element(11.0, Fp6Format::E3M2, Rounding::NearestEven).unwrap(),
            26
        );
        assert_eq!(
            encode_element(0.1, Fp6Format::E3M2, Rounding::NearestEven).unwrap(),
            2
        );
        assert_eq!(
            encode_element(0.03125, Fp6Format::E3M2, Rounding::NearestEven).unwrap(),
            0
        );
        assert_eq!(
            encode_element(0.09375, Fp6Format::E3M2, Rounding::NearestEven).unwrap(),
            2
        );
        assert_eq!(
            encode_element(0.1875, Fp6Format::E2M3, Rounding::NearestEven).unwrap(),
            2
        );
        assert_eq!(
            encode_element(f32::INFINITY, Fp6Format::E3M2, Rounding::NearestEven).unwrap(),
            31
        );
        assert_eq!(
            encode_element(f32::NEG_INFINITY, Fp6Format::E2M3, Rounding::NearestEven).unwrap(),
            63
        );
        assert_eq!(
            encode_element(-0.0, Fp6Format::E3M2, Rounding::NearestEven).unwrap(),
            32
        );
        assert!(encode_element(f32::NAN, Fp6Format::E3M2, Rounding::NearestEven).is_err());
        assert_eq!(decode_scale(127).unwrap(), 1.0);
        assert_eq!(decode_scale(126).unwrap(), 0.5);
        assert_eq!(decode_scale(0).unwrap(), 2.0_f32.powi(-127));
        assert!(decode_scale(0).unwrap() > 0.0);
        assert_eq!(decode_scale(254).unwrap(), 2.0_f32.powi(127));
        assert!(decode_scale(255).is_err());
    }

    #[test]
    fn six_bit_packing_handles_partial_final_bytes() {
        for count in 0_usize..12 {
            let values: Vec<_> = (0..count).map(|x| (x * 5 % 64) as u8).collect();
            let packed = pack_six(&values).unwrap();
            assert_eq!(packed.len(), (count * 6).div_ceil(8));
            assert_eq!(unpack_six(&packed, count).unwrap(), values);
        }
        assert_eq!(pack_six(&[1, 2, 3, 4]).unwrap(), vec![0x81, 0x30, 0x10]);
        assert!(pack_six(&[64]).is_err());
        assert!(unpack_six(&[0xc1], 1).is_err());
    }

    #[test]
    fn model_file_roundtrip_preserves_format_scale_codes_and_inference() {
        let source = Model::new(42);
        let mut encoded = Fp6Model::uniform(&source, Fp6Format::E3M2).unwrap();
        encoded
            .set_block_from_source(&source, 2, Fp6Format::E2M3, BlockOptions::default())
            .unwrap();
        let path =
            std::env::temp_dir().join(format!("quant-search-fp6-{}.qsf", std::process::id()));
        encoded.save(&path).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), FILE_BYTES as u64);
        let restored = Fp6Model::load(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(restored.format_name(), "QSF1-MXFP6-MIXED");
        for block in 0..BLOCKS {
            assert_eq!(restored.block_format(block), encoded.block_format(block));
            assert_eq!(restored.block_scale(block), encoded.block_scale(block));
            assert_eq!(
                restored.block_codes(block).unwrap(),
                encoded.block_codes(block).unwrap()
            );
        }
        let image = RgbImage::from_pixel(8, 8, Rgb([100, 120, 140]));
        assert_eq!(restored.enhance(&image), encoded.enhance(&image));
    }

    #[test]
    fn model_file_rejects_reserved_and_corrupt_metadata() {
        let model = Fp6Model::uniform(&Model::new(17), Fp6Format::E2M3).unwrap();
        let path = std::env::temp_dir().join(format!(
            "quant-search-fp6-invalid-{}.qsf",
            std::process::id()
        ));
        model.save(&path).unwrap();
        let original = fs::read(&path).unwrap();
        for (offset, value) in [
            (4, 2_u8),
            (HEADER_BYTES, 255_u8),
            (HEADER_BYTES + BLOCKS + TAG_BYTES - 1, 0x80_u8),
        ] {
            let mut bytes = original.clone();
            bytes[offset] = value;
            fs::write(&path, bytes).unwrap();
            assert!(Fp6Model::load(&path).is_err(), "offset {offset}");
        }
        let mut bytes = original;
        let bias_start = HEADER_BYTES + BLOCKS + TAG_BYTES + PACKED_BYTES;
        bytes[bias_start..bias_start + 2].copy_from_slice(&f16::NAN.to_bits().to_le_bytes());
        fs::write(&path, bytes).unwrap();
        assert!(Fp6Model::load(&path).is_err());
        fs::remove_file(path).unwrap();
    }
}
