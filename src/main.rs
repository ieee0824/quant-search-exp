use quant_search_exp::{Model, Result, bicubic_2x, evaluate, load_rgb, train, upscale};
use std::path::Path;
use std::time::{Duration, Instant};

fn usage() -> &'static str {
    "Usage:\n  quant-search-exp train <high_res_dir> <model.qsr> [epochs=8] [samples_per_epoch=20000] [jpeg_quality=70] [seed=42] [learning_rate=0.001]\n  quant-search-exp eval <model.qsr> <held_out_dir> [jpeg_quality=70]\n  quant-search-exp upscale <model.qsr> <input.jpg> <output.jpg>\n  quant-search-exp bench <model.qsr> <input.jpg> [runs=5]"
}

fn optional<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> Result<T>
where
    T::Err: std::error::Error + 'static,
{
    match args.get(index) {
        Some(value) => Ok(value.parse::<T>()?),
        None => Ok(default),
    }
}

fn bench(model: &Model, input: &Path, runs: usize) -> Result<()> {
    if runs == 0 {
        return Err("runs must be positive".into());
    }
    let base = bicubic_2x(&load_rgb(input)?)?;
    let scalar = model.enhance_scalar(&base);
    let simd = model.enhance(&base);
    let max_delta = scalar
        .pixels()
        .zip(simd.pixels())
        .flat_map(|(a, b)| a.0.into_iter().zip(b.0).map(|(x, y)| x.abs_diff(y)))
        .max()
        .unwrap_or(0);
    let mut scalar_time = Duration::ZERO;
    let mut simd_time = Duration::ZERO;
    for _ in 0..runs {
        let start = Instant::now();
        std::hint::black_box(model.enhance_scalar(&base));
        scalar_time += start.elapsed();
        let start = Instant::now();
        std::hint::black_box(model.enhance(&base));
        simd_time += start.elapsed();
    }
    let scalar_ms = scalar_time.as_secs_f64() * 1_000.0 / runs as f64;
    let simd_ms = simd_time.as_secs_f64() * 1_000.0 / runs as f64;
    println!(
        "{}x{} pixels; NEON: {}; maximum RGB difference: {}",
        base.width(),
        base.height(),
        Model::simd_enabled(),
        max_delta
    );
    println!(
        "scalar: {scalar_ms:.3} ms; selected path: {simd_ms:.3} ms; ratio: {:.2}x",
        scalar_ms / simd_ms
    );
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("train") if (4..=9).contains(&args.len()) => {
            let epochs = optional(&args, 4, 8)?;
            let samples = optional(&args, 5, 20_000)?;
            let quality = optional(&args, 6, 70_u8)?;
            let seed = optional(&args, 7, 42_u64)?;
            let learning_rate = optional(&args, 8, 0.001_f32)?;
            train(
                Path::new(&args[2]),
                Path::new(&args[3]),
                epochs,
                samples,
                quality,
                seed,
                learning_rate,
            )
        }
        Some("eval") if (4..=5).contains(&args.len()) => {
            let model = Model::load_fp16(Path::new(&args[2]))?;
            let quality = optional(&args, 4, 70_u8)?;
            let metrics = evaluate(&model, Path::new(&args[3]), quality)?;
            println!(
                "{} images; bicubic {:.3} dB; model {:.3} dB; change {:+.3} dB",
                metrics.images,
                metrics.baseline_psnr,
                metrics.model_psnr,
                metrics.model_psnr - metrics.baseline_psnr
            );
            Ok(())
        }
        Some("upscale") if args.len() == 5 => {
            let model = Model::load_fp16(Path::new(&args[2]))?;
            upscale(&model, Path::new(&args[3]), Path::new(&args[4]))?;
            println!("saved {}", args[4]);
            Ok(())
        }
        Some("bench") if (4..=5).contains(&args.len()) => {
            let model = Model::load_fp16(Path::new(&args[2]))?;
            bench(&model, Path::new(&args[3]), optional(&args, 4, 5)?)
        }
        _ => Err(usage().into()),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
