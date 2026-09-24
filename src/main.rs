use quant_search_exp::evaluation::{ReportOptions, evaluate_report, verify_manifest_data};
use quant_search_exp::experiments::{search_uniform_fp6, select_mixed_fp6};
use quant_search_exp::fp6::{Fp6Format, Fp6Model};
use quant_search_exp::{
    InferenceModel, Model, Result, bicubic_2x, evaluate, load_rgb, train, upscale,
};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn usage() -> &'static str {
    "Usage:\n  quant-search-exp train <high_res_dir> <model.qsr> [epochs=8] [samples_per_epoch=20000] [jpeg_quality=70] [seed=42] [learning_rate=0.001]\n  quant-search-exp eval <model.qsr> <held_out_dir> [jpeg_quality=70]\n  quant-search-exp quantize-int8 <model.qsr> <model.qi8>\n  quant-search-exp quantize-fp6 <model.qsr> <model.qsf> <e3m2|e2m3>\n  quant-search-exp select-mixed-fp6 <model.qsr> <manifest.json> <data_root> <output.qsf> <record.json> [quality=70]\n  quant-search-exp search-fp6 <model.qsr> <manifest.json> <data_root> <output.qsf> <record.json> <e3m2|e2m3> [quality=70]\n  quant-search-exp verify-data <manifest.json> <data_root>\n  quant-search-exp eval-report <manifest.json> <data_root> <val|test|final> <output.json> <model_file> [more_model_files...] [--quality N] [--runs N] [--exclude-manifest path]\n  quant-search-exp upscale <model_file> <input.jpg> <output.jpg>\n  quant-search-exp bench <model.qsr> <input.jpg> [runs=5]"
}

fn parse_fp6_format(value: &str) -> Result<Fp6Format> {
    match value.to_ascii_lowercase().as_str() {
        "e3m2" => Ok(Fp6Format::E3M2),
        "e2m3" => Ok(Fp6Format::E2M3),
        _ => Err("FP6 format must be e3m2 or e2m3".into()),
    }
}

fn report_command(args: &[String]) -> Result<()> {
    if args.len() < 7 {
        return Err(usage().into());
    }
    let mut models = Vec::<PathBuf>::new();
    let mut quality = 70_u8;
    let mut runs = 5_usize;
    let mut exclusion = None;
    let mut index = 6;
    while index < args.len() {
        match args[index].as_str() {
            "--quality" | "--runs" | "--exclude-manifest" => {
                let value = args.get(index + 1).ok_or("option requires a value")?;
                match args[index].as_str() {
                    "--quality" => quality = value.parse()?,
                    "--runs" => runs = value.parse()?,
                    _ => exclusion = Some(PathBuf::from(value)),
                }
                index += 2;
            }
            value if value.starts_with("--") => {
                return Err(format!("unknown option: {value}").into());
            }
            value => {
                models.push(PathBuf::from(value));
                index += 1;
            }
        }
    }
    evaluate_report(
        Path::new(&args[2]),
        Path::new(&args[3]),
        &args[4],
        Path::new(&args[5]),
        &models,
        ReportOptions {
            quality,
            runs,
            exclusion_manifest: exclusion.as_deref(),
        },
    )?;
    println!("saved {}", args[5]);
    Ok(())
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
        Some("quantize-int8") if args.len() == 4 => {
            if args[2] == args[3] {
                return Err("input and output model paths must differ".into());
            }
            Model::load_fp16(Path::new(&args[2]))?.save_int8(Path::new(&args[3]))?;
            println!("saved {}", args[3]);
            Ok(())
        }
        Some("quantize-fp6") if args.len() == 5 => {
            if args[2] == args[3] {
                return Err("input and output model paths must differ".into());
            }
            let format = parse_fp6_format(&args[4])?;
            let source = Model::load_fp16(Path::new(&args[2]))?;
            Fp6Model::uniform(&source, format)?.save(Path::new(&args[3]))?;
            println!("saved {}", args[3]);
            Ok(())
        }
        Some("select-mixed-fp6") if (7..=8).contains(&args.len()) => {
            select_mixed_fp6(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
                Path::new(&args[6]),
                optional(&args, 7, 70_u8)?,
            )?;
            println!("saved {} and {}", args[5], args[6]);
            Ok(())
        }
        Some("search-fp6") if (8..=9).contains(&args.len()) => {
            search_uniform_fp6(
                Path::new(&args[2]),
                Path::new(&args[3]),
                Path::new(&args[4]),
                Path::new(&args[5]),
                Path::new(&args[6]),
                parse_fp6_format(&args[7])?,
                optional(&args, 8, 70_u8)?,
            )?;
            println!("saved {} and {}", args[5], args[6]);
            Ok(())
        }
        Some("verify-data") if args.len() == 4 => {
            let verification = verify_manifest_data(Path::new(&args[2]), Path::new(&args[3]))?;
            println!("{}", serde_json::to_string_pretty(&verification)?);
            Ok(())
        }
        Some("eval-report") => report_command(&args),
        Some("upscale") if args.len() == 5 => {
            let (model, _) = InferenceModel::load(Path::new(&args[2]))?;
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
