//! Attribute the throughput gap between dynamic attention matmuls and baked linears.
//!
//! Each candidate is checked against an independent CPU matrix multiplication,
//! then timed back-to-back with the same dynamic baseline in alternating order.
//! Repeating the baseline makes the ratios useful even when other work changes
//! host load during the run.
//! Timings use the cached wall-clock path and therefore include the roughly 92 us
//! dispatch cost that hardware statistics do not expose on this machine.

use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use ane::{Executable, Graph, NSQualityOfService, Shape, TensorData};

const HEADS: usize = 12;
const SEQ: usize = 512;
const HEAD_DIM: usize = 64;
const REPEATS: usize = 25;
const MAX_ABS_ERROR: f32 = 0.003;

#[derive(Clone, Copy)]
struct MatmulShape {
    channels: usize,
    rows: usize,
    inner: usize,
    columns: usize,
}

impl MatmulShape {
    fn lhs(self) -> Shape {
        Shape {
            batch: 1,
            channels: self.channels,
            height: self.rows,
            width: self.inner,
        }
    }

    fn rhs(self) -> Shape {
        Shape {
            batch: 1,
            channels: self.channels,
            height: self.columns,
            width: self.inner,
        }
    }

    fn output(self) -> Shape {
        Shape {
            batch: 1,
            channels: self.channels,
            height: self.rows,
            width: self.columns,
        }
    }

    fn gflop(self) -> f64 {
        2.0 * (self.channels * self.rows * self.inner * self.columns) as f64 / 1e9
    }
}

struct Arm {
    label: String,
    gflop: f64,
    executable: Executable,
    inputs: Vec<TensorData>,
    output: TensorData,
    expected: Arc<Vec<f32>>,
}

struct Measurement {
    median_ms: f64,
    load_one: f64,
}

fn pattern(index: usize, seed: usize) -> f32 {
    let mixed = index
        .wrapping_mul(2_654_435_761)
        .wrapping_add(seed * 104_729);
    ((mixed % 257) as i32 - 128) as f32 / 4096.0
}

fn matrix_values(shape: MatmulShape, seed: usize, shared_channels: bool, rhs: bool) -> Vec<f32> {
    let rows = if rhs { shape.columns } else { shape.rows };
    let mut values = vec![0.0; shape.channels * rows * shape.inner];
    for channel in 0..shape.channels {
        for row in 0..rows {
            for column in 0..shape.inner {
                let local = row * shape.inner + column;
                let source_channel = if shared_channels { 0 } else { channel };
                values[(channel * rows + row) * shape.inner + column] =
                    pattern(source_channel * rows * shape.inner + local, seed);
            }
        }
    }
    values
}

fn cpu_matmul(shape: MatmulShape, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; shape.channels * shape.rows * shape.columns];
    for channel in 0..shape.channels {
        for row in 0..shape.rows {
            for column in 0..shape.columns {
                let mut sum = 0.0_f32;
                for inner in 0..shape.inner {
                    let lhs_index = (channel * shape.rows + row) * shape.inner + inner;
                    let rhs_index = (channel * shape.columns + column) * shape.inner + inner;
                    sum += lhs[lhs_index] * rhs[rhs_index];
                }
                output[(channel * shape.rows + row) * shape.columns + column] = sum;
            }
        }
    }
    output
}

fn compile_dynamic(
    label: impl Into<String>,
    shape: MatmulShape,
    lhs: Vec<f32>,
    rhs: Vec<f32>,
    expected: Arc<Vec<f32>>,
) -> Result<Arm, String> {
    let mut graph = Graph::new();
    let lhs_tensor = graph.placeholder(shape.lhs());
    let rhs_tensor = graph.placeholder(shape.rhs());
    let _ = graph.matrix_multiplication(lhs_tensor, rhs_tensor, false, true);
    let executable = graph
        .compile(NSQualityOfService::UserInteractive)
        .map_err(|error| format!("compile failed: {error:?}"))?;
    Ok(Arm {
        label: label.into(),
        gflop: shape.gflop(),
        executable,
        inputs: vec![
            TensorData::with_f32(&lhs, shape.lhs()),
            TensorData::with_f32(&rhs, shape.rhs()),
        ],
        output: TensorData::new(shape.output()),
        expected,
    })
}

fn model_layout(ready: &[f32], channels: usize, rows: usize, inner: usize) -> Vec<f32> {
    let mut model = vec![0.0; ready.len()];
    for channel in 0..channels {
        for row in 0..rows {
            for column in 0..inner {
                let ready_index = (channel * rows + row) * inner + column;
                let model_index = (channel * inner + column) * rows + row;
                model[model_index] = ready[ready_index];
            }
        }
    }
    model
}

fn compile_with_transposes(
    shape: MatmulShape,
    lhs: &[f32],
    rhs: &[f32],
    expected: Arc<Vec<f32>>,
) -> Result<Arm, String> {
    let source_shape = Shape {
        batch: 1,
        channels: shape.channels * shape.inner,
        height: 1,
        width: shape.rows,
    };
    let heads_shape = Shape {
        batch: 1,
        channels: shape.channels,
        height: shape.inner,
        width: shape.rows,
    };
    let mut graph = Graph::new();
    let lhs_tensor = graph.placeholder(source_shape);
    let rhs_tensor = graph.placeholder(source_shape);
    let lhs_tensor = graph.reshape(lhs_tensor, heads_shape);
    let rhs_tensor = graph.reshape(rhs_tensor, heads_shape);
    let permutation = [0, 1, 3, 2];
    let lhs_tensor = graph.transpose(lhs_tensor, permutation);
    let rhs_tensor = graph.transpose(rhs_tensor, permutation);
    let _ = graph.matrix_multiplication(lhs_tensor, rhs_tensor, false, true);
    let executable = graph
        .compile(NSQualityOfService::UserInteractive)
        .map_err(|error| format!("compile failed: {error:?}"))?;
    Ok(Arm {
        label: "layout reshape+transpose".into(),
        gflop: shape.gflop(),
        executable,
        inputs: vec![
            TensorData::with_f32(
                &model_layout(lhs, shape.channels, shape.rows, shape.inner),
                source_shape,
            ),
            TensorData::with_f32(
                &model_layout(rhs, shape.channels, shape.rows, shape.inner),
                source_shape,
            ),
        ],
        output: TensorData::new(shape.output()),
        expected,
    })
}

fn compile_constant_rhs(
    shape: MatmulShape,
    lhs: &[f32],
    rhs: &[f32],
    expected: Arc<Vec<f32>>,
) -> Result<Arm, String> {
    let mut graph = Graph::new();
    let lhs_tensor = graph.placeholder(shape.lhs());
    let rhs_tensor = graph.constant(rhs, shape.rhs());
    let _ = graph.matrix_multiplication(lhs_tensor, rhs_tensor, false, true);
    let executable = graph
        .compile(NSQualityOfService::UserInteractive)
        .map_err(|error| format!("compile failed: {error:?}"))?;
    Ok(Arm {
        label: "constant RHS matmul".into(),
        gflop: shape.gflop(),
        executable,
        inputs: vec![TensorData::with_f32(lhs, shape.lhs())],
        output: TensorData::new(shape.output()),
        expected,
    })
}

fn compile_inner_product(
    shape: MatmulShape,
    lhs: &[f32],
    rhs: &[f32],
    expected_matmul: &[f32],
) -> Result<Arm, String> {
    let positions = shape.channels * shape.rows;
    let input_shape = Shape {
        batch: 1,
        channels: shape.inner,
        height: 1,
        width: positions,
    };
    let output_shape = Shape {
        batch: 1,
        channels: shape.columns,
        height: 1,
        width: positions,
    };

    let mut packed_lhs = vec![0.0; lhs.len()];
    for channel in 0..shape.channels {
        for row in 0..shape.rows {
            let position = channel * shape.rows + row;
            for inner in 0..shape.inner {
                packed_lhs[inner * positions + position] =
                    lhs[(channel * shape.rows + row) * shape.inner + inner];
            }
        }
    }
    let constant_weights = rhs[..shape.columns * shape.inner].to_vec();
    let mut expected = vec![0.0; expected_matmul.len()];
    for channel in 0..shape.channels {
        for row in 0..shape.rows {
            let position = channel * shape.rows + row;
            for column in 0..shape.columns {
                expected[column * positions + position] =
                    expected_matmul[(channel * shape.rows + row) * shape.columns + column];
            }
        }
    }

    let mut graph = Graph::new();
    let input = graph.placeholder(input_shape);
    let _ = graph.inner_product(input, &constant_weights, shape.inner, shape.columns);
    let executable = graph
        .compile(NSQualityOfService::UserInteractive)
        .map_err(|error| format!("compile failed: {error:?}"))?;
    Ok(Arm {
        label: "constant RHS inner_product".into(),
        gflop: shape.gflop(),
        executable,
        inputs: vec![TensorData::with_f32(&packed_lhs, input_shape)],
        output: TensorData::new(output_shape),
        expected: Arc::new(expected),
    })
}

fn compile_grouped(
    groups: usize,
    shape: MatmulShape,
    lhs: &[f32],
    rhs: &[f32],
    expected: Arc<Vec<f32>>,
) -> Result<Arm, String> {
    let channels_per_group = shape.channels / groups;
    let mut graph = Graph::new();
    let lhs_tensor = graph.placeholder(shape.lhs());
    let rhs_tensor = graph.placeholder(shape.rhs());
    let mut outputs = Vec::with_capacity(groups);
    for group in 0..groups {
        let begin = [0, group * channels_per_group, 0, 0];
        let lhs_slice = graph.slice(
            lhs_tensor,
            begin,
            [1, channels_per_group, shape.rows, shape.inner],
        );
        let rhs_slice = graph.slice(
            rhs_tensor,
            begin,
            [1, channels_per_group, shape.columns, shape.inner],
        );
        outputs.push(graph.matrix_multiplication(lhs_slice, rhs_slice, false, true));
    }
    let _ = graph.concat(&outputs, 1);
    let executable = graph
        .compile(NSQualityOfService::UserInteractive)
        .map_err(|error| format!("compile failed: {error:?}"))?;
    Ok(Arm {
        label: format!("grouped matmul x{groups}"),
        gflop: shape.gflop(),
        executable,
        inputs: vec![
            TensorData::with_f32(lhs, shape.lhs()),
            TensorData::with_f32(rhs, shape.rhs()),
        ],
        output: TensorData::new(shape.output()),
        expected,
    })
}

fn borrowed_inputs(arm: &Arm) -> Vec<&TensorData> {
    arm.inputs.iter().collect()
}

fn verify(arm: &Arm) -> Result<f32, String> {
    let inputs = borrowed_inputs(arm);
    arm.executable
        .run_cached(&inputs, &[&arm.output])
        .map_err(|error| format!("execute failed: {error:?}"))?;
    let produced = arm.output.read_f32();
    if produced.len() != arm.expected.len() {
        return Err(format!(
            "wrong output length: expected {}, got {}",
            arm.expected.len(),
            produced.len()
        ));
    }
    let reference_peak = arm
        .expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    if reference_peak <= 2.0 * MAX_ABS_ERROR {
        return Err(format!(
            "CPU reference peak {reference_peak:e} is too small to reject an all-zero output"
        ));
    }
    let worst = produced
        .iter()
        .zip(arm.expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    if !worst.is_finite() || worst > MAX_ABS_ERROR {
        return Err(format!(
            "CPU reference mismatch: max abs error {worst:e} exceeds {MAX_ABS_ERROR:e}"
        ));
    }
    Ok(worst)
}

fn one_minute_load() -> f64 {
    let Ok(output) = Command::new("sysctl").args(["-n", "vm.loadavg"]).output() else {
        return f64::NAN;
    };
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .find_map(|word| word.trim_matches(|c| c == '{' || c == '}').parse().ok())
        .unwrap_or(f64::NAN)
}

fn timed_run(arm: &Arm) -> Result<u64, String> {
    let inputs = borrowed_inputs(arm);
    let started = Instant::now();
    arm.executable
        .run_cached(&inputs, &[&arm.output])
        .map_err(|error| format!("execution failed mid-run: {error:?}"))?;
    Ok(started.elapsed().as_nanos() as u64)
}

fn measure_pair(baseline: &Arm, candidate: &Arm) -> Result<(Measurement, Measurement), String> {
    let baseline_load = one_minute_load();
    let candidate_load = one_minute_load();
    let mut baseline_samples = Vec::with_capacity(REPEATS);
    let mut candidate_samples = Vec::with_capacity(REPEATS);
    for repeat in 0..REPEATS {
        // Alternate order so neither arm benefits systematically from running
        // immediately after the other while every sample remains back-to-back.
        if repeat % 2 == 0 {
            baseline_samples.push(timed_run(baseline)?);
            candidate_samples.push(timed_run(candidate)?);
        } else {
            candidate_samples.push(timed_run(candidate)?);
            baseline_samples.push(timed_run(baseline)?);
        }
    }
    baseline_samples.sort_unstable();
    candidate_samples.sort_unstable();
    Ok((
        Measurement {
            median_ms: baseline_samples[baseline_samples.len() / 2] as f64 / 1_000_000.0,
            load_one: baseline_load,
        },
        Measurement {
            median_ms: candidate_samples[candidate_samples.len() / 2] as f64 / 1_000_000.0,
            load_one: candidate_load,
        },
    ))
}

fn print_measurement(label: &str, gflop: f64, measurement: &Measurement) {
    println!(
        "{label:<34} {:>10.3}  {:>10.1}  {:>8.2}",
        measurement.median_ms,
        gflop / (measurement.median_ms / 1000.0),
        measurement.load_one
    );
}

fn run_candidate(baseline: &Arm, candidate: Result<Arm, String>, failures: &mut Vec<String>) {
    let candidate = match candidate {
        Ok(candidate) => candidate,
        Err(reason) => {
            println!("FAILED to build candidate: {reason}");
            failures.push(reason);
            return;
        }
    };
    match verify(&candidate) {
        Ok(worst) => println!(
            "VERIFY PASS {:<32} max abs error {worst:e}",
            candidate.label
        ),
        Err(reason) => {
            println!("VERIFY FAIL {:<32} {reason}", candidate.label);
            failures.push(format!("{}: {reason}", candidate.label));
            return;
        }
    }

    let (baseline_measurement, candidate_measurement) = match measure_pair(baseline, &candidate) {
        Ok(measurements) => measurements,
        Err(reason) => {
            println!("FAILED pair for {}: {reason}", candidate.label);
            failures.push(format!("pair for {}: {reason}", candidate.label));
            return;
        }
    };
    print_measurement(
        &format!("paired baseline for {}", candidate.label),
        baseline.gflop,
        &baseline_measurement,
    );
    print_measurement(&candidate.label, candidate.gflop, &candidate_measurement);
    println!(
        "  ratio candidate/baseline: {:.3}x\n",
        candidate_measurement.median_ms / baseline_measurement.median_ms
    );
}

fn main() {
    println!("Dynamic attention matmul attribution on one process/run");
    println!("Wall-clock medians include roughly 92 us dispatch; load1 is recorded per row.");
    println!("Every arm is checked against an independent CPU reference before timing.\n");
    println!(
        "{:<34} {:>10}  {:>10}  {:>8}",
        "arm", "median ms", "GFLOP/s", "load1"
    );

    let canonical = MatmulShape {
        channels: HEADS,
        rows: SEQ,
        inner: HEAD_DIM,
        columns: SEQ,
    };
    let lhs = matrix_values(canonical, 11, false, false);
    // Sharing the same key matrix across channels permits an exactly equivalent
    // inner_product arm without changing the number of useful multiply-adds.
    let rhs = matrix_values(canonical, 29, true, true);
    let expected = Arc::new(cpu_matmul(canonical, &lhs, &rhs));
    let baseline = compile_dynamic(
        "dynamic ready-layout baseline",
        canonical,
        lhs.clone(),
        rhs.clone(),
        Arc::clone(&expected),
    )
    .unwrap_or_else(|reason| {
        eprintln!("BASELINE BUILD FAILED: {reason}");
        std::process::exit(2);
    });
    match verify(&baseline) {
        Ok(worst) => println!(
            "VERIFY PASS {:<32} max abs error {worst:e}\n",
            baseline.label
        ),
        Err(reason) => {
            eprintln!("BASELINE VERIFY FAILED: {reason}");
            std::process::exit(3);
        }
    }

    let mut failures = Vec::new();

    println!("\n== operand layout ==");
    run_candidate(
        &baseline,
        compile_with_transposes(canonical, &lhs, &rhs, Arc::clone(&expected)),
        &mut failures,
    );

    println!("\n== constant versus runtime operand ==");
    run_candidate(
        &baseline,
        compile_constant_rhs(canonical, &lhs, &rhs, Arc::clone(&expected)),
        &mut failures,
    );
    run_candidate(
        &baseline,
        compile_inner_product(canonical, &lhs, &rhs, &expected),
        &mut failures,
    );

    println!("\n== equal-FLOP shape sweep ==");
    for shape in [
        MatmulShape {
            channels: 6,
            rows: 512,
            inner: 128,
            columns: 512,
        },
        MatmulShape {
            channels: 3,
            rows: 512,
            inner: 256,
            columns: 512,
        },
        MatmulShape {
            channels: 1,
            rows: 512,
            inner: 768,
            columns: 512,
        },
        MatmulShape {
            channels: 12,
            rows: 256,
            inner: 256,
            columns: 256,
        },
        MatmulShape {
            channels: 12,
            rows: 128,
            inner: 1024,
            columns: 128,
        },
    ] {
        let shape_lhs = matrix_values(shape, 11, false, false);
        let shape_rhs = matrix_values(shape, 29, false, true);
        let shape_expected = Arc::new(cpu_matmul(shape, &shape_lhs, &shape_rhs));
        let label = format!(
            "shape c{} {}x{}x{}",
            shape.channels, shape.rows, shape.inner, shape.columns
        );
        run_candidate(
            &baseline,
            compile_dynamic(label, shape, shape_lhs, shape_rhs, shape_expected),
            &mut failures,
        );
    }

    println!("\n== split versus fused heads ==");
    for groups in [2usize, 3, 4, 6, 12] {
        run_candidate(
            &baseline,
            compile_grouped(groups, canonical, &lhs, &rhs, Arc::clone(&expected)),
            &mut failures,
        );
    }

    if failures.is_empty() {
        println!("All arms compiled, executed, and matched their CPU references.");
    } else {
        eprintln!("{} arm(s) failed:", failures.len());
        for failure in failures {
            eprintln!("- {failure}");
        }
        std::process::exit(4);
    }
}
