//! Attribute ModernBERT's sequence-length scaling inside the direct ANE graph.
//!
//! Comparisons run back-to-back in one process, alternate execution order, and
//! report medians with a fresh one-minute load reading for every row. Equivalent
//! attention, normalization, RoPE, and transfer arms are checked against CPU
//! references before timing. Deliberately incomplete timing bounds say so in
//! their output labels.

use std::time::Instant;

use ane::{Executable, Graph, NSQualityOfService, Shape, Tensor, TensorData};
use anyhow::{ensure, Context, Result};

const HIDDEN: usize = 768;
const INTERMEDIATE: usize = 1152;
const HEADS: usize = 12;
const HEAD_DIM: usize = HIDDEN / HEADS;
const QUERY_TILE: usize = 128;
const LOCAL_RADIUS: usize = 64;
const MASK_MIN: f32 = -10_000.0;
const EPSILON: f32 = 1e-5;
const REPEATS: usize = 15;
const CHAIN_LENGTH: usize = 22;

#[derive(Clone, Copy)]
struct Measurement {
    median_ms: f64,
    load_one: f64,
}

struct Arm {
    label: String,
    executable: Executable,
    inputs: Vec<TensorData>,
    output: TensorData,
    expected: Option<Vec<f32>>,
    tolerance: f32,
    timing_only: bool,
}

struct ChainArm {
    label: String,
    executables: Vec<Executable>,
    input: TensorData,
    first: TensorData,
    second: TensorData,
    expected: Vec<f32>,
}

fn shape(sequence: usize, channels: usize) -> Shape {
    Shape {
        batch: 1,
        channels,
        height: 1,
        width: sequence,
    }
}

fn head_shape(sequence: usize) -> Shape {
    Shape {
        batch: 1,
        channels: HEADS,
        height: sequence,
        width: HEAD_DIM,
    }
}

fn scalar_shape() -> Shape {
    Shape::channels(1)
}

fn pattern(index: usize, seed: usize) -> f32 {
    let mixed = index
        .wrapping_mul(2_654_435_761)
        .wrapping_add(seed.wrapping_mul(104_729));
    ((mixed % 257) as i32 - 128) as f32 / 128.0
}

fn values(count: usize, seed: usize) -> Vec<f32> {
    (0..count).map(|index| pattern(index, seed)).collect()
}

fn weights(count: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|index| pattern(index, seed) * scale / 16.0)
        .collect()
}

fn one_minute_load() -> f64 {
    let mut load = [0.0_f64; 1];
    unsafe extern "C" {
        fn getloadavg(load_average: *mut f64, count: i32) -> i32;
    }
    let count = unsafe { getloadavg(load.as_mut_ptr(), 1) };
    if count == 1 {
        load[0]
    } else {
        f64::NAN
    }
}

fn borrowed_inputs(arm: &Arm) -> Vec<&TensorData> {
    arm.inputs.iter().collect()
}

fn execute(arm: &Arm) -> Result<()> {
    arm.executable
        .run_cached(&borrowed_inputs(arm), &[&arm.output])
        .with_context(|| format!("execute {}", arm.label))
}

fn verify(arm: &Arm) -> Result<f32> {
    ensure!(!arm.timing_only, "timing-only arm has no equivalence claim");
    execute(arm)?;
    let actual = arm.output.read_f32();
    let expected = arm.expected.as_ref().context("missing CPU reference")?;
    ensure!(
        actual.len() == expected.len(),
        "{} output length changed",
        arm.label
    );
    let reference_peak = expected
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    ensure!(
        reference_peak > 2.0 * arm.tolerance,
        "{} CPU reference cannot reject zeros",
        arm.label
    );
    let error = actual
        .iter()
        .zip(expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    ensure!(
        error.is_finite() && error <= arm.tolerance,
        "{} CPU mismatch: {error:e} > {:e}",
        arm.label,
        arm.tolerance
    );
    Ok(error)
}

fn timed_run(arm: &Arm) -> Result<u64> {
    let started = Instant::now();
    execute(arm)?;
    Ok(started.elapsed().as_nanos() as u64)
}

fn measure_pair(left: &Arm, right: &Arm) -> Result<(Measurement, Measurement)> {
    execute(left)?;
    execute(right)?;
    let left_load = one_minute_load();
    let right_load = one_minute_load();
    let mut left_samples = Vec::with_capacity(REPEATS);
    let mut right_samples = Vec::with_capacity(REPEATS);
    for repeat in 0..REPEATS {
        if repeat % 2 == 0 {
            left_samples.push(timed_run(left)?);
            right_samples.push(timed_run(right)?);
        } else {
            right_samples.push(timed_run(right)?);
            left_samples.push(timed_run(left)?);
        }
    }
    left_samples.sort_unstable();
    right_samples.sort_unstable();
    Ok((
        Measurement {
            median_ms: left_samples[left_samples.len() / 2] as f64 / 1_000_000.0,
            load_one: left_load,
        },
        Measurement {
            median_ms: right_samples[right_samples.len() / 2] as f64 / 1_000_000.0,
            load_one: right_load,
        },
    ))
}

fn print_pair(sequence: usize, left: &Arm, left_m: Measurement, right: &Arm, right_m: Measurement) {
    let left_kind = if left.timing_only { " TIMING-ONLY" } else { "" };
    let right_kind = if right.timing_only {
        " TIMING-ONLY"
    } else {
        ""
    };
    println!(
        "ROW seq={sequence} arm=\"{}{}\" median_ms={:.4} load1={:.2}",
        left.label, left_kind, left_m.median_ms, left_m.load_one
    );
    println!(
        "ROW seq={sequence} arm=\"{}{}\" median_ms={:.4} load1={:.2} ratio_to_paired={:.3}",
        right.label,
        right_kind,
        right_m.median_ms,
        right_m.load_one,
        right_m.median_ms / left_m.median_ms
    );
}

fn attention_reference(
    sequence: usize,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    local: bool,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; HEADS * sequence * HEAD_DIM];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    for head in 0..HEADS {
        let head_base = head * sequence * HEAD_DIM;
        for query_position in 0..sequence {
            let key_begin = if local {
                query_position.saturating_sub(LOCAL_RADIUS)
            } else {
                0
            };
            let key_end = if local {
                (query_position + LOCAL_RADIUS + 1).min(sequence)
            } else {
                sequence
            };
            let mut scores = Vec::with_capacity(key_end - key_begin);
            let mut maximum = f32::NEG_INFINITY;
            for key_position in key_begin..key_end {
                let mut score = 0.0_f32;
                for channel in 0..HEAD_DIM {
                    score += query[head_base + query_position * HEAD_DIM + channel]
                        * key[head_base + key_position * HEAD_DIM + channel];
                }
                score *= scale;
                maximum = maximum.max(score);
                scores.push(score);
            }
            let mut denominator = 0.0_f32;
            for score in &mut scores {
                *score = (*score - maximum).exp();
                denominator += *score;
            }
            for (offset, probability) in scores.iter().enumerate() {
                let key_position = key_begin + offset;
                let probability = *probability / denominator;
                for channel in 0..HEAD_DIM {
                    output[head_base + query_position * HEAD_DIM + channel] +=
                        probability * value[head_base + key_position * HEAD_DIM + channel];
                }
            }
        }
    }
    output
}

fn distance_mask(
    query_start: usize,
    query_len: usize,
    key_start: usize,
    key_len: usize,
) -> Vec<f32> {
    let mut mask = vec![0.0; query_len * key_len];
    for query in 0..query_len {
        for key in 0..key_len {
            if (query_start + query).abs_diff(key_start + key) > LOCAL_RADIUS {
                mask[query * key_len + key] = MASK_MIN;
            }
        }
    }
    mask
}

fn build_attention(
    sequence: usize,
    tiled: bool,
    local: bool,
    masks: bool,
    timing_only: bool,
) -> Result<Arm> {
    let tensor_shape = head_shape(sequence);
    let query_values = values(HEADS * sequence * HEAD_DIM, 11);
    let key_values = values(HEADS * sequence * HEAD_DIM, 29);
    let value_values = values(HEADS * sequence * HEAD_DIM, 47);
    let expected = (!timing_only)
        .then(|| attention_reference(sequence, &query_values, &key_values, &value_values, local));
    let mut graph = Graph::new();
    let query = graph.placeholder(tensor_shape);
    let key = graph.placeholder(tensor_shape);
    let value = graph.placeholder(tensor_shape);
    let mask_input = masks.then(|| graph.placeholder(shape(sequence, 1)));
    let scale = graph.constant_with_scalar(1.0 / (HEAD_DIM as f32).sqrt(), scalar_shape());
    let tile = if tiled {
        QUERY_TILE.min(sequence)
    } else {
        sequence
    };
    let mut contexts = Vec::new();
    for query_start in (0..sequence).step_by(tile) {
        let query_end = (query_start + tile).min(sequence);
        let query_len = query_end - query_start;
        let (key_start, key_end) = if local {
            (
                query_start.saturating_sub(LOCAL_RADIUS),
                (query_end + LOCAL_RADIUS).min(sequence),
            )
        } else {
            (0, sequence)
        };
        let key_len = key_end - key_start;
        let query_slice = graph.slice(
            query,
            [0, 0, query_start, 0],
            [1, HEADS, query_len, HEAD_DIM],
        );
        let key_slice = graph.slice(key, [0, 0, key_start, 0], [1, HEADS, key_len, HEAD_DIM]);
        let value_slice = graph.slice(value, [0, 0, key_start, 0], [1, HEADS, key_len, HEAD_DIM]);
        let scores = graph.matrix_multiplication(query_slice, key_slice, false, true);
        let scores = graph.multiplication(scores, scale);
        let mut prepared = scores;
        if let Some(mask) = mask_input {
            let padding = graph.slice(mask, [0, 0, 0, key_start], [1, 1, 1, key_len]);
            prepared = graph.addition(prepared, padding);
            if local {
                let distance = distance_mask(query_start, query_len, key_start, key_len);
                let distance = graph.constant(
                    &distance,
                    Shape {
                        batch: 1,
                        channels: 1,
                        height: query_len,
                        width: key_len,
                    },
                );
                prepared = graph.addition(prepared, distance);
            }
        }
        let probabilities = graph.soft_max(prepared, -1);
        contexts.push(graph.matrix_multiplication(probabilities, value_slice, false, false));
    }
    let _context = if contexts.len() == 1 {
        contexts[0]
    } else {
        graph.concat(&contexts, 2)
    };
    let executable = graph
        .compile(NSQualityOfService::UserInteractive)
        .context("compile attention arm")?;
    let mut inputs = vec![
        TensorData::with_f32(&query_values, tensor_shape),
        TensorData::with_f32(&key_values, tensor_shape),
        TensorData::with_f32(&value_values, tensor_shape),
    ];
    if masks {
        inputs.push(TensorData::with_f32(
            &vec![0.0; sequence],
            shape(sequence, 1),
        ));
    }
    let label = match (local, tiled, masks) {
        (false, true, _) => "global attention tiled q128",
        (false, false, _) => "global attention untiled",
        (true, _, true) => "local attention with padding slices + distance constants",
        (true, _, false) => "local attention masks removed (numerically wrong)",
    };
    Ok(Arm {
        label: label.to_owned(),
        executable,
        inputs,
        output: TensorData::new(tensor_shape),
        expected,
        tolerance: 0.015,
        timing_only,
    })
}

fn layer_norm_graph(graph: &mut Graph, input: Tensor) -> Tensor {
    let mean = graph.reduce_mean(input, 1);
    let centered = graph.subtraction(input, mean);
    let magnitude = graph.absolute(centered);
    let magnitude = graph.reduce_max(magnitude, 1);
    let epsilon_root = graph.constant_with_scalar(EPSILON.sqrt(), scalar_shape());
    let scale = graph.maximum(magnitude, epsilon_root);
    let scaled = graph.division(centered, scale);
    let squared = graph.multiplication(scaled, scaled);
    let variance = graph.reduce_mean(squared, 1);
    let scaled_epsilon = graph.division(epsilon_root, scale);
    let scaled_epsilon = graph.multiplication(scaled_epsilon, scaled_epsilon);
    let variance = graph.addition(variance, scaled_epsilon);
    let negative_half = graph.constant_with_scalar(-0.5, scalar_shape());
    let inverse_stddev = graph.power(variance, negative_half);
    graph.multiplication(scaled, inverse_stddev)
}

fn build_layer_norm(sequence: usize) -> Result<Arm> {
    let tensor_shape = shape(sequence, HIDDEN);
    let input_values = values(HIDDEN * sequence, 71);
    let mut expected = vec![0.0_f32; input_values.len()];
    for position in 0..sequence {
        let mean = (0..HIDDEN)
            .map(|channel| input_values[channel * sequence + position])
            .sum::<f32>()
            / HIDDEN as f32;
        let variance = (0..HIDDEN)
            .map(|channel| {
                let centered = input_values[channel * sequence + position] - mean;
                centered * centered
            })
            .sum::<f32>()
            / HIDDEN as f32;
        let inverse = (variance + EPSILON).sqrt().recip();
        for channel in 0..HIDDEN {
            expected[channel * sequence + position] =
                (input_values[channel * sequence + position] - mean) * inverse;
        }
    }
    let mut graph = Graph::new();
    let input = graph.placeholder(tensor_shape);
    let _ = layer_norm_graph(&mut graph, input);
    Ok(Arm {
        label: "one rescaled LayerNorm".to_owned(),
        executable: graph
            .compile(NSQualityOfService::UserInteractive)
            .context("compile LayerNorm")?,
        inputs: vec![TensorData::with_f32(&input_values, tensor_shape)],
        output: TensorData::new(tensor_shape),
        expected: Some(expected),
        tolerance: 0.02,
        timing_only: false,
    })
}

fn build_reduce_mean(sequence: usize) -> Result<Arm> {
    let tensor_shape = shape(sequence, HIDDEN);
    let input_values = values(HIDDEN * sequence, 71);
    let mut graph = Graph::new();
    let input = graph.placeholder(tensor_shape);
    let _ = graph.reduce_mean(input, 1);
    Ok(Arm {
        label: "plain reduce_mean comparison".to_owned(),
        executable: graph
            .compile(NSQualityOfService::UserInteractive)
            .context("compile reduce_mean")?,
        inputs: vec![TensorData::with_f32(&input_values, tensor_shape)],
        output: TensorData::new(shape(sequence, 1)),
        expected: None,
        tolerance: 0.0,
        timing_only: true,
    })
}

fn rope_tables(sequence: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cosine = vec![0.0; HEAD_DIM * sequence];
    let mut sine = vec![0.0; HEAD_DIM * sequence];
    for channel in 0..HEAD_DIM {
        let frequency_channel = channel % (HEAD_DIM / 2);
        let inverse_frequency =
            1.0 / 160_000.0_f32.powf((2 * frequency_channel) as f32 / HEAD_DIM as f32);
        for position in 0..sequence {
            let angle = position as f32 * inverse_frequency;
            cosine[channel * sequence + position] = angle.cos();
            sine[channel * sequence + position] = angle.sin();
        }
    }
    (cosine, sine)
}

fn apply_rope(
    graph: &mut Graph,
    input: Tensor,
    cosine: &[f32],
    sine: &[f32],
    sequence: usize,
) -> Tensor {
    let half = HEAD_DIM / 2;
    let first = graph.slice(input, [0, 0, 0, 0], [1, HEADS, half, sequence]);
    let second = graph.slice(input, [0, 0, half, 0], [1, HEADS, half, sequence]);
    let negative = graph.constant_with_scalar(-1.0, scalar_shape());
    let negative_second = graph.multiplication(negative, second);
    let rotated = graph.concat(&[negative_second, first], 2);
    let table_shape = Shape {
        batch: 1,
        channels: 1,
        height: HEAD_DIM,
        width: sequence,
    };
    let cosine = graph.constant(cosine, table_shape);
    let sine = graph.constant(sine, table_shape);
    let direct = graph.multiplication(input, cosine);
    let turned = graph.multiplication(rotated, sine);
    graph.addition(direct, turned)
}

fn rope_reference(sequence: usize, left: &[f32], right: &[f32]) -> Vec<f32> {
    let (cosine, sine) = rope_tables(sequence);
    let mut output = vec![0.0; left.len()];
    for head in 0..HEADS {
        for channel in 0..HEAD_DIM {
            let paired = if channel < HEAD_DIM / 2 {
                channel + HEAD_DIM / 2
            } else {
                channel - HEAD_DIM / 2
            };
            let sign = if channel < HEAD_DIM / 2 { -1.0 } else { 1.0 };
            for position in 0..sequence {
                let index = (head * HEAD_DIM + channel) * sequence + position;
                let paired_index = (head * HEAD_DIM + paired) * sequence + position;
                output[index] = (left[index] + right[index])
                    * cosine[channel * sequence + position]
                    + sign
                        * (left[paired_index] + right[paired_index])
                        * sine[channel * sequence + position];
            }
        }
    }
    output
}

fn build_rope(sequence: usize, identity: bool) -> Result<Arm> {
    let model_shape = shape(sequence, HIDDEN);
    let left_values = values(HIDDEN * sequence, 83);
    let right_values = values(HIDDEN * sequence, 97);
    let mut graph = Graph::new();
    let left = graph.placeholder(model_shape);
    let right = graph.placeholder(model_shape);
    let sum = if identity {
        graph.addition(left, right)
    } else {
        let (cosine, sine) = rope_tables(sequence);
        let left = graph.reshape(
            left,
            Shape {
                batch: 1,
                channels: HEADS,
                height: HEAD_DIM,
                width: sequence,
            },
        );
        let right = graph.reshape(
            right,
            Shape {
                batch: 1,
                channels: HEADS,
                height: HEAD_DIM,
                width: sequence,
            },
        );
        let left = apply_rope(&mut graph, left, &cosine, &sine, sequence);
        let right = apply_rope(&mut graph, right, &cosine, &sine, sequence);
        let sum = graph.addition(left, right);
        graph.reshape(sum, model_shape)
    };
    let _ = sum;
    Ok(Arm {
        label: if identity {
            "RoPE identity comparison"
        } else {
            "two RoPE applications"
        }
        .to_owned(),
        executable: graph
            .compile(NSQualityOfService::UserInteractive)
            .context("compile RoPE arm")?,
        inputs: vec![
            TensorData::with_f32(&left_values, model_shape),
            TensorData::with_f32(&right_values, model_shape),
        ],
        output: TensorData::new(model_shape),
        expected: (!identity).then(|| rope_reference(sequence, &left_values, &right_values)),
        tolerance: 0.01,
        timing_only: identity,
    })
}

fn compile_identity(sequence: usize) -> Result<Executable> {
    let tensor_shape = shape(sequence, HIDDEN);
    let mut graph = Graph::new();
    let input = graph.placeholder(tensor_shape);
    let one = graph.constant_with_scalar(1.0, scalar_shape());
    let _ = graph.multiplication(input, one);
    graph
        .compile(NSQualityOfService::UserInteractive)
        .context("compile empty-chain executable")
}

fn build_chain(sequence: usize, length: usize) -> Result<ChainArm> {
    let tensor_shape = shape(sequence, HIDDEN);
    let expected = values(HIDDEN * sequence, 109);
    let mut executables = Vec::with_capacity(length);
    for _ in 0..length {
        executables.push(compile_identity(sequence)?);
    }
    Ok(ChainArm {
        label: format!("empty executable chain x{length}"),
        executables,
        input: TensorData::with_f32(&expected, tensor_shape),
        first: TensorData::new(tensor_shape),
        second: TensorData::new(tensor_shape),
        expected,
    })
}

fn execute_chain(chain: &ChainArm) -> Result<()> {
    for (index, executable) in chain.executables.iter().enumerate() {
        let source = if index == 0 {
            &chain.input
        } else if index % 2 == 1 {
            &chain.first
        } else {
            &chain.second
        };
        let destination = if index % 2 == 0 {
            &chain.first
        } else {
            &chain.second
        };
        executable
            .run_cached(&[source], &[destination])
            .with_context(|| format!("execute {} step {index}", chain.label))?;
    }
    Ok(())
}

fn verify_chain(chain: &ChainArm) -> Result<f32> {
    execute_chain(chain)?;
    let output = if chain.executables.len() % 2 == 1 {
        &chain.first
    } else {
        &chain.second
    };
    let actual = output.read_f32();
    let error = actual
        .iter()
        .zip(&chain.expected)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    ensure!(error <= 0.001, "{} CPU mismatch: {error:e}", chain.label);
    Ok(error)
}

fn timed_chain(chain: &ChainArm) -> Result<u64> {
    let started = Instant::now();
    execute_chain(chain)?;
    Ok(started.elapsed().as_nanos() as u64)
}

fn measure_chains(left: &ChainArm, right: &ChainArm) -> Result<(Measurement, Measurement)> {
    execute_chain(left)?;
    execute_chain(right)?;
    let left_load = one_minute_load();
    let right_load = one_minute_load();
    let mut left_samples = Vec::with_capacity(REPEATS);
    let mut right_samples = Vec::with_capacity(REPEATS);
    for repeat in 0..REPEATS {
        if repeat % 2 == 0 {
            left_samples.push(timed_chain(left)?);
            right_samples.push(timed_chain(right)?);
        } else {
            right_samples.push(timed_chain(right)?);
            left_samples.push(timed_chain(left)?);
        }
    }
    left_samples.sort_unstable();
    right_samples.sort_unstable();
    Ok((
        Measurement {
            median_ms: left_samples[left_samples.len() / 2] as f64 / 1_000_000.0,
            load_one: left_load,
        },
        Measurement {
            median_ms: right_samples[right_samples.len() / 2] as f64 / 1_000_000.0,
            load_one: right_load,
        },
    ))
}

fn gelu(graph: &mut Graph, input: Tensor) -> Tensor {
    let squared = graph.multiplication(input, input);
    let cubic = graph.multiplication(squared, input);
    let coefficient = graph.constant_with_scalar(0.044_715, scalar_shape());
    let adjusted_cubic = graph.multiplication(cubic, coefficient);
    let inside = graph.addition(input, adjusted_cubic);
    let root = graph.constant_with_scalar((2.0 / std::f32::consts::PI).sqrt(), scalar_shape());
    let inside = graph.multiplication(inside, root);
    let activated = graph.tanh(inside);
    let one = graph.constant_with_scalar(1.0, scalar_shape());
    let half = graph.constant_with_scalar(0.5, scalar_shape());
    let shifted = graph.addition(one, activated);
    let gated = graph.multiplication(input, shifted);
    graph.multiplication(gated, half)
}

fn build_full_layer(sequence: usize, local: bool) -> Result<Arm> {
    let hidden_shape = shape(sequence, HIDDEN);
    let mask_shape = shape(sequence, 1);
    let hidden_values = values(HIDDEN * sequence, 127);
    let mut graph = Graph::new();
    let hidden = graph.placeholder(hidden_shape);
    let key_mask = graph.placeholder(mask_shape);
    let normalized = layer_norm_graph(&mut graph, hidden);
    let qkv = graph.inner_product(
        normalized,
        &weights(3 * HIDDEN * HIDDEN, 131, 1.0 / (HIDDEN as f32).sqrt()),
        HIDDEN,
        3 * HIDDEN,
    );
    let ready_shape = Shape {
        batch: 1,
        channels: HEADS,
        height: HEAD_DIM,
        width: sequence,
    };
    let mut parts = Vec::new();
    for part in 0..3 {
        let sliced = graph.slice(qkv, [0, part * HIDDEN, 0, 0], [1, HIDDEN, 1, sequence]);
        parts.push(graph.reshape(sliced, ready_shape));
    }
    let (cosine, sine) = rope_tables(sequence);
    parts[0] = apply_rope(&mut graph, parts[0], &cosine, &sine, sequence);
    parts[1] = apply_rope(&mut graph, parts[1], &cosine, &sine, sequence);
    for part in &mut parts {
        *part = graph.transpose(*part, [0, 1, 3, 2]);
    }
    let scale = graph.constant_with_scalar(1.0 / (HEAD_DIM as f32).sqrt(), scalar_shape());
    let mut contexts = Vec::new();
    for query_start in (0..sequence).step_by(QUERY_TILE) {
        let query_end = (query_start + QUERY_TILE).min(sequence);
        let query_len = query_end - query_start;
        let (key_start, key_end) = if local {
            (
                query_start.saturating_sub(LOCAL_RADIUS),
                (query_end + LOCAL_RADIUS).min(sequence),
            )
        } else {
            (0, sequence)
        };
        let key_len = key_end - key_start;
        let query = graph.slice(
            parts[0],
            [0, 0, query_start, 0],
            [1, HEADS, query_len, HEAD_DIM],
        );
        let key = graph.slice(
            parts[1],
            [0, 0, key_start, 0],
            [1, HEADS, key_len, HEAD_DIM],
        );
        let value = graph.slice(
            parts[2],
            [0, 0, key_start, 0],
            [1, HEADS, key_len, HEAD_DIM],
        );
        let scores = graph.matrix_multiplication(query, key, false, true);
        let scores = graph.multiplication(scores, scale);
        let padding = graph.slice(key_mask, [0, 0, 0, key_start], [1, 1, 1, key_len]);
        let mut scores = graph.addition(scores, padding);
        if local {
            let distance = distance_mask(query_start, query_len, key_start, key_len);
            let distance = graph.constant(
                &distance,
                Shape {
                    batch: 1,
                    channels: 1,
                    height: query_len,
                    width: key_len,
                },
            );
            scores = graph.addition(scores, distance);
        }
        let probabilities = graph.soft_max(scores, -1);
        contexts.push(graph.matrix_multiplication(probabilities, value, false, false));
    }
    let context = if contexts.len() == 1 {
        contexts[0]
    } else {
        graph.concat(&contexts, 2)
    };
    let context = graph.transpose(context, [0, 1, 3, 2]);
    let context = graph.reshape(context, hidden_shape);
    let projected = graph.inner_product(
        context,
        &weights(HIDDEN * HIDDEN, 137, 1.0 / (HIDDEN as f32).sqrt()),
        HIDDEN,
        HIDDEN,
    );
    let attended = graph.addition(hidden, projected);
    let normalized = layer_norm_graph(&mut graph, attended);
    let projected = graph.inner_product(
        normalized,
        &weights(2 * INTERMEDIATE * HIDDEN, 139, 1.0 / (HIDDEN as f32).sqrt()),
        HIDDEN,
        2 * INTERMEDIATE,
    );
    let activation = graph.slice(projected, [0, 0, 0, 0], [1, INTERMEDIATE, 1, sequence]);
    let gate = graph.slice(
        projected,
        [0, INTERMEDIATE, 0, 0],
        [1, INTERMEDIATE, 1, sequence],
    );
    let activation = gelu(&mut graph, activation);
    let gated = graph.multiplication(activation, gate);
    let output = graph.inner_product(
        gated,
        &weights(
            HIDDEN * INTERMEDIATE,
            149,
            1.0 / (INTERMEDIATE as f32).sqrt(),
        ),
        INTERMEDIATE,
        HIDDEN,
    );
    let _ = graph.addition(attended, output);
    Ok(Arm {
        label: if local {
            "paired full local layer"
        } else {
            "paired full global layer"
        }
        .to_owned(),
        executable: graph
            .compile(NSQualityOfService::UserInteractive)
            .context("compile full layer")?,
        inputs: vec![
            TensorData::with_f32(&hidden_values, hidden_shape),
            TensorData::with_f32(&vec![0.0; sequence], mask_shape),
        ],
        output: TensorData::new(hidden_shape),
        expected: None,
        tolerance: 0.0,
        timing_only: true,
    })
}

fn mask_constant_stats(sequence: usize) -> (usize, usize) {
    let mut values = 0usize;
    let mut constants = 0usize;
    for query_start in (0..sequence).step_by(QUERY_TILE) {
        let query_end = (query_start + QUERY_TILE).min(sequence);
        let key_start = query_start.saturating_sub(LOCAL_RADIUS);
        let key_end = (query_end + LOCAL_RADIUS).min(sequence);
        values += (query_end - query_start) * (key_end - key_start);
        constants += 1;
    }
    // Graph constants are converted from their f32 builder input to fp16 bytes.
    (constants, values * std::mem::size_of::<u16>())
}

fn main() -> Result<()> {
    println!("ModernBERT direct-API sequence scaling attribution");
    println!("All timings are alternating paired medians in one process; every ROW records load1.");
    println!(
        "TIMING-ONLY means the arm is deliberately non-equivalent and makes no numerical claim.\n"
    );

    for sequence in [512usize, 1024, 2048] {
        println!("== sequence {sequence} ==");
        let global_layer = build_full_layer(sequence, false)?;
        let local_layer = build_full_layer(sequence, true)?;
        let (global_layer_m, local_layer_m) = measure_pair(&global_layer, &local_layer)?;
        print_pair(
            sequence,
            &global_layer,
            global_layer_m,
            &local_layer,
            local_layer_m,
        );
        let scheduled_layer_ms =
            (8.0 * global_layer_m.median_ms + 14.0 * local_layer_m.median_ms) / 22.0;
        println!(
            "ATTR seq={sequence} suspect=layer_baseline scheduled_layer_ms={scheduled_layer_ms:.4}"
        );

        let tiled = build_attention(sequence, true, false, false, false)?;
        let untiled = build_attention(sequence, false, false, false, false)?;
        println!("VERIFY PASS {} max_abs={:e}", tiled.label, verify(&tiled)?);
        println!(
            "VERIFY PASS {} max_abs={:e}",
            untiled.label,
            verify(&untiled)?
        );
        let (tiled_m, untiled_m) = measure_pair(&tiled, &untiled)?;
        print_pair(sequence, &tiled, tiled_m, &untiled, untiled_m);
        println!("ATTR seq={sequence} suspect=global_tiling tiled_per_layer_ms={:.4} tiled_fraction_of_global_layer={:.3} untiled_per_layer_ms={:.4} untiled_fraction_of_global_layer={:.3} saved_fraction_of_global_layer={:.3}", tiled_m.median_ms, tiled_m.median_ms / global_layer_m.median_ms, untiled_m.median_ms, untiled_m.median_ms / global_layer_m.median_ms, (tiled_m.median_ms - untiled_m.median_ms) / global_layer_m.median_ms);

        let masked = build_attention(sequence, true, true, true, false)?;
        let unmasked = build_attention(sequence, true, true, false, true)?;
        println!(
            "VERIFY PASS {} max_abs={:e}",
            masked.label,
            verify(&masked)?
        );
        let (masked_m, unmasked_m) = measure_pair(&masked, &unmasked)?;
        print_pair(sequence, &masked, masked_m, &unmasked, unmasked_m);
        let (constant_count, constant_bytes) = mask_constant_stats(sequence);
        println!("ATTR seq={sequence} suspect=masks constants={constant_count} constant_bytes={constant_bytes} overhead_per_local_layer_ms={:.4} fraction_of_local_layer={:.3} timing_only_bound=true", (masked_m.median_ms - unmasked_m.median_ms).max(0.0), (masked_m.median_ms - unmasked_m.median_ms).max(0.0) / local_layer_m.median_ms);

        let layer_norm = build_layer_norm(sequence)?;
        let reduce_mean = build_reduce_mean(sequence)?;
        println!(
            "VERIFY PASS {} max_abs={:e}",
            layer_norm.label,
            verify(&layer_norm)?
        );
        let (mean_m, norm_m) = measure_pair(&reduce_mean, &layer_norm)?;
        print_pair(sequence, &reduce_mean, mean_m, &layer_norm, norm_m);
        println!("ATTR seq={sequence} suspect=layernorm normalizations_per_transformer_layer=2 per_layer_ms={:.4} fraction_of_scheduled_layer={:.3} single_norm_ms={:.4} plain_mean_ms={:.4}", 2.0 * norm_m.median_ms, 2.0 * norm_m.median_ms / scheduled_layer_ms, norm_m.median_ms, mean_m.median_ms);

        let identity_rope = build_rope(sequence, true)?;
        let rope = build_rope(sequence, false)?;
        println!("VERIFY PASS {} max_abs={:e}", rope.label, verify(&rope)?);
        let (identity_m, rope_m) = measure_pair(&identity_rope, &rope)?;
        print_pair(sequence, &identity_rope, identity_m, &rope, rope_m);
        println!("ATTR seq={sequence} suspect=rope per_layer_upper_bound_ms={:.4} incremental_ms={:.4} fraction_of_scheduled_layer={:.3}", rope_m.median_ms, (rope_m.median_ms - identity_m.median_ms).max(0.0), rope_m.median_ms / scheduled_layer_ms);

        let one = build_chain(sequence, 1)?;
        let chain = build_chain(sequence, CHAIN_LENGTH)?;
        println!(
            "VERIFY PASS {} max_abs={:e}",
            one.label,
            verify_chain(&one)?
        );
        println!(
            "VERIFY PASS {} max_abs={:e}",
            chain.label,
            verify_chain(&chain)?
        );
        let (one_m, chain_m) = measure_chains(&one, &chain)?;
        println!(
            "ROW seq={sequence} arm=\"{}\" median_ms={:.4} load1={:.2}",
            one.label, one_m.median_ms, one_m.load_one
        );
        println!(
            "ROW seq={sequence} arm=\"{}\" median_ms={:.4} load1={:.2} ratio_to_paired={:.3}",
            chain.label,
            chain_m.median_ms,
            chain_m.load_one,
            chain_m.median_ms / one_m.median_ms
        );
        println!("ATTR seq={sequence} suspect=iosurface_chain per_layer_upper_bound_ms={:.4} fraction_of_scheduled_layer={:.3} chain_total_ms={:.4}", chain_m.median_ms / CHAIN_LENGTH as f64, chain_m.median_ms / CHAIN_LENGTH as f64 / scheduled_layer_ms, chain_m.median_ms);
        println!();
    }
    Ok(())
}
