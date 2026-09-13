# Direct Neural Engine access on M5 Max / macOS 27

Measured 2026-09-12 on an Apple M5 Max, macOS 27.0 (26A5425a), driving the
private `_ANEInMemoryModel` API through the Rust bindings published at
`mutable-state-inc/siliconswarm-at-ensue-plugin` (MIT).

Probes live in `bench/spikes/ane-direct-probe/`. They keep their own workspace
and reach the bindings by relative path, because vendoring an external checkout
into this repository's workspace would make every build depend on a clone that
is not part of it.

Load average was 8 to 12 throughout, not idle. Absolute figures carry that; the
ratios between arms measured in the same run are what the conclusions rest on.

## Verdict

A faithful port of gte-modernbert-base to this API lands at roughly **30 ms** for
a 512-position row, against **25.5 ms** for the same model through the shipping
Core ML lane. So direct access reaches Core ML's league but does not beat it by
being faithful. Any win has to come from doing something Core ML does not.

The gap is concentrated in one place: constant-weight projection and
feed-forward graphs run at 4,800 to 10,500 GFLOP/s, while the complete batched
attention graph runs at about 1,000. This is not a tenfold, apples-to-apples
penalty for runtime operands. The matched-shape probe below found that baking one
operand into `matrix_multiplication` did not make that operator faster; operator
choice, matrix geometry, and the rest of the attention graph account for the
comparison. Attention is still over half the remaining cost.

## The API works here, and computes exactly

The published compatibility matrix for these bindings covers M1 through M4 on
macOS 15. Neither this chip nor this OS is in it, and the bindings resolve the
private framework by `dlopen` at runtime, which a successful build does not
prove.

A projection through an identity matrix returned its input with a worst absolute
error of **exactly 0**. The input was distinct per (channel, position) rather
than constant, so a collapsed projection or a transposed axis would have failed
rather than passed quietly.

One layout fact is easy to get backwards: in this API's NCHW, **the sequence axis
is WIDTH** and the hidden dimension goes on channels. A placeholder narrower than
64 is refused at compile time with `SpatialWidthTooSmall`. With height 1, the
element for channel `c` at position `w` sits at `c * width + w`.

## Cost of a ModernBERT layer, by part

512 positions, model geometry as configured (768 hidden, 12 heads of 64, 1152
intermediate), random weights. Only shapes affect timing, so random values time
identically to trained ones; nothing here says anything about numerics.

| part | median | GFLOP | GFLOP/s |
|---|---:|---:|---:|
| one projection, constant weights | 0.13 ms | 0.60 | 4,825 |
| gated feed-forward, constant weights | 0.26 ms | 2.72 | 10,482 |
| attention, batched over heads | 0.79 ms | 0.81 | 1,023 |
| attention, 128-token window | 0.50 ms | 0.40 | 810 |
| attention, sliced per head | 41.47 ms | 0.81 | 19 |

Assembling those: a global layer is about 1.57 ms and a windowed layer about
1.28 ms. ModernBERT runs global attention on every third layer, so 8 of its 22
layers are global and 14 windowed, giving **30.5 ms** for the model.

### Batching heads matters more than anything else measured

Expressing attention as one matmul per head — slice query, key and value per
head, then two small matmuls each — costs **41 ms**. Reshaping so heads sit on
the channel axis and issuing a single matmul across all of them costs **0.79
ms**, for identical arithmetic. That is a **53x** difference from graph
construction alone, and it was the entire content of an earlier conclusion in
this document that a direct port was 44x slower than Core ML. It was not; the
port was wrong. The reference implementation's own encoder example expresses it
the batched way.

### Windowing is worth less than sequence arithmetic suggests

A 128-token window at 512 positions touches a quarter of the score matrix, but
measured only 1.6x cheaper rather than 4x. Each 128-query tile attends to a
256-position halo, so the saving is 2x in arithmetic before accounting for four
tile matmuls running slightly slower per FLOP than one large one.

## Why dynamic attention is slower

A follow-up run on 2026-09-13 isolated one attention score multiplication:
12 channels of `[512, 64] @ [64, 512]`, or 0.403 GFLOP. Each candidate ran in
the same process immediately back-to-back with the dynamic, already-head-laid-out
baseline. The 25 samples alternated execution order and the table reports their
medians. The one-minute load average was **14.35 for every row**. These are wall
clock figures and include the roughly 92 µs dispatch cost.

Unlike the earlier timing-only arms, every arm in `attention_gap` used distinct
deterministic inputs and had to match an independent CPU matrix multiplication
before it was timed. Maximum absolute errors ranged from 0.000238 to 0.001874;
the failure threshold was 0.003. The reference signal also had to be strong enough
that an all-zero output would fail. **No arm failed** to compile, execute, or match
its reference. Run it from the probe workspace with:

```sh
cargo run --release --bin attention_gap
```

### Layout and operand residency

Lower ratios are faster. The baseline is remeasured beside each candidate rather
than borrowed from another point in the run.

| candidate | candidate ms | paired baseline ms | candidate / baseline | GFLOP/s |
|---|---:|---:|---:|---:|
| reshape + transpose before matmul | 0.187 | 0.222 | 0.842x | 2,150 |
| constant RHS, still `matrix_multiplication` | 0.201 | 0.209 | 0.964x | 2,002 |
| constant RHS through `inner_product` | 0.183 | 0.233 | 0.786x | 2,196 |

The transposes do not explain the gap. They were effectively free in this graph;
the graph containing them was actually 16% faster in the paired run, consistent
with fusion or a different internal tiling choice rather than materialized layout
traffic.

Baking the right-hand operand changed dynamic matmul time by only 4%. That
rejects operand residency as the main cause: `matrix_multiplication` does not
turn into the optimized constant-weight path merely because one input is a
constant. An exactly equivalent `inner_product` was 21% faster, but not ten
times faster. For that comparison the key matrix was shared across heads so the
same useful multiply-adds could be represented by one baked linear; real
attention keys depend on the input and cannot use this formulation.

The earlier 4,800-10,500 versus 1,000 GFLOP/s comparison therefore combines
different operators, shapes, and graph contents. In isolation the canonical
runtime matmul sustained about 1,700-2,020 GFLOP/s. The remaining drop to the
roughly 1,000 GFLOP/s full-attention result includes the second matmul, softmax,
and surrounding graph. The evidence available through this private API points
to its `matrix_multiplication` lowering, not IOSurface residency or explicit
transpose cost; the private compiler exposes no lower-level counter with which
to attribute that lowering further.

### Shape and tiling

Each row performs the same 0.403 GFLOP as the canonical 12-channel
`512 x 64 x 512` baseline. Changing channel count or dimensions changes the
operation's meaning, so these rows diagnose the kernel rather than propose a
faithful attention replacement.

| channels and M x K x N | candidate ms | paired baseline ms | candidate / baseline | GFLOP/s |
|---|---:|---:|---:|---:|
| 6 and 512 x 128 x 512 | 0.173 | 0.220 | 0.786x | 2,332 |
| 3 and 512 x 256 x 512 | 0.148 | 0.207 | 0.715x | 2,713 |
| 1 and 512 x 768 x 512 | 0.121 | 0.200 | 0.605x | 3,332 |
| 12 and 256 x 256 x 256 | 0.168 | 0.212 | 0.789x | 2,403 |
| 12 and 128 x 1024 x 128 | 0.236 | 0.216 | 1.096x | 1,703 |

Geometry matters, but not enough to close the gap. Combining all head dimensions
into one channel was the best case at 1.65x lower latency and 3,332 GFLOP/s. It
also replaces twelve independent attention distributions with one distribution
whose dot products cross head boundaries, so it is not usable by the model.
A larger reduction dimension is not sufficient by itself: the 128 x 1024 x 128
case was 10% slower than canonical.

### Split versus fused heads

These arms preserve the canonical result. Inputs are already in
`[head, sequence, head_dim]` layout, the channel axis is sliced into equal groups,
and grouped outputs are concatenated in their original order. Every row matched
the same CPU reference as the one-op baseline.

| matmul groups | candidate ms | paired baseline ms | candidate / baseline | GFLOP/s |
|---:|---:|---:|---:|---:|
| 2 | 0.348 | 0.227 | 1.533x | 1,157 |
| 3 | 0.309 | 0.208 | 1.480x | 1,305 |
| 4 | 0.264 | 0.238 | 1.111x | 1,523 |
| 6 | 0.233 | 0.208 | 1.120x | 1,726 |
| 12 | 0.228 | 0.228 | 1.001x | 1,763 |

One matmul over all 12 head channels remains the optimum. Twelve correctly laid
out matmuls tied it within 1%, but no split beat it; two and three groups were
about 50% slower. This does not contradict the earlier 53x result, whose sliced
form starts from model layout and carries two matmuls plus softmax per head. It
does show there is no profitable intermediate group count hidden between the
one-op and per-head endpoints.

### Closability verdict

**The gap is not closable by any formulation tested.** Transposes are not the
cost, a constant operand does not accelerate `matrix_multiplication`, and every
semantically faithful split is no faster than one batched op. Shape can recover
at most 1.65x here only by changing multi-head attention's meaning. The best
available faithful formulation therefore remains the existing batched attention
graph, and the estimate remains **30.5 ms for 22 layers versus Core ML's 25.5
ms**. There is no new per-layer extrapolation to report because no valid arm
improved that graph.

## No SRAM cliff on this chip

The binding reference reports ~32 MB of SRAM, with weights under 16 MB streaming
at ~15,000 GB/s and larger ones dropping to ~51 GB/s — a ~300x discontinuity.

It does not reproduce. Growing a square projection's weights from 2 MiB to
72 MiB, past the claimed SRAM size, gave a smooth monotonic curve: 105 µs at
2 MiB, 187 µs at 15.1 MiB, 208 µs at 18.0 MiB, 309 µs at 32 MiB, 565 µs at
72 MiB. Either side of the claimed threshold the curve is continuous.

Two sweeps with different sample points agree on a fixed cost of **~92 µs per
dispatch**, matching the same reference's dispatch-overhead figure. So that
document's dispatch number reproduces here and its memory-hierarchy number does
not.

That sweep cannot attribute its slope: a square projection grows weight bytes
and multiply-accumulates together. Holding weights at 8 MiB and growing only the
sequence separates them — 64 and 128 positions cost the same 145 to 154 µs,
after which cost rises linearly at about 0.48 µs per position. So a dispatch is
fixed-cost bound at short sequences and arithmetic bound at long ones.

## Fusion is not the lever

An earlier reading of the missing cliff concluded that layers should be fused as
deeply as the compiler allows. Measuring real layers retired that: fusing one,
two and three ModernBERT layers changed per-layer time by under 1%, because the
~92 µs dispatch cost is about 0.2% of a 1.5 ms layer.

Fusion is worth pursuing only where the fixed cost is a large share of the work,
which means very short sequences. At serving lengths it is noise.

## A new sequence shape costs 0.2 s, not minutes

This is the finding with product consequences. The Core ML lane carries one
compiled package per bucket, each paying specialization on first load — 32 s at
1024 positions, 579 s at 8192 — and about 275 MB on disk, because every package
embeds its own weights.

Compiling a full ModernBERT-shaped layer through the direct API took 0.20, 0.20,
0.16, 0.19 and 0.15 s at 128, 256, 512, 1024 and 2048 positions. Flat in
sequence length, where Core ML's cost grows superlinearly.

At that price shapes stop being artifacts. A ladder can be compiled at startup
rather than shipped, and its step count stops being a cost worth designing
around. It also revisits the 8192 bucket, which passed parity but was shelved
partly because of a 735 s load.

Extrapolation caveat: this is one layer. A model compiles as per-layer
executables, since the op-depth limit forbids one graph for all 22, so a whole
model at one shape is roughly 4.4 s if compile cost is linear in layer count,
which is untested.

## Weights are not duplicated per shape, as far as process memory can tell

Compiling and EXECUTING five graphs holding 22.5 MiB of distinct weights grew
resident size by 4.8 MiB. Holding one weight set at five sequence shapes grew it
by 0.2 to 1.3 MiB per shape after the first, rising to 6.2 MiB at 2048 — a
pattern that tracks activation buffers, which scale with sequence, rather than
weights, which do not.

Execution matters to that claim and was added after a first attempt measured
only compilation: a graph that never runs need not have materialized anything.

The instrument is the limit here. Process resident size did not reveal the
accelerator memory behind an earlier incident on this machine, where several
decode workers held roughly 100 GB of kernel-wired IOAccelerator memory while
process listings showed nothing. Neural Engine allocations may be accounted the
same way. So this establishes that weights are not duplicated in process-visible
memory, and settling it properly needs system-wide wired-page accounting.

## Instrumentation gap

`run_cached_with_stats` returns 0 ns at every size on this machine. The bindings
set the statistics mask on the request after compilation; the accompanying
comment suggests it belongs on the model before load. Every timing here is
therefore wall clock and includes dispatch overhead.

This is worth fixing if the path is pursued: it is the only tool that would
separate hardware time from framework overhead, including for the Core ML lane
this would be compared against.

## What is not measured

Full-model numerics. The original layer and attribution timings use random
weights and check no output, so none of them says a ported model would produce
correct vectors. The follow-up score-matmul arms do check their complete outputs
against a CPU reference, but a correctness gate against the existing model
reference is still required before any full port is believed.

Two historical arms in `layer_attribution` still report `execute failed`: four
projections in one graph, and its original single dynamic matmul. Both build
graphs with several terminal tensors while that harness binds one output buffer,
so execution refuses them. This is a harness limitation rather than an API one.
Chaining the projections into a single terminal fixes the first case, as the
residency probe does; `attention_gap` now prices isolated dynamic matmul with one
terminal and a checked output.

Everything here is one machine, one chip, one OS build. The private API is
undocumented and can change without notice.
