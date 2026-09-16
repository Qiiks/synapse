# PR #15 CUDA verification

Date: 2026-09-16 UTC

## Result

PR #15 preserves the owned-CUDA embedding bytes for this certification corpus. All **64/64 rows were byte-identical** to the baseline. The concatenated 64 × 1024 little-endian `f32` payload had SHA-256 `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` for both workers. There is therefore no differing row, maximum absolute difference, or first differing index to report.

The steady-state process RSS after the first embed was **2,331,812 KiB (2,277.16 MiB, 2.224 GiB) lower** for PR #15 than for `master`. Peak RSS did not improve: both workers reached about 4.55 GiB while loading. The measured claim is thus a post-first-embed residency reduction, not a reduction in load-time peak memory.

All requested checks completed.

## Rig and pinned inputs

A single on-demand Vast.ai instance was used:

- Instance: `51215428`, verified host reliability `0.9994451`
- Image/OS: `nvidia/cuda:12.4.1-devel-ubuntu22.04`, Ubuntu 22.04.4 LTS
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9
- GPU UUID: `GPU-c5103ade-d0a0-5dbb-4f1f-f6c38314f8fd`
- NVIDIA driver: `580.82.09`
- CUDA driver API: `13000` (13.0)
- CUDA runtime/toolkit: `12040` / 12.4; `nvcc` 12.4.131
- Baseline `master`: `e88128cd8a687c1599f53f71f13bc51bf585e83c`
- PR #15: `f28da35bf654c20abf8de0520e7b6ad036b23da8`
- `subconscious`: `0ed4dcb54264f96930cda6f1f097885094a6d89d`
- `commons`: `43ca3541ca0969e2e1eff46c46876158911a95a6`
- Model: `Qwen/Qwen3-Embedding-0.6B` revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
- `model.safetensors` SHA-256: `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`

The baseline was pinned to the task's `master` commit above rather than a later moving remote `master`. Both worktrees used the same sibling checkouts and model directory. Each worker was built in its own worktree with:

```text
cargo build -p synapse-worker-cuda --no-default-features --features cuda --release
```

Both builds passed and identified themselves as `ck-synapse-worker-cuda 0.1.0-alpha.2`.

## Input and protocol method

One Python client drove both binaries over the version-1 Unix-socket worker protocol. It tokenized all 64 fixture texts once, retained that one token-ID set in memory, and sent it unchanged to each worker. The policy was:

1. `add_special_tokens=true`
2. remove a trailing EOS (`151643`) if present
3. truncate to 2048 IDs
4. append EOS (`151643`)

The 64 rows contained 822 token IDs in total; row lengths ranged from 5 to 28. The first fixture row (`p00`, `The quick brown fox jumps over the lazy dog.`) produced:

```text
[785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13, 151643]
```

For each binary, the client sent one `LOAD` and then one `EMBED_BATCH` containing all 64 rows with `pooling=last` and `normalize=true`. It read the worker's raw little-endian `f32` frame and SHA-256 hashed each 1024-value row independently. The reported aggregate hash is over the unmodified concatenated raw frame.

## Byte identity

| Build | Identical rows | Raw 64-row `f32` SHA-256 |
|---|---:|---|
| `master` | 64/64 | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` |
| PR #15 | 64/64 | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` |

No row differed. Maximum absolute difference and first differing index are not applicable.

## Fixture comparison

Cosine was computed independently for every worker row against the corresponding `vector` in `probe_corpus_qwen3_embedding_fp32.json`, then summarized across all 64 rows.

| Build | Mean cosine | Minimum cosine | Certification threshold |
|---|---:|---:|---:|
| `master` | 0.9999977120864367 | 0.9999958651986215 | >= 0.999 |
| PR #15 | 0.9999977120864367 | 0.9999958651986215 | >= 0.999 |

Both sets clear the threshold, and the equal summaries follow from their byte identity.

## Resident memory

Values below were read from `/proc/<worker-pid>/status` 200 ms after the worker returned the `LOADED` response and 200 ms after it returned the first and only `VECTORS` response. `VmHWM` is a cumulative high-water mark and therefore does not fall after host buffers are released.

| Build | Measurement point | `VmRSS` KiB (MiB) | `VmHWM` KiB (MiB) |
|---|---|---:|---:|
| `master` | after load | 2,536,288 (2,476.84) | 4,767,412 (4,655.68) |
| `master` | after first embed | 2,812,784 (2,746.86) | 4,767,412 (4,655.68) |
| PR #15 | after load | 2,535,760 (2,476.33) | 4,767,052 (4,655.32) |
| PR #15 | after first embed | 480,972 (469.70) | 4,767,052 (4,655.32) |

The after-load RSS values are nearly equal because PR #15 uploads weights and the embedding table on the first forward call, not during `LOAD`. After that first embed, PR #15 used 2,331,812 KiB less RSS. Worker diagnostics also recorded the PR-only persistent embedding upload as 310,618,112 bytes (`vocab=151669`, `hidden=1024`); both workers recorded 28 persistent f16 weight layers.

## Corrupted digest refusal

A copy of `model.safetensors` was changed by flipping its final byte at zero-based offset `1,191,586,415`. Its SHA-256 changed from:

```text
0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd
```

to:

```text
72e2352bffa34bc69b4aa12958d8d6133fdf6577b9ef21eff7b370a656343857
```

The PR worker was asked to load that file while supplying the pinned original digest. It refused the load with:

```json
{
  "type": "ERR",
  "req_id": "corrupt-load",
  "code": "artifact_invalid",
  "msg": "artifact digest mismatch: expected 0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd, got 72e2352bffa34bc69b4aa12958d8d6133fdf6577b9ef21eff7b370a656343857"
}
```

No `LOADED` response or model reference was produced. The streamed digest path therefore remained fail-closed for this corruption.

## Spend and teardown

The instance rate, including 70 GB of storage, was `$0.4594444444/hour`. Vast account credit was `$12.06887349680062` before rental and `$12.015695629800604` at and after destruction, for a measured total spend of **$0.053177867**. No other instance was present when the rental began.

The instance was destroyed after evidence collection. A post-destruction `show instances-v1` query returned no active entry with ID `51215428`, and the account charge stopped changing. No rented instance was left running.
