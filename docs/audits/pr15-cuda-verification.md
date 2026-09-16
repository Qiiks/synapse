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


## Re-verification: `ad3c62794fce851b181faccb4b777bbac82872fd`

Date: 2026-09-16 UTC

### Why this section exists

The earlier section verified **one forward only**. After that run, the contributor found a multi-forward defect: the second forward failed with `Qwen3 CUDA received invalid dimensions` because the non-upload path derived `layer_count` from the now-empty host parameter slice. This section exists specifically because that defect was invisible to the original single-request test.

The new head, `ad3c62794fce851b181faccb4b777bbac82872fd`, passes the repeated-forward sequence. The second forward, two different-shape forwards, and the post-`PING` forward all completed, and every returned row was byte-identical to `master`. Within the PR worker, the three repeated 64-row forwards were also byte-identical. There was no non-identity for which a maximum absolute difference or first differing index applied.

### Rig, builds, and pinned inputs

A single on-demand Vast.ai instance was used:

- Instance: `51217330`, verified host reliability `0.9952378`
- Image/OS: `nvidia/cuda:12.4.1-devel-ubuntu22.04`, Ubuntu 22.04.4 LTS
- GPU: NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9
- GPU UUID: `GPU-c1d08dd5-8e3c-1cb0-9def-5ff5b37e0bec`
- NVIDIA driver: `580.95.05`
- CUDA runtime/toolkit: 12.4; `nvcc` 12.4.131
- Baseline `master`: `e88128cd8a687c1599f53f71f13bc51bf585e83c`
- PR #15: `ad3c62794fce851b181faccb4b777bbac82872fd`
- `subconscious`: `0ed4dcb54264f96930cda6f1f097885094a6d89d`
- `commons`: `43ca3541ca0969e2e1eff46c46876158911a95a6`
- Model: `Qwen/Qwen3-Embedding-0.6B` revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
- `model.safetensors` SHA-256: `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`
- Fixture file SHA-256: `7e6be40222f24609d4d80dd93ad00b1d380ef82be27c5f02ab219b64fb48c098`

Each worker was built from its named checkout using the following command. The retained `master` binary came from the initial clean target build, and the final PR binary was rebuilt independently in a fresh target directory before the reported run:

```text
cargo build -p synapse-worker-cuda --no-default-features --features cuda --release
```

Both builds passed and reported `ck-synapse-worker-cuda 0.1.0-alpha.2`.

The same Python protocol client drove both workers. It tokenized the 64 fixture texts once and reused the same in-memory token-ID rows for both builds and all forwards. The tokenization policy was unchanged from the first run: add special tokens, remove a trailing EOS if present, truncate to 2048 IDs, and append EOS `151643`. The corpus again contained 822 IDs total, with row lengths from 5 to 28; `p00` again produced `[785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13, 151643]`. Each worker received one `LOAD`, followed by the five requested operations with `pooling=last` and `normalize=true`.

### Cross-build identity by forward

SHA-256 is over each complete, unmodified little-endian `f32` payload. Row identity compares the corresponding 1024-value row payloads directly.

| Forward | Request | Identical rows | `master` raw payload SHA-256 | PR raw payload SHA-256 |
|---:|---|---:|---|---|
| 1 | fixture rows 0..63 | 64/64 | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` |
| 2 | fixture rows 0..63 again | 64/64 | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` |
| 3 | fixture rows 0..7 | 8/8 | `f0655eb993b4ef088d1d03e1668e67efef6cec48cb5c013d12509770d74a7a17` | `f0655eb993b4ef088d1d03e1668e67efef6cec48cb5c013d12509770d74a7a17` |
| 4 | fixture row 63 | 1/1 | `a55e55795c66939561e4f15d45e903bb187226533325257122fa713d62baa9ee` | `a55e55795c66939561e4f15d45e903bb187226533325257122fa713d62baa9ee` |
| 5 | `PING`, then fixture rows 0..63 | 64/64 | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` |

Both `PING` requests returned `PONG` with one model loaded before forward 5. No forward differed, so maximum absolute difference and first differing index are not applicable.

### Within-PR repeat identity

| PR comparison | Identical rows | Left raw payload SHA-256 | Right raw payload SHA-256 |
|---|---:|---|---|
| forward 1 vs forward 2 | 64/64 | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` |
| forward 1 vs forward 5 | 64/64 | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` |
| forward 2 vs forward 5 | 64/64 | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` |

The persistent PR weights therefore reproduced the originally uploaded bytes on both the immediate repeat and the repeat after shape changes and a control frame.

### Fixture comparison

Forward 1 was compared row-by-row to `probe_corpus_qwen3_embedding_fp32.json`:

| Build | Mean cosine | Minimum cosine | Certification threshold |
|---|---:|---:|---:|
| `master` | 0.9999977120864367 | 0.9999958651986215 | >= 0.999 |
| PR #15 at `ad3c62794fce` | 0.9999977120864367 | 0.9999958651986215 | >= 0.999 |

Both builds clear the gate, and the summaries are equal because forward 1 is byte-identical.

### Resident memory across forwards

Values were read from `/proc/<worker-pid>/status` 200 ms after `LOADED` and after the indicated `VECTORS` response. `VmHWM` is cumulative.

| Build | Measurement point | `VmRSS` KiB (MiB) | `VmHWM` KiB (MiB) |
|---|---|---:|---:|
| `master` | after load | 2,536,656 (2,477.20) | 4,766,736 (4,655.02) |
| `master` | after forward 1 | 2,789,076 (2,723.71) | 4,766,736 (4,655.02) |
| `master` | after forward 2 | 2,789,076 (2,723.71) | 4,766,736 (4,655.02) |
| `master` | after forward 5 | 2,799,928 (2,734.30) | 4,766,736 (4,655.02) |
| PR #15 | after load | 2,536,892 (2,477.43) | 4,767,244 (4,655.51) |
| PR #15 | after forward 1 | 483,148 (471.82) | 4,767,244 (4,655.51) |
| PR #15 | after forward 2 | 483,148 (471.82) | 4,767,244 (4,655.51) |
| PR #15 | after forward 5 | 487,920 (476.48) | 4,767,244 (4,655.51) |

The extra points confirm the earlier interpretation. Peak RSS is effectively unchanged and was reached during `LOAD` for both builds; the PR high-water mark was 508 KiB higher in this run. The reduction is in post-upload resident memory: PR used 2,305,928 KiB (2,251.88 MiB) less after forwards 1 and 2, and 2,312,008 KiB (2,257.82 MiB) less after forward 5. The low PR residency persisted through the repeated forward, new shape plans, singleton, `PING`, and final repeat.

### Corrupted digest refusal

The final byte of a copied `model.safetensors` was flipped at zero-based offset `1,191,586,415`. Its SHA-256 changed from `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` to `72e2352bffa34bc69b4aa12958d8d6133fdf6577b9ef21eff7b370a656343857`. A fresh PR worker was given the corrupted file with the pinned original digest and refused it:

```json
{
  "type": "ERR",
  "req_id": "corrupt-load",
  "code": "artifact_invalid",
  "msg": "artifact digest mismatch: expected 0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd, got 72e2352bffa34bc69b4aa12958d8d6133fdf6577b9ef21eff7b370a656343857"
}
```

No `LOADED` response was produced.

### Spend and teardown

The instance rate, including 70 GB of storage, was `$0.4181111111/hour`. Vast account credit was `$12.00629265880059` before rental and `$11.942570864800501` at and after destruction, for a measured total spend of **$0.063721794**. No other instance was present at the start.

Instance `51217330` was destroyed after the results file was copied off-host. A post-destruction `show instances-v1` query returned `instances_found: 0`, `total_instances: 0`, and an empty `instances` array. The account credit remained unchanged after destruction. No rented instance was left running.
