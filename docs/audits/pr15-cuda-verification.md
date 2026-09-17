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


## Hardware-floor probe verification: PR #17/#18 (`10ec3fc80f0d` / `ce87e1beef98`)

Date: 2026-09-17 UTC

### Result

PR #17/#18's hardware-floor probe (`ck-synapse-worker-cuda --probe-floor`) was verified on real NVIDIA CUDA hardware. The probe executed in **442.47 ms** and printed `{"driver_api":13000,"compute_capability":{"major":8,"minor":9}}` with exit code 0. Both numbers were confirmed against independent `nvidia-smi` measurements (`Driver Version: 580.105.08`, `CUDA Version: 13.0`, compute capability `8.9`).

Failure-path controls verified that device masking (`CUDA_VISIBLE_DEVICES=""`) fails immediately with `cuInit failed with status 100` (exit code 1) rather than fabricating zero readings. Stripping the CUDA runtime libraries off the library search path confirmed an ELF design property on Linux: `ck-synapse-worker-cuda` links `libcublasLt.so.12` dynamically as `DT_NEEDED` (delay-load is currently implemented only for MSVC on Windows in PR #18), producing exit code 127 at loader startup when cuBLASLt is absent even though `--probe-floor` only calls the driver API.

End-to-end integration through the compiled module binary (`ck-synapse`) confirmed that real hardware meets the floor, admits the load, and records the observed triple `{"driver_api": 13000, "compute_capability": 8.9}` under `cuda.observed` in certification evidence with `floor_state: "supported"`. When configured with below-floor environment overrides (`CUDA_COMPUTE_CAPABILITY=7.4`), the module refuses `model.load` before worker process creation with `code: "owned_cuda_unsupported"` (`ComputeCapabilityBelowFloor`), embedding the observed hardware values in the refusal error message.

PR #16's 2-entry LRU shape-plan cache bound was verified in parallel on the same instance: four distinct shapes (batch 64, batch 8, batch 1, and batch 64 repeated) confirmed that plan arenas are evicted when the 2-entry bound is exceeded, device memory is bounded without growing past two arenas, the evicted shape is cleanly rebuilt on cache miss rather than crashing, and the rebuilt plan produces output **byte-identical** to the initial forward (SHA-256 `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02`, identical to `master`).

### Rig and pinned inputs

A single on-demand Vast.ai instance was rented and destroyed upon completion:

- Instance ID: `51297726`
- Host ID: `60742`, Machine ID: `12976`, Geolocation: Denmark, DK (host reliability `0.9977751` / `99.8%`)
- Image/OS: `nvidia/cuda:12.4.1-devel-ubuntu22.04`, Ubuntu 22.04.4 LTS (Kernel `5.4.0-216-generic`, x86_64)
- CPU: Intel Xeon E7-4880 v2, 40 vCPUs effective, 386.8 GiB RAM
- GPU: 1x NVIDIA GeForce RTX 4090, 24,564 MiB VRAM
- GPU UUID: `GPU-85a6453d-b97a-1f65-7c7e-2163406f0396`, PCI Bus ID: `00000000:81:00.0`
- NVIDIA Driver: `580.105.08`
- CUDA Toolkit: 12.4 (`nvcc` 12.4.131, `Build cuda_12.4.r12.4/compiler.34097967_0`)
- Pinned sibling checkouts from `siblings.lock`:
  - `subconscious`: `4a258064f584c2696c8c1294c07652911ed828aa`
  - `commons`: `a03b9621b57e0674a305e8558ce20ef811ceae89`
- Model: `Qwen/Qwen3-Embedding-0.6B` revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
  - `model.safetensors` SHA-256: `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`

Commit attribution across steps:
- PR #17/#18 tree under test: `10ec3fc80f0d8bf1abc108daad0b36bba2614aad` (steps 1, 2, 3, 5, 6)
- PR #17 updated head for step 4: `ce87e1beef98eda1c8427fe594d258932c650a9e` (step 4 module test)
- PR #16 LRU shape cache tree: `5b6685b32429cf450af7a7923941b2ad563ca6a8`
- Baseline `master`: `e7659370d2e6cff90783cd4cc769444b153342e9`

### Direct `--probe-floor` execution and independent cross-check (Step 3)

The worker binary was built from `10ec3fc80f0d`:

```text
cargo build -p synapse-worker-cuda --no-default-features --features cuda --release
```

Direct invocation of `--probe-floor`:

```text
target/release/ck-synapse-worker-cuda --probe-floor
```

Measured outcome:

| Metric | Measured value | Independent cross-check (`nvidia-smi`) | Match |
|---|---|---|---|
| `driver_api` | `13000` | Driver version `580.105.08`, CUDA version `13.0` | Exact |
| `compute_capability.major` | `8` | `nvidia-smi --query-gpu=compute_cap` -> `8.9` | Exact |
| `compute_capability.minor` | `9` | `nvidia-smi --query-gpu=compute_cap` -> `8.9` | Exact |
| Exit code | `0` | - | PASS |
| Wall time | `442.47 ms` | - | Sub-second |
| Exact stdout | `{"driver_api":13000,"compute_capability":{"major":8,"minor":9}}` | - | Single JSON object |
| Stderr | (empty) | - | Clean |

### Module regression test branch (Step 4)

1. On `pr-18` at `10ec3fc80f0d`, `cuda_floor_probe_matches_real_worker_binary_output` in `crates/synapse-module/src/lib.rs:15167` was **vacuous**:
   - The test looked for `PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../target/release/ck-synapse-worker-cuda.exe")`.
   - On Linux, `join("../../../...")` points outside the repository root into the parent directory and hardcodes a Windows `.exe` suffix. `worker.is_file()` evaluated to `false`, causing the test to silently return early (`return; // release worker not staged on this host`) and report green without executing.
   - If symlinked at that path, the test failed with exit status 2 because `run_owned_cuda_probe(&mut Command::new(&worker), ...)` omitted `--probe-floor`, causing clap argument validation to fail (`error: the following required arguments were not provided: --nonce <NONCE>`).
2. On PR #17's updated head `ce87e1beef98` (`pr-17-new`), the test was rewritten: marked `#[ignore]`, asserts that `worker.is_file()`, honours `SYNAPSE_TEST_CUDA_WORKER`, and passes `--probe-floor`.
   The worker was built from `ce87e1beef98` and tested:

   ```text
   SYNAPSE_TEST_CUDA_WORKER=/root/work/synapse-pr17/target/release/ck-synapse-worker-cuda \
     cargo test -p synapse-module --lib cuda_floor_probe_matches_real_worker_binary_output -- --ignored --nocapture
   ```

   **Output:**
   ```text
   running 1 test
   test tests::cuda_floor_probe_matches_real_worker_binary_output ... ok

   test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 437 filtered out; finished in 0.24s
   ```

   The real-hardware branch was taken: it verified `reading.driver_api >= 12040` (measured `13000`) and compute capability `>= 7.5` (measured `8.9`). Pointing `SYNAPSE_TEST_CUDA_WORKER` at a nonexistent path failed loudly with `stage CUDA worker at ...`.

### Failure-path controls (Step 5)

| Control condition | Command / Environment | Exit code | Observed stdout | Observed stderr |
|---|---|---:|---|---|
| Device 0 absent | `CUDA_VISIBLE_DEVICES="" target/release/ck-synapse-worker-cuda --probe-floor` | `1` | (empty) | `Error: cuInit failed with status 100: no CUDA-capable device is detected` |
| Toolkit runtime off library path | `/etc/ld.so.conf.d/` CUDA entries masked, `ldconfig` updated, `target/release/ck-synapse-worker-cuda --probe-floor` | `127` | (empty) | `/root/work/synapse/target/release/ck-synapse-worker-cuda: error while loading shared libraries: libcublasLt.so.12: cannot open shared object file: No such file or directory` |

**Architectural finding on Linux dynamic linking:**
On Windows, PR #18 configures `/DELAYLOAD:cublasLt64_13.dll` so runtime DLL resolution happens on first cuBLASLt call. On Linux, however, the ELF binary carries a direct `DT_NEEDED` entry for `libcublasLt.so.12` and `libcudart.so.12`. Even though `--probe-floor` only requires `libcuda.so.1` (the CUDA driver API), the operating system's dynamic linker refuses to start the process if `libcublasLt.so.12` is not discoverable. Once `LD_LIBRARY_PATH` includes `/usr/local/cuda-12.4/targets/x86_64-linux/lib`, the probe loads and exits 0 cleanly.

### End-to-end module admission and refusal pair (Step 6)

The release module binary (`ck-synapse`) built from `10ec3fc80f0d` was executed with `subc` protocol transport against the `Qwen3-Embedding-0.6B` fixture:

#### 1. Real-hardware admission (floor met)

With no environment overrides, the module invoked the hardware probe via the sibling `ck-synapse-worker-cuda --probe-floor`. The hardware floor was met (`driver_api=13000 >= 12040`, `cc=8.9 >= 7.5`), admitting worker creation:

- `model.status` for `qwen3-embedding-0.6b` reached `state: "ready"` with worker connected (`loaded_worker_models: 1`).
- `probe.start` certification ran over the 64 fixture rows: `status: "certified"`, mean cosine `0.9999977120852748` (certification threshold `>= 0.999`), rank overlap `1.0`.
- `probe.report` recorded the hardware floor evidence:
  ```json
  {
    "cuda": {
      "engine": "owned-cuda",
      "backend": "cuda-ptx",
      "floor_state": "supported",
      "floor_refusal": null,
      "minimum_cuda_driver_api": "12040",
      "minimum_device_cc": "7.5",
      "ptx_virtual_arch": "compute_75",
      "observed": {
        "driver_api": 13000,
        "compute_capability": 8.899999618530273
      },
      "cold_load_ms": 15337.278343,
      "resident_process_count": 1
    }
  }
  ```

#### 2. Below-floor refusal

With environment overrides `SYNAPSE_CUDA_DRIVER_API=13000` and `SYNAPSE_CUDA_COMPUTE_CAPABILITY=7.4` (below the 7.5 floor), the module evaluated `evaluate_cuda_floor`:

- `model.load` was refused **before** worker creation.
- Wire response:
  ```json
  {
    "state": "failed",
    "job_id": "job_0d344caf2430a00a783b2a9e666a8e24",
    "error": {
      "class": "permanent",
      "code": "owned_cuda_unsupported",
      "message": "owned-cuda floor refused before worker creation: decision=owned_cuda_unsupported, observed={\"compute_capability\":7.400000095367432,\"driver_api\":13000}",
      "safe_to_retry_same_request": false
    }
  }
  ```
- The decision mapped internally to `CudaUnsupportedReason::ComputeCapabilityBelowFloor`, exposing the observed pair without starting a worker process.

---

### PR #16: 2-entry LRU shape-plan cache bound verification

PR #16 (`5b6685b32429`) bounds the Qwen3 CUDA shape-plan cache to `max_plans = 2` in `crates/synapse-engine-cuda/src/port/cuda_qwen3.cu`. Both `pr-16` and baseline `master` workers were built with `--features cuda --release` and driven with four distinct shapes in sequence over Unix socket IPC:

- Forward 1: all 64 fixture rows (batch 64, seq 28)
- Forward 2: rows 0..7 (batch 8, seq 13)
- Forward 3: row 63 alone (batch 1, seq 9)
- Forward 4: all 64 fixture rows again (batch 64, seq 28)

#### Stderr allocation and eviction telemetry

| Forward | Shape | PR #16 worker stderr log | Master worker stderr log |
|---|---|---|---|
| Load | - | `CUDA Qwen3 persistent weights: layers=28 dtype=f16 accum=fp32 norm_params=fp32`<br>`CUDA Qwen3 persistent embeddings: vocab=151669 hidden=1024 bytes=310618112` | Same persistent weights and embeddings logged |
| Forward 1 | `64x28` | `CUDA Qwen3 shape 64x28: arena=97031168 workspace=1344 captured_exact=true launches=534 ...` (arena 92.5 MiB allocated) | `CUDA Qwen3 shape 64x28: arena=97031168 workspace=1344 captured_exact=true launches=534 ...` |
| Forward 2 | `8x13` | `CUDA Qwen3 shape 8x13: arena=5586176 workspace=851968 captured_exact=true launches=534 ...` (arena 5.3 MiB allocated; **capacity 2 reached**) | `CUDA Qwen3 shape 8x13: arena=5586176 workspace=851968 captured_exact=true launches=534 ...` |
| Forward 3 | `1x9` | `CUDA Qwen3 shape 1x9: arena=486944 workspace=147456 captured_exact=true launches=534 ...`<br>*(Prior to allocation, LRU victim `64x28` was evicted and freed)* | `CUDA Qwen3 shape 1x9: arena=486944 workspace=147456 captured_exact=true launches=534 ...`<br>*(Unbounded: all 3 plans retained)* |
| Forward 4 | `64x28` | `CUDA Qwen3 shape 64x28: arena=97031168 workspace=1344 captured_exact=true launches=534 ...`<br>*(LRU victim `8x13` evicted; **`64x28` was REBUILT** on cache miss)* | *(No log: cache HIT, 0 allocations)* |

#### Device memory residency across forwards (`nvidia-smi`)

| Forward | Shape | PR #16 device memory (`memory.used`) | Master device memory (`memory.used`) |
|---|---|---:|---:|
| After `LOAD` | - | 397 MiB | 397 MiB |
| Forward 1 | `64x28` | 1,651 MiB (+1,254 MiB context + arena 1) | 1,651 MiB |
| Forward 2 | `8x13` | 1,663 MiB (+12 MiB arena 2) | 1,663 MiB |
| Forward 3 | `1x9` | **1,563 MiB (-100 MiB eviction of `64x28`)** | **1,671 MiB (+8 MiB, no eviction)** |
| Forward 4 | `64x28` | **1,659 MiB (+96 MiB rebuilt arena)** | **1,671 MiB (0 MiB change, cache hit)** |

The eviction was observed: between forward 2 and forward 3, device memory dropped by 100 MiB as the ~92.5 MiB `64x28` arena was freed before allocating the 0.46 MiB `1x9` arena. On forward 4, the evicted `64x28` plan was rebuilt cleanly without crashing, restoring device memory to 1,659 MiB. PR #16's device memory never grew past two arenas.

#### Byte identity and exactness

| Build | Forward | Payload bytes | Raw `f32` SHA-256 | Identity check |
|---|---:|---:|---|---|
| PR #16 | 1 | 262,144 | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` | Baseline |
| PR #16 | 2 | 32,768 | `f0655eb993b4ef088d1d03e1668e67efef6cec48cb5c013d12509770d74a7a17` | Batch 8 |
| PR #16 | 3 | 4,096 | `a55e55795c66939561e4f15d45e903bb187226533325257122fa713d62baa9ee` | Batch 1 |
| PR #16 | 4 | 262,144 | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` | **Byte-identical to Forward 1** |
| `master` | 1 | 262,144 | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` | **Byte-identical to PR #16 Fwd 1** |
| `master` | 4 | 262,144 | `8f813b09d30c71f8ff67e6b6f41fae7f6bc7c9b7c96f3af66a090e218b4cdd02` | Byte-identical |

1. **Rebuilt plan determinism:** Forward 4 (rebuilt plan after eviction) produced the identical SHA-256 hash and 64/64 byte-identical row vectors to Forward 1. Rebuilding an evicted shape plan introduces zero numerical drift or defect.
2. **Cross-build parity with master:** Forward 1 on PR #16 was byte-identical to `master` Forward 1 across all 64 rows. The 2-entry LRU cache changes retention policy only, with zero impact on arithmetic.

---

### Spend and teardown

- Instance: `51297726` on host `60742` (RTX 4090, Denmark)
- Rental rate: `$0.4400/hour` GPU + `$0.0111/hour` storage = `$0.4511/hour` total.
- Vast account credit before rental: `$11.912374303800476`
- Vast account credit after destruction: `$10.69722990380049`
- Measured total spend: **$1.2151444** (GPU charges: $1.181; storage: $0.030; network download: $0.004; invoice total $1.215). Well within the $3.00 budget cap.
- Instance `51297726` was destroyed with `vastai destroy instance 51297726`.
- Post-destruction verification: `vastai show instances-v1` returned `Total: 0 instances` with `No instances found`. No rented instance was left running.
