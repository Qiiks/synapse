# Direct-API GTE ModernBERT probe

This standalone Cargo workspace drives `_ANEInMemoryModel` directly. It is not a
member of the repository workspace because the private-API binding comes from a
separate MIT-licensed checkout.

## Full-model correctness probe

Build and run with a local Hugging Face snapshot and the committed pre-tokenized
rows:

```sh
cargo run --release --bin modernbert_full -- \
  "$MODEL_SNAPSHOT" rows.jsonl \
  --seq 512 \
  --layers-per-executable 1 \
  --warm-repetitions 5 \
  --report "$REPORT_PATH" \
  --vectors-out "$VECTORS_PATH"
```

`$MODEL_SNAPSHOT` must contain `config.json` and `model.safetensors`. The binary
never tokenizes. Each JSONL row supplies either `input_ids` for one shape or an
`input_ids_by_shape` object keyed by sequence length. IDs are padded with the
checkpoint's pad ID, matching the production engine boundary.

The default is one transformer layer per executable. `--layers-per-executable
2` is the bounded fusion control. Attention keeps heads on the channel axis and
uses one batched matrix multiplication per query tile rather than dispatching a
matrix multiplication per head.

The report contains every normalized 768-dimensional vector, row-set and model
digests, row-wise cosine against the built-in CPU fp32 reference, determinism,
checkpoint localization, warm wall time, and one-minute load average. A 512 run
exits unsuccessfully unless minimum cosine is at least 0.999 and repeated vectors
are byte-identical.

`rows.jsonl` contains four short real-text rows and four repeated-prose rows near
each of the 512, 1024, and 2048 shape limits. Its SHA-256 is
`f4889a38df77b9940ce973c4d9b82857d0c401987ae8e77b5ca25e6062808c39`.
