# Apple Silicon backend and benchmark notes

Read this reference when setting up, validating, or comparing the optional
`whispercpp` backend. It is not needed for an official-subtitle or local
subtitle import.

## Runtime boundary

The extractor consumes a runtime that is already installed and described by
`runtime.json`. A typical configuration contains absolute paths and an
identity record, for example:

```json
{
  "whisper_cli": "/absolute/path/to/whisper-cli",
  "model_dir": "/absolute/path/to/models",
  "runtime_identity": {
    "whisper_cpp": "cppv1.8.7",
    "decoder": "Metal",
    "encoder_compute_units": "CPUAndNeuralEngine"
  },
  "coreml_enabled": true,
  "compute_units": "CPUAndNeuralEngine",
  "require_coreml": true
}
```

The exact paths and identity values come from the setup receipt. Keep the
runtime setup separate from inference: the optional setup script prepares the
pinned runtime, model, Core ML bundle, hashes, and `runtime.json`; its
`--doctor` mode checks the installed paths and expected artifacts before a
run. Follow [runtime-setup.md](runtime-setup.md) for the exact command and
receipt schema. Normal extraction must not trigger a binary, model, or Core ML
download for the optimized `whispercpp` path, and setup must not mutate system
Python packages. The preserved OpenAI fallback may download a missing `.pt`
checkpoint on first use; verify that cache separately if offline fallback must
work. Run setup and `--doctor` before offline work so the optimized runtime is
known-good.

The reproducible setup target is whisper.cpp `cppv1.8.7` built for Apple
Silicon, using the original OpenAI multilingual `small.pt` as the model
source. Keep the source hash and generated artifact hashes in the setup
receipt. The model and runtime are installation artifacts, not part of a
normal transcription's wall time.

## Core ML configuration that needs evidence

The accelerated path is a strict CPUAndNeuralEngine Core ML encoder with a
Metal decoder. The current tested conversion used the same OpenAI `small.pt`
checkpoint for the ggml F16 model and the Core ML encoder. It passed the
upstream converter options `--optimize-ane True --quantize True`; in this
conversion path, the `--quantize` option selects FLOAT16 compute and does not
mean low-bit quantized weights. Preserve that meaning in receipts and reports.

The default FP32 Core ML graph can load successfully without producing
observable ANE power in the tested setup. The F16 graph produced nonzero ANE
power in that setup. Treat this as measured evidence for that exact artifact
and machine, not as a universal guarantee. Record the graph precision and
runtime identity so a later run cannot confuse a successful Core ML load with
actual ANE execution.

A Core ML load marker proves initialization only. It does not prove ANE
occupancy or process-level attribution. If hardware telemetry is available,
include it as a separate measured output and state that tools such as rootless
rail telemetry are system-wide. Do not prompt for a password for unavailable
root-only `powermetrics`, and do not label a transcript as “ANE” merely
because `coreml_enabled` is true or a load line appeared.

## Backend selection and failure policy

| Request | Selection | Failure behavior |
| --- | --- | --- |
| `--backend auto` | Validated Core ML + Metal runtime on Darwin arm64 | Warn, record `fallback_reason`, and use original OpenAI Whisper when the platform, config, model, Core ML bundle, or runtime execution is unavailable or fails |
| `--backend openai` | Original OpenAI Whisper implementation | Report its actual dependency or transcription error |
| `--backend whispercpp` | The configured whisper.cpp runtime; `require_coreml` controls whether Core ML evidence is required | Fail clearly on missing config, missing model/bundle, conversion failure, backend failure, malformed JSON, or missing strict Core ML load evidence; never silently use OpenAI |

`auto` is a convenience fallback for ordinary work. `whispercpp` is the
strict choice for a benchmark because a fallback would invalidate the engine
comparison. `--force-transcribe` belongs to an explicit Bilibili URL/BV
experiment that must bypass official subtitles; it is not a backend fallback
switch and never overrides local subtitle import.

## Reproducible comparison

Use the following boundaries when producing a formal report:

1. Freeze the original script and hash. Run the original URL workflow once as
   reconnaissance, retaining media, temporary files, and logs. Exclude any
   dependency build or setup overlap from formal timings.
2. Download the source media once and hash it. Give both wrappers the same
   source MP3 for the primary product comparison, including each wrapper's
   conversion in its wall time. For an engine ablation, give both backends the
   same canonical WAV and state any unavoidable decode difference.
3. Warm model caches before warm runs. Run CPU/OpenAI and optimized
   CoreML+Metal sequentially in balanced order, with at least three full-video
   repeats per primary backend when feasible. Include process startup in every
   CLI run and record cold runs separately. Do not run builds or concurrent
   transcriptions during a timed run.
4. Add an explicit whisper.cpp Metal-only run with the same source, model,
   language, thread setting, and decoding settings where possible. Keep the
   product comparison (existing defaults) separate from the matched-decoding
   comparison: upstream whisper.cpp defaults and the original OpenAI CLI may
   use different beam/greedy behavior.
5. Record duration, wall seconds, user and system CPU seconds, average cores,
   real-time factor, maximum RSS, audio-conversion time, transcription time,
   backend startup, model/runtime hashes, OS/chip/power state, run order, and
   the machine-contention caveat. Keep setup, downloads, and first compilation
   out of inference timings, or label them as cold/setup observations.
6. Validate full-duration coverage, finite ordered timestamps, non-empty text,
   and samples at the beginning, middle, and end. Compare engine disagreement
   and manually audit short samples when accuracy matters. Do not fabricate
   CER or treat the CPU transcript as ground truth without a reference.

Do not embed a speed target or a speedup number in this skill. Use the actual
`--metrics-json` receipts and external benchmark outputs. A slower Core ML
configuration remains a valid result: report it, choose the observed best
efficiency mode if a default must be selected, and retain the explicit ANE
option for users who need it.

## What to report

When `--metrics-json` is supplied, preserve the receipt alongside the raw
transcript. It should distinguish requested and resolved backend, model and
language, full-run and stage CPU/wall timing, sanitized commands, fallback
reason, runtime identity, and Core ML load evidence. Cookie values must remain
redacted. Treat runtime configuration as “configured” evidence and telemetry
or a measured power trace as “measured” evidence; keep those labels separate.

The transcript's `字幕来源` remains an honest source label such as
`Whisper small 转写` or `whisper.cpp small 转写`. It describes the engine that
produced the segments, not an unproven hardware execution claim.
