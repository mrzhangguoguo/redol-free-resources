# Bilibili Subtitle Extractor: M3 benchmark

This release adds an optional Apple Silicon transcription backend while preserving official-caption priority, local subtitle imports, the Bilibili 412 browser-cookie retry, and faithful raw/organized Markdown outputs.

## Changes

- `auto`, `openai`, and strict `whispercpp` backend selection; explicit failure and automatic fallback receipts.
- Pinned whisper.cpp v1.8.7; Core ML CPUAndNeuralEngine encoder with FLOAT16 compute and Metal decoder.
- Isolated, resumable setup and a read-only doctor validating checkpoint, model, shared libraries, configuration, and dependency metadata.
- Explicit checkpoint binding and venv interpreter-path isolation.
- Millisecond timestamp normalization, malformed-output rejection, native logs, and stage metrics.

## Measurements, not a universal speed promise

On one M3 MacBook Air (16 GiB), using the same 301.58-second Chinese video and multilingual small model, three final installed runs per backend produced:

| Median | Original OpenAI CPU | Updated Core ML / Metal |
|---|---:|---:|
| Wall time | 227.81 s | 39.92 s |
| Cumulative CPU time | 450.89 s | 12.80 s |
| Maximum RSS | 2.25 GiB | 0.91 GiB |

This is a 5.71× median / 5.65× mean whole-backend speedup and 97.16% less median cumulative CPU time. It is not an ANE-only speedup. A separate five-run-per-backend same-WAV comparison improved median encoder time by 3.60×, but end-to-end means and medians did not consistently favor Core ML over Metal. Both ASR engines retained recognition errors; no full-video CER/WER or accuracy improvement is claimed.

34 Skill tests and 7 benchmark-harness tests passed, alongside real installation, package extraction, URL workflow, strict failure, and fallback checks.

## Install

Unzip `bilibili-subtitle-extractor-skill.zip` into the Skill directory, then follow `SKILL.md`. On an Apple Silicon Mac run the bundled `scripts/setup_whispercpp.py`, then the same helper with `--doctor`.

Machine-local runtime configuration, models, credentials, and test media are excluded. Existing ffmpeg / yt-dlp tools are still required for media downloads, and OpenAI fallback requires the Whisper CLI. No repository-wide license is added by this release.

[Sanitized per-run data](bilibili-subtitle-extractor-m3.json) contains no local paths or video media.
