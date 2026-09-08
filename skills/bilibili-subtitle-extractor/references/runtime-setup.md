# whisper.cpp runtime setup

This reference covers the optional Apple Silicon `whispercpp` backend. It is
not part of the ordinary official-subtitle or local-subtitle path. The setup
script is opt-in and installs a local runtime; transcription uses only the
artifacts already described by `runtime.json`.

## Commands

The installed skill normally lives at
`~/.codex/skills/bilibili-subtitle-extractor`:

```bash
SETUP=~/.codex/skills/bilibili-subtitle-extractor/scripts/setup_whispercpp.py

python3 "$SETUP" --help
python3 "$SETUP"
python3 "$SETUP" --doctor
```

The default runtime root is:

```text
~/.cache/bilibili-subtitle-extractor/whispercpp-v1.8.7
```

The default configuration path is `runtime.json` beside the installed skill.
Use `--runtime-dir PATH` and `--config PATH` when a different location is
needed. `--backend-config` is an alias for `--config`.

The setup options and their defaults are:

| Option | Default | Purpose |
| --- | --- | --- |
| `--runtime-dir PATH` | `~/.cache/bilibili-subtitle-extractor/whispercpp-v1.8.7` | Runtime cache and receipts |
| `--config PATH` / `--backend-config PATH` | installed skill `runtime.json` | Machine-local extractor configuration |
| `--model NAME` | `small` | OpenAI Whisper model used for conversion |
| `--pt-model PATH` | `~/.cache/whisper/small.pt` | Existing local OpenAI Whisper checkpoint |
| `--force` | off | Rerun stages despite matching receipts |
| `--quantize` / `--coreml-quantize` | on | Use the upstream Core ML FLOAT16 conversion path |
| `--no-quantize` / `--no-coreml-quantize` | off | Explicit FP32 diagnostic conversion |
| `--doctor` | off | Read-only readiness and receipt check |

For a named installation, make the paths explicit and then run the doctor:

```bash
SKILL=~/.codex/skills/bilibili-subtitle-extractor
RUNTIME=~/.cache/bilibili-subtitle-extractor/whispercpp-v1.8.7
CONFIG="$SKILL/runtime.json"

python3 "$SKILL/scripts/setup_whispercpp.py" \
  --runtime-dir "$RUNTIME" \
  --config "$CONFIG"
python3 "$SKILL/scripts/setup_whispercpp.py" \
  --runtime-dir "$RUNTIME" \
  --config "$CONFIG" \
  --doctor
```

Do not use `--force` as a first recovery step. The normal invocation is
resumable and only skips a stage when its receipt fingerprint and output hashes
still match.

## Prerequisites and pinned recipe

The target is macOS on Apple Silicon. The script checks for `git`, `cmake`, and
`uv` on `PATH`; `uv` creates or manages the isolated Python 3.12 environment.
The clone, package installation, and an uncached model download require
network access. CMake also needs a working local compiler and the macOS Core ML
and Metal build environment.

The setup passes the `.venv/bin/python` entrypoint itself to `uv pip`. It keeps
that path even when uv represents it as a symlink, so package installation does
not accidentally target uv's shared base interpreter.

The source is cloned from `ggml-org/whisper.cpp`, tag `v1.8.7`, and verified at
commit `48f628a84833905ee4a0658ee6d4a5c915ce1997`. Before building, the setup
performs the exact active-line replacement:

```text
MLComputeUnitsAll -> MLComputeUnitsCPUAndNeuralEngine
```

The build uses:

```text
-DWHISPER_COREML=ON
-DWHISPER_COREML_ALLOW_FALLBACK=OFF
-DGGML_METAL=ON
-DCMAKE_BUILD_TYPE=Release
--target whisper-cli -j4
```

The isolated environment installs these exact distributions:

```text
torch==2.5.1
coremltools==8.3.0
openai-whisper==20250625
numpy==1.26.4
ane_transformers==0.1.3 --no-deps
```

The last package is installed without its old dependency set because only its
`LayerNormANE` implementation is needed by the pinned conversion script. The
setup verifies the installed distribution versions before continuing.

The OpenAI checkpoint is converted with the pinned upstream
`convert-pt-to-ggml.py`. The Core ML encoder is produced by the pinned
`convert-whisper-to-coreml.py` with:

```text
--model small --encoder-only True --optimize-ane True --quantize True
```

Here `--quantize True` selects the upstream FLOAT16 compute-precision path; it
does not mean low-bit integer weight quantization. The default FP32 conversion
can load while showing no useful ANE power evidence on the tested machine, so
FLOAT16 is the default. `--no-quantize` remains available for an explicit
diagnostic comparison. The compiled encoder is created through
`coremltools.utils.compile_model`; the nonfunctional upstream
`download-coreml-model.sh` helper is not used.

The configured runtime uses a CPU-and-Neural-Engine Core ML encoder and a
Metal decoder. `runtime_identity.compute_units` records that configuration,
and `require_coreml` is `true`. A configuration or Core ML load message proves
initialization only; it does not prove ANE occupancy. Hardware telemetry and
process attribution must be recorded separately when that claim matters.

## Checkpoint and generated files

With no `--pt-model`, the setup reuses `~/.cache/whisper/<model>.pt` when it
exists. If the default checkpoint is absent, it invokes `whisper.load_model`
during setup, which may download that checkpoint into the OpenAI Whisper cache.
A missing custom path passed through `--pt-model` is an error and is never
downloaded implicitly. The Core ML conversion runs through a small `runpy`
wrapper that patches the upstream symbolic `whisper.load_model("small")` call
to that exact explicit path, so the same local checkpoint is the source for
both ggml and Core ML conversion without mutating the Whisper cache.

The accelerated `whispercpp` transcription path never downloads a binary,
checkpoint, or Core ML bundle. The preserved OpenAI fallback can still download
its missing `.pt` checkpoint on first use, so offline workflows should verify
both the accelerated runtime and the OpenAI cache.

The runtime root contains generated, machine-local artifacts similar to:

```text
whisper.cpp/                         pinned source and build tree
.venv/                               isolated Python 3.12 environment
models/ggml-small.bin                ggml model
models/ggml-small-encoder.mlmodelc/  compiled Core ML encoder
receipts/stage-*.json                per-stage fingerprints and hashes
receipts/setup.json                  final installation receipt
```

The uncompiled `.mlpackage` remains under the source checkout's `models/`
directory for the conversion receipt. The setup receipt records the source
commit, source tree hash, binary hash, checkpoint hash, ggml hash, Core ML
package and compiled-bundle tree hashes, exact tool/package versions, and
sanitized commands.

## Machine-local configuration

Setup writes a generated `runtime.json` containing absolute paths for
`whisper_cli`, `model_dir`, and `coreml_encoder`, plus strict Core ML settings
and the runtime identity. Those paths, hashes, and identities belong to the
machine that ran setup. Treat this file as a local installation artifact: do
not include it in a portable skill package or copy it to another machine.
Run setup again on the destination machine to generate its own configuration.

The normal extractor may use the sibling config automatically. An explicit
benchmark can pass the same file through `--backend-config`. The config is a
selection and readiness record, not evidence that every inference used the
Neural Engine.

## Doctor and recovery

`--doctor` makes no changes and does not run inference. It reports whether the
runtime root and `receipts/setup.json` exist, the config is valid JSON with
absolute paths, and `model_dir` contains the requested
`ggml-<model>.bin` plus its matching sibling
`ggml-<model>-encoder.mlmodelc`. It also requires the config paths to match the
receipt paths, verifies the ggml/checkpoint/Core ML/binary hashes, checks every
recorded `*.dylib` hash and the current dylib set under the build tree, and
launches `whisper-cli --version` to catch missing dynamic libraries. It checks
the pinned Python distributions with `importlib.metadata` from the venv's
site-packages metadata and does not import torch. It also requires
`require_coreml: true`, `runtime_identity.compute_units` to be
`CPUAndNeuralEngine`, the pinned source commit, and an executable isolated venv
Python. It exits `0` when all checks pass and `1` when the installation is not
ready.

For an interrupted or partially completed setup, rerun the same command. The
stage receipts make completed work resumable and the setup lock is removed on a
normal error path. If `.setup.lock` remains after a forced process termination,
inspect the recorded PID and timestamp before removing that one lock file; do
not delete the runtime tree blindly.

If the runtime directory contains a checkout at another commit, setup refuses
to replace it. Choose a new `--runtime-dir` or preserve the old tree for
rollback. Existing generated model and compiled Core ML outputs are replaced
through recoverable `.previous-*` siblings when regeneration is required.
When an existing config is updated, the prior JSON is copied to a
`.previous-<sha256-prefix>` sibling and unknown config fields are retained.
Before building or reusing a build receipt, setup rejects tracked source edits
other than the exact `MLComputeUnitsAll` to
`MLComputeUnitsCPUAndNeuralEngine` replacement. Generated untracked build and
model files are allowed. Inspect the receipt and any backup before restoring
one manually; no broad cleanup is part of setup.

After recovery, rerun `--doctor` and inspect the new `receipts/setup.json`.
Only then pass the generated config to the extractor. A successful doctor
check confirms artifact/config readiness, not model accuracy or measured ANE
occupancy.
