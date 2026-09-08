"""Stdlib contract tests for the optional whisper.cpp transcription backend.

These tests deliberately use temporary files and mocked platform detection. No
model is loaded and no ASR process is started. They exercise the validation,
selection, output normalization, command receipt, and metrics contracts that
the production CLI relies on.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "extract_bilibili_subtitles.py"
SPEC = importlib.util.spec_from_file_location("bilibili_extractor_backend_contract", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import setup failure
    raise RuntimeError(f"Could not load extractor from {SCRIPT_PATH}")
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


def make_runtime_tree(tmp: str, *, model: str = "small", coreml: bool = False) -> tuple[Path, Path, Path]:
    root = Path(tmp)
    cli = root / "bin" / "whisper-cli"
    cli.parent.mkdir(parents=True, exist_ok=True)
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(cli.stat().st_mode | 0o111)
    model_dir = root / "models"
    model_dir.mkdir()
    model_path = model_dir / f"ggml-{model}.bin"
    model_path.write_bytes(b"fixture model")
    if coreml:
        (model_dir / f"ggml-{model}-encoder.mlmodelc").mkdir()
    return cli, model_dir, model_path


class WhisperCppOutputContractTests(unittest.TestCase):
    def test_cpp_offsets_are_milliseconds_and_normalize_like_other_subtitles(self) -> None:
        payload = {
            "transcription": [
                {"offsets": {"from": 4_200, "to": 5_000}, "text": " second  &amp; "},
                {"offsets": {"from": 1_250, "to": 2_000}, "text": " first <i>line</i> "},
                {"offsets": {"from": 3_000, "to": 3_500}, "text": "   "},
            ]
        }

        segments = extractor.parse_whispercpp_json(payload)

        self.assertEqual(
            segments,
            [
                {"start": 1.25, "end": 2.0, "text": "first line"},
                {"start": 4.2, "end": 5.0, "text": "second &"},
            ],
        )
        # The generic parser must also recognize the upstream shape when a
        # caller does not know which engine produced the JSON.
        self.assertEqual(extractor.parse_json_subtitle(payload), segments)

    def test_cpp_empty_and_malformed_output_are_rejected_at_the_right_boundary(self) -> None:
        self.assertEqual(extractor.parse_whispercpp_json({"transcription": []}), [])
        with self.assertRaisesRegex(extractor.SubtitleError, "transcription array"):
            extractor.parse_whispercpp_json({"unexpected": []})
        with self.assertRaises(extractor.SubtitleError):
            extractor.parse_whispercpp_json("not an object")


class WhisperCppConfigContractTests(unittest.TestCase):
    def test_public_config_loader_returns_object_and_does_not_create_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runtime.json"
            payload = {"whisper_cli": "/tmp/whisper-cli", "model_dir": "/tmp/models"}
            path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = extractor.load_backend_config(path)

        self.assertEqual(loaded, payload)

    def test_valid_runtime_derives_model_and_coreml_paths_without_downloading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cli, model_dir, model_path = make_runtime_tree(tmp, coreml=True)
            runtime = extractor.validate_whispercpp_config(
                {
                    "whisper_cli": str(cli),
                    "model_dir": str(model_dir),
                    "runtime_identity": {"build": "fixture", "metal": True},
                    "require_coreml": True,
                },
                "small",
            )

        self.assertEqual(runtime.whisper_cli, cli.resolve())
        self.assertEqual(runtime.model_dir, model_dir.resolve())
        self.assertEqual(runtime.model_path, model_path.resolve())
        self.assertEqual(runtime.coreml_encoder, (model_dir / "ggml-small-encoder.mlmodelc").resolve())
        self.assertTrue(runtime.require_coreml)
        self.assertEqual(runtime.runtime_identity["build"], "fixture")

    def test_config_requires_absolute_eligible_executable_model_and_coreml_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cli, model_dir, _ = make_runtime_tree(tmp)
            base = {
                "whisper_cli": str(cli),
                "model_dir": str(model_dir),
            }
            with self.assertRaisesRegex(extractor.SubtitleError, "absolute path"):
                extractor.validate_whispercpp_config(
                    {**base, "whisper_cli": "bin/whisper-cli"}, "small"
                )
            with self.assertRaisesRegex(extractor.SubtitleError, "model not found"):
                extractor.validate_whispercpp_config(base, "medium")
            with self.assertRaisesRegex(extractor.SubtitleError, "Core ML encoder bundle"):
                extractor.validate_whispercpp_config({**base, "require_coreml": True}, "small")
            with self.assertRaisesRegex(extractor.SubtitleError, "boolean"):
                extractor.validate_whispercpp_config({**base, "require_coreml": "yes"}, "small")

    def test_auto_falls_back_with_reason_but_explicit_cpp_fails_strictly(self) -> None:
        metrics = extractor.MetricsRecorder("auto", "small", "Chinese")
        with mock.patch.object(extractor, "is_apple_silicon", return_value=True):
            resolved, runtime = extractor.resolve_transcription_backend(
                "auto", "small", backend_config=Path(tmp := "/definitely/missing/runtime.json"), metrics=metrics
            )
        self.assertEqual((resolved, runtime), ("openai", None))
        self.assertIn("unavailable", metrics.data["fallback_reason"])

        with self.assertRaisesRegex(extractor.SubtitleError, "Backend config not found"):
            extractor.resolve_transcription_backend(
                "whispercpp", "small", backend_config=Path(tmp)
            )

    def test_auto_requires_validated_arm_coreml_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cli, model_dir, _ = make_runtime_tree(tmp, coreml=True)
            config = {
                "whisper_cli": str(cli),
                "model_dir": str(model_dir),
                "runtime_identity": "fixture-coreml-metal",
            }
            with mock.patch.object(extractor, "is_apple_silicon", return_value=True):
                resolved, runtime = extractor.resolve_transcription_backend(
                    "auto", "small", backend_config=config
                )
            self.assertEqual(resolved, "whispercpp")
            self.assertIsNotNone(runtime)
            assert runtime is not None
            self.assertTrue(runtime.require_coreml)

            with mock.patch.object(extractor, "is_apple_silicon", return_value=False):
                fallback, fallback_runtime = extractor.resolve_transcription_backend(
                    "auto", "small", backend_config=config
                )
            self.assertEqual((fallback, fallback_runtime), ("openai", None))


class BackendCommandAndMetricsContractTests(unittest.TestCase):
    def test_command_receipt_redacts_cookie_values_and_preserves_other_arguments(self) -> None:
        command = [
            "/tmp/whisper-cli",
            "-m",
            "/tmp/ggml-small.bin",
            "-l",
            "zh",
            "-t",
            "4",
            "--cookies",
            "/Users/private/cookies.txt",
            "--cookies-from-browser",
            "chrome",
        ]

        sanitized = extractor.sanitize_command(command)

        self.assertEqual(sanitized[:7], command[:7])
        self.assertEqual(sanitized[7:], ["--cookies", "<redacted>", "--cookies-from-browser", "<redacted>"])
        self.assertNotIn("cookies.txt", " ".join(sanitized))
        self.assertNotIn("chrome", sanitized)

    def test_metrics_include_full_run_and_stage_cpu_wall_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics.json"
            metrics = extractor.MetricsRecorder("whispercpp", "small", "Chinese", metrics_path)
            metrics.set_backend("whispercpp")
            metrics.record_command(["whisper-cli", "--cookies", "/secret/cookies.txt", "--threads", "2"])
            with metrics.stage("audio_conversion"):
                pass
            with metrics.stage("transcription"):
                pass
            metrics.finish(success=True)
            metrics.write()
            receipt = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertEqual(receipt["status"], "success")
        self.assertTrue(receipt["commands_sanitized"])
        self.assertEqual(receipt["commands"][0][2], "<redacted>")
        self.assertNotIn("cookies.txt", json.dumps(receipt["commands"], ensure_ascii=False))
        for mapping in (receipt, receipt["stages"]["audio_conversion"], receipt["stages"]["transcription"]):
            self.assertIn("wall_seconds", mapping)
            self.assertIn("cpu_seconds", mapping)
            self.assertIn("user_cpu_seconds", mapping)
            self.assertIn("system_cpu_seconds", mapping)
            self.assertIn("average_cores", mapping)
            self.assertGreaterEqual(mapping["wall_seconds"], 0.0)
            self.assertGreaterEqual(mapping["cpu_seconds"], 0.0)

    def test_whispercpp_command_forwards_model_language_and_threads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli, model_dir, _ = make_runtime_tree(tmp)
            runtime = extractor.validate_whispercpp_config(
                {
                    "whisper_cli": str(cli),
                    "model_dir": str(model_dir),
                    "runtime_identity": "fixture-metal",
                },
                "small",
            )
            media = root / "clip.wav"
            media.write_bytes(b"RIFF fixture")
            temp_dir = root / "run"
            calls: list[list[str]] = []

            def fake_run(command: list[str], **_: object) -> object:
                calls.append(command)
                output_base = Path(command[command.index("--output-file") + 1])
                output_base.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "transcription": [
                                {"offsets": {"from": 0, "to": 1250}, "text": "hello"}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                return extractor.subprocess.CompletedProcess(command, 0, "", "")

            with mock.patch.object(extractor.subprocess, "run", side_effect=fake_run):
                bundle = extractor.transcribe_with_whispercpp(
                    media,
                    temp_dir,
                    "clip",
                    "small",
                    "Chinese",
                    runtime,
                    threads=3,
                )

        self.assertEqual(bundle.source_label, "whisper.cpp small 转写")
        self.assertEqual(bundle.segments[0]["start"], 0.0)
        self.assertEqual(bundle.segments[0]["end"], 1.25)
        self.assertEqual(len(calls), 1)
        command = calls[0]
        self.assertEqual(command[0], str(runtime.whisper_cli))
        self.assertEqual(command[command.index("--model") + 1], str(runtime.model_path))
        self.assertEqual(command[command.index("--file") + 1], str(media))
        self.assertEqual(command[command.index("--language") + 1], "zh")
        self.assertEqual(command[command.index("--threads") + 1], "3")
        self.assertNotIn("--cookies", command)

    def test_whispercpp_strict_coreml_requires_load_evidence_and_accepts_valid_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli, model_dir, _ = make_runtime_tree(tmp, coreml=True)
            runtime = extractor.validate_whispercpp_config(
                {
                    "whisper_cli": str(cli),
                    "model_dir": str(model_dir),
                    "require_coreml": True,
                },
                "small",
            )
            media = root / "clip.wav"
            media.write_bytes(b"RIFF fixture")

            def write_output(command: list[str], stderr: str) -> object:
                output_base = Path(command[command.index("--output-file") + 1])
                output_base.with_suffix(".json").write_text(
                    json.dumps({"transcription": [{"offsets": {"from": 0, "to": 1000}, "text": "ok"}]}),
                    encoding="utf-8",
                )
                return extractor.subprocess.CompletedProcess(command, 0, "", stderr)

            def no_evidence(command: list[str], **_: object) -> object:
                return write_output(command, "runtime started")

            with self.assertRaisesRegex(extractor.SubtitleError, "Core ML model load evidence"):
                with mock.patch.object(extractor.subprocess, "run", side_effect=no_evidence):
                    extractor.transcribe_with_whispercpp(
                        media, root / "no-evidence", "clip", "small", "Chinese", runtime
                    )

            def loaded(command: list[str], **_: object) -> object:
                return write_output(command, "whisper_init_state: Core ML model loaded")

            with mock.patch.object(extractor.subprocess, "run", side_effect=loaded):
                bundle = extractor.transcribe_with_whispercpp(
                    media, root / "loaded", "clip", "small", "Chinese", runtime
                )
            self.assertEqual(bundle.segments[0]["text"], "ok")

    def test_whispercpp_reports_command_failure_empty_output_and_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli, model_dir, _ = make_runtime_tree(tmp)
            runtime = extractor.validate_whispercpp_config(
                {"whisper_cli": str(cli), "model_dir": str(model_dir)}, "small"
            )
            media = root / "clip.wav"
            media.write_bytes(b"RIFF fixture")

            def failing(command: list[str], **_: object) -> object:
                return extractor.subprocess.CompletedProcess(command, 7, "", "backend exploded")

            with self.assertRaisesRegex(extractor.SubtitleError, "whisper.cpp failed: backend exploded"):
                with mock.patch.object(extractor.subprocess, "run", side_effect=failing):
                    extractor.transcribe_with_whispercpp(
                        media, root / "failure", "clip", "small", "Chinese", runtime
                    )

            def empty(command: list[str], **_: object) -> object:
                output_base = Path(command[command.index("--output-file") + 1])
                output_base.with_suffix(".json").write_text('{"transcription": []}', encoding="utf-8")
                return extractor.subprocess.CompletedProcess(command, 0, "", "")

            with self.assertRaisesRegex(extractor.SubtitleError, "no subtitle segments"):
                with mock.patch.object(extractor.subprocess, "run", side_effect=empty):
                    extractor.transcribe_with_whispercpp(
                        media, root / "empty", "clip", "small", "Chinese", runtime
                    )

            def malformed(command: list[str], **_: object) -> object:
                output_base = Path(command[command.index("--output-file") + 1])
                output_base.with_suffix(".json").write_text("not json", encoding="utf-8")
                return extractor.subprocess.CompletedProcess(command, 0, "", "")

            with self.assertRaisesRegex(extractor.SubtitleError, "JSON output could not be read"):
                with mock.patch.object(extractor.subprocess, "run", side_effect=malformed):
                    extractor.transcribe_with_whispercpp(
                        media, root / "malformed", "clip", "small", "Chinese", runtime
                    )

    def test_ffmpeg_conversion_failure_is_reported_without_running_cpp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli, model_dir, _ = make_runtime_tree(tmp)
            runtime = extractor.validate_whispercpp_config(
                {"whisper_cli": str(cli), "model_dir": str(model_dir)}, "small"
            )
            media = root / "clip.mp4"
            media.write_bytes(b"video fixture")
            calls: list[list[str]] = []

            def failing(command: list[str], **_: object) -> object:
                calls.append(command)
                return extractor.subprocess.CompletedProcess(command, 1, "", "decode failed")

            with self.assertRaisesRegex(extractor.SubtitleError, "ffmpeg audio conversion failed: decode failed"):
                with mock.patch.object(extractor, "ensure_command"), mock.patch.object(
                    extractor.subprocess, "run", side_effect=failing
                ):
                    extractor.transcribe_with_whispercpp(
                        media, root / "conversion", "clip", "small", "Chinese", runtime
                    )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "ffmpeg")


class CliOptionContractTests(unittest.TestCase):
    def test_cli_exposes_backend_experiment_options_and_keeps_legacy_defaults(self) -> None:
        argv = [
            "extract_bilibili_subtitles.py",
            "input.wav",
            "--backend",
            "whispercpp",
            "--backend-config",
            "/tmp/runtime.json",
            "--threads",
            "3",
            "--force-transcribe",
            "--metrics-json",
            "/tmp/metrics.json",
            "--whisper-model",
            "small",
            "--language",
            "Chinese",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = extractor.parse_args()

        self.assertEqual(args.backend, "whispercpp")
        self.assertEqual(args.backend_config, "/tmp/runtime.json")
        self.assertEqual(args.threads, 3)
        self.assertTrue(args.force_transcribe)
        self.assertEqual(args.metrics_json, "/tmp/metrics.json")
        self.assertEqual(args.whisper_model, "small")
        self.assertEqual(args.language, "Chinese")


class BackendSelectionIntegrationTests(unittest.TestCase):
    def test_auto_cpp_execution_error_falls_back_and_preserves_failed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli, model_dir, _ = make_runtime_tree(tmp, coreml=True)
            config = {
                "whisper_cli": str(cli),
                "model_dir": str(model_dir),
                "require_coreml": True,
                "runtime_identity": "fixture-coreml",
            }
            expected = extractor.TranscriptBundle(
                title="clip",
                bvid=None,
                source_label="Whisper small 转写",
                segments=[{"start": 0.0, "end": 1.0, "text": "fallback after crash"}],
            )
            metrics = extractor.MetricsRecorder("auto", "small", "Chinese")
            def cpp_failure(command: list[str], **_: object) -> object:
                return extractor.subprocess.CompletedProcess(command, 7, "", "cpp crashed")

            with mock.patch.object(extractor, "is_apple_silicon", return_value=True), mock.patch.object(
                extractor.subprocess, "run", side_effect=cpp_failure
            ), mock.patch.object(
                extractor, "transcribe_with_openai", return_value=expected
            ) as openai:
                result = extractor.transcribe_media(
                    root / "clip.wav",
                    root / "run",
                    "clip",
                    "small",
                    "Chinese",
                    backend="auto",
                    backend_config=config,
                    threads=4,
                    metrics=metrics,
                )

        self.assertIs(result, expected)
        self.assertTrue(openai.called)
        self.assertEqual(openai.call_args.args[-1], 4)
        self.assertEqual(metrics.data["resolved_backend"], "openai")
        self.assertIn("runtime execution failed", metrics.data["fallback_reason"])
        self.assertIn("cpp crashed", metrics.data["runtime_execution_error"])
        self.assertEqual(metrics.data["runtime_identity"], "fixture-coreml")
        self.assertEqual(metrics.data["stage_attempts"]["transcription"][-1]["status"], "error")

    def test_explicit_cpp_execution_error_is_strict_and_does_not_call_openai(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli, model_dir, _ = make_runtime_tree(tmp)
            config = {"whisper_cli": str(cli), "model_dir": str(model_dir)}
            metrics = extractor.MetricsRecorder("whispercpp", "small", "Chinese")
            with mock.patch.object(
                extractor,
                "transcribe_with_whispercpp",
                side_effect=extractor.SubtitleError("cpp crashed"),
            ), mock.patch.object(
                extractor, "transcribe_with_openai", side_effect=AssertionError("explicit cpp must not fallback")
            ):
                with self.assertRaisesRegex(extractor.SubtitleError, "cpp crashed"):
                    extractor.transcribe_media(
                        root / "clip.wav",
                        root / "run",
                        "clip",
                        "small",
                        "Chinese",
                        backend="whispercpp",
                        backend_config=config,
                        metrics=metrics,
                    )

        self.assertEqual(metrics.data["resolved_backend"], "whispercpp")
        self.assertNotIn("runtime_execution_error", metrics.data)

    def test_openai_backend_receives_threads_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = extractor.TranscriptBundle(
                title="clip",
                bvid=None,
                source_label="Whisper small 转写",
                segments=[{"start": 0.0, "end": 1.0, "text": "openai"}],
            )
            with mock.patch.object(
                extractor, "transcribe_with_openai", return_value=expected
            ) as openai:
                result = extractor.transcribe_media(
                    root / "clip.wav",
                    root / "run",
                    "clip",
                    "small",
                    "Chinese",
                    backend="openai",
                    threads=7,
                )

        self.assertIs(result, expected)
        self.assertEqual(openai.call_args.args[-1], 7)

    def test_main_explicit_cpp_failure_does_not_silently_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "clip.wav"
            media.write_bytes(b"RIFF fixture")
            output_dir = root / "out"
            metrics_path = root / "metrics.json"
            argv = [
                "extract_bilibili_subtitles.py",
                str(media),
                "--output-dir",
                str(output_dir),
                "--backend",
                "whispercpp",
                "--backend-config",
                str(root / "missing.json"),
                "--metrics-json",
                str(metrics_path),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                extractor, "transcribe_with_openai", side_effect=AssertionError("explicit cpp must not fallback")
            ):
                result = extractor.main()

            receipt = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 1)
        self.assertEqual(receipt["status"], "error")
        self.assertEqual(receipt["resolved_backend"], None)
        self.assertIn("Backend config not found", receipt["error"])
        self.assertFalse(list(output_dir.glob("*_字幕.md")))

    def test_main_auto_uses_openai_fallback_when_cpp_config_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "clip.wav"
            media.write_bytes(b"RIFF fixture")
            output_dir = root / "out"
            metrics_path = root / "metrics.json"
            expected = extractor.TranscriptBundle(
                title="clip",
                bvid=None,
                source_label="Whisper small 转写",
                segments=[{"start": 0.0, "end": 1.0, "text": "fallback"}],
            )
            argv = [
                "extract_bilibili_subtitles.py",
                str(media),
                "--output-dir",
                str(output_dir),
                "--backend",
                "auto",
                "--backend-config",
                str(root / "missing.json"),
                "--metrics-json",
                str(metrics_path),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                extractor, "transcribe_with_openai", return_value=expected
            ) as openai:
                result = extractor.main()
                self.assertTrue(openai.called)

            receipt = json.loads(metrics_path.read_text(encoding="utf-8"))
            transcript = next(output_dir.glob("*_字幕.md"))
            transcript_text = transcript.read_text(encoding="utf-8")

        self.assertEqual(result, 0)
        self.assertEqual(receipt["status"], "success")
        self.assertEqual(receipt["resolved_backend"], "openai")
        self.assertIn("unavailable", receipt["fallback_reason"])
        self.assertIn("fallback", transcript_text)

    def test_main_cpp_success_writes_transcript_and_metrics_with_cli_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli, model_dir, _ = make_runtime_tree(tmp)
            config_path = root / "runtime.json"
            config_path.write_text(
                json.dumps(
                    {
                        "whisper_cli": str(cli),
                        "model_dir": str(model_dir),
                        "runtime_identity": {"fixture": True},
                    }
                ),
                encoding="utf-8",
            )
            media = root / "clip.wav"
            media.write_bytes(b"RIFF fixture")
            output_dir = root / "out"
            metrics_path = root / "metrics.json"
            calls: list[list[str]] = []

            def fake_run(command: list[str], **_: object) -> object:
                calls.append(command)
                output_base = Path(command[command.index("--output-file") + 1])
                output_base.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "transcription": [
                                {"offsets": {"from": 500, "to": 1750}, "text": "cpp main"}
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                return extractor.subprocess.CompletedProcess(command, 0, "", "")

            argv = [
                "extract_bilibili_subtitles.py",
                str(media),
                "--output-dir",
                str(output_dir),
                "--backend",
                "whispercpp",
                "--backend-config",
                str(config_path),
                "--threads",
                "2",
                "--metrics-json",
                str(metrics_path),
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                extractor.subprocess, "run", side_effect=fake_run
            ):
                result = extractor.main()

            receipt = json.loads(metrics_path.read_text(encoding="utf-8"))
            transcript = next(output_dir.glob("*_字幕.md"))
            transcript_text = transcript.read_text(encoding="utf-8")

        self.assertEqual(result, 0)
        self.assertEqual(receipt["resolved_backend"], "whispercpp")
        self.assertEqual(receipt["status"], "success")
        self.assertEqual(receipt["commands_sanitized"], True)
        self.assertEqual(receipt["commands"][0][receipt["commands"][0].index("--threads") + 1], "2")
        self.assertEqual(len(calls), 1)
        self.assertIn("cpp main", transcript_text)

    def test_force_transcribe_skips_official_subtitle_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            expected = extractor.TranscriptBundle(
                title="video",
                bvid=None,
                source_label="Whisper small 转写",
                segments=[{"start": 0.0, "end": 1.0, "text": "forced"}],
            )
            with mock.patch.object(
                extractor, "fetch_video_metadata", return_value=("video", 123)
            ), mock.patch.object(
                extractor, "fetch_official_subtitle", side_effect=AssertionError("force must skip official lookup")
            ), mock.patch.object(
                extractor, "download_audio", return_value=Path(tmp) / "clip.wav"
            ), mock.patch.object(
                extractor, "transcribe_media", return_value=expected
            ):
                result = extractor.process_bilibili_input(
                    "bvid",
                    "BV1iYwRzKED5",
                    Path(tmp),
                    None,
                    "small",
                    "Chinese",
                    backend="openai",
                    force_transcribe=True,
                )

        self.assertIs(result, expected)
        self.assertEqual(result.bvid, "BV1iYwRzKED5")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
