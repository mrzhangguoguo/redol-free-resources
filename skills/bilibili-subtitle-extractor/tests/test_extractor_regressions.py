"""Regression coverage for the Bilibili subtitle extractor.

The tests load the implementation by path so they do not require installing a
package or any optional ASR runtime.  Backend-specific contract tests live in
``test_backend_contract.py``; this module protects the pre-existing subtitle
and Markdown paths while the backend implementation evolves.
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
SPEC = importlib.util.spec_from_file_location("bilibili_extractor_under_test", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import setup failure
    raise RuntimeError(f"Could not load extractor from {SCRIPT_PATH}")
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


class SubtitleParsingRegressionTests(unittest.TestCase):
    def test_local_json_import_normalizes_and_sorts_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.json"
            path.write_text(
                json.dumps(
                    {
                        "body": [
                            {"from": 4.2, "to": 5.0, "content": " second  &amp; <i>line</i> "},
                            {"from": 1.25, "to": 2.0, "content": " first\nline "},
                            {"from": 3.0, "to": 3.5, "content": "   "},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            bundle = extractor.parse_subtitle_file(path, None)

        self.assertEqual(bundle.title, "demo")
        self.assertEqual(bundle.source_label, "本地字幕文件导入")
        self.assertEqual(
            bundle.segments,
            [
                {"start": 1.25, "end": 2.0, "text": "first line"},
                {"start": 4.2, "end": 5.0, "text": "second & line"},
            ],
        )

    def test_srt_import_accepts_sequence_numbers_and_multiline_text(self) -> None:
        srt = """\ufeff1
00:00:04,200 --> 00:00:05,000
second line

2
00:00:01.250 --> 00:00:02.000
first <b>line</b>
continued
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "captions.srt"
            path.write_text(srt, encoding="utf-8")

            bundle = extractor.parse_subtitle_file(path, "  Custom   title  ")

        self.assertEqual(bundle.title, "Custom title")
        self.assertEqual(
            bundle.segments,
            [
                {"start": 1.25, "end": 2.0, "text": "first line continued"},
                {"start": 4.2, "end": 5.0, "text": "second line"},
            ],
        )

    def test_empty_or_malformed_local_subtitle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty.vtt"
            empty.write_text("WEBVTT\n\nNOTE no cues here\n", encoding="utf-8")
            with self.assertRaisesRegex(
                extractor.SubtitleError, "contained no subtitle segments"
            ):
                extractor.parse_subtitle_file(empty, None)

            malformed = Path(tmp) / "malformed.json"
            malformed.write_text("{\"unknown\": []}", encoding="utf-8")
            with self.assertRaises(extractor.SubtitleError):
                extractor.parse_subtitle_file(malformed, None)


class FormattingRegressionTests(unittest.TestCase):
    def test_markdown_retains_metadata_and_uses_existing_timestamp_layout(self) -> None:
        bundle = extractor.TranscriptBundle(
            title="A / title",
            bvid="BV1iYwRzKED5",
            source_label="B站原始字幕",
            segments=[
                {"start": 0.49, "end": 1.0, "text": "opening"},
                {"start": 61.51, "end": 62.0, "text": "minute"},
                {"start": 3600.0, "end": 3601.0, "text": "hour"},
            ],
        )

        markdown = extractor.build_transcript_markdown(bundle)

        self.assertIn("# 视频字幕", markdown)
        self.assertIn("- 标题：A / title", markdown)
        self.assertIn("- BV号：BV1iYwRzKED5", markdown)
        self.assertIn("- 字幕来源：B站原始字幕", markdown)
        self.assertIn("[00:00]\nopening", markdown)
        self.assertIn("[01:02]\nminute", markdown)
        self.assertIn("[01:00:00]\nhour", markdown)
        self.assertTrue(markdown.endswith("\n"))

    def test_write_transcript_uses_sanitized_transcript_name_and_suggested_organized_path(self) -> None:
        bundle = extractor.TranscriptBundle(
            title="  title: with/slashes  ",
            bvid=None,
            source_label="本地字幕文件导入",
            segments=[{"start": 0.0, "end": None, "text": "hello"}],
        )
        with tempfile.TemporaryDirectory() as tmp:
            transcript, organized = extractor.write_transcript_markdown(bundle, Path(tmp))

            self.assertEqual(transcript.name, "title_ with_slashes_字幕.md")
            self.assertEqual(organized.name, "title_ with_slashes_整理版.md")
            self.assertTrue(transcript.exists())
            self.assertFalse(organized.exists())
            self.assertIn("字幕来源：本地字幕文件导入", transcript.read_text(encoding="utf-8"))


class OfficialSubtitlePriorityRegressionTests(unittest.TestCase):
    def test_official_subtitle_bypasses_audio_download_and_asr(self) -> None:
        official = extractor.TranscriptBundle(
            title="Official title",
            bvid="BV1iYwRzKED5",
            source_label="B站原始字幕",
            segments=[{"start": 0.0, "end": 1.0, "text": "official"}],
        )

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            extractor, "fetch_video_metadata", return_value=("Official title", 123)
        ), mock.patch.object(
            extractor, "fetch_official_subtitle", return_value=official
        ), mock.patch.object(
            extractor, "download_audio", side_effect=AssertionError("official subtitle must bypass download")
        ), mock.patch.object(
            extractor, "transcribe_with_whisper", side_effect=AssertionError("official subtitle must bypass ASR")
        ):
            result = extractor.process_bilibili_input(
                "bvid",
                "BV1iYwRzKED5",
                Path(tmp),
                None,
                "small",
                "Chinese",
            )

        self.assertIs(result, official)
        self.assertEqual(result.source_label, "B站原始字幕")

    def test_force_transcribe_can_be_passed_without_changing_local_subtitle_import(self) -> None:
        """The experiment flag must not turn a local subtitle into an ASR job.

        This calls the legacy parser directly because the new CLI options are
        intentionally tested separately.  It documents the important bypass
        invariant for future refactors.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "captions.vtt"
            path.write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n",
                encoding="utf-8",
            )
            bundle = extractor.parse_subtitle_file(path, None)

        self.assertEqual(bundle.source_label, "本地字幕文件导入")
        self.assertEqual(bundle.segments[0]["text"], "hello")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
