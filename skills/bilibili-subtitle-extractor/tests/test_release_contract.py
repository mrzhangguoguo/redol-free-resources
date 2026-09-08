"""Release invariants found during integration review; no model or network work."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "extractor_release",
    Path(__file__).resolve().parents[1] / "scripts/extract_bilibili_subtitles.py",
)
extractor = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = extractor
spec.loader.exec_module(extractor)


class ReleaseContract(unittest.TestCase):
    def test_full_language_names_remain_usable_case_insensitively(self):
        self.assertEqual(extractor._whispercpp_language(" French "), "french")
        self.assertEqual(extractor._whispercpp_language("ZH"), "zh")
        self.assertEqual(extractor._whispercpp_language("Chinese"), "zh")

    def test_malformed_cpp_timestamps_fail_instead_of_partial_transcript(self):
        invalid = [
            None,
            {"from": 0},
            {"from": 0, "to": "bad"},
            {"from": 2, "to": 1},
            {"from": -1, "to": 5},
            {"from": 0, "to": float("nan")},
        ]
        for offsets in invalid:
            with (
                self.subTest(offsets=offsets),
                self.assertRaises(extractor.SubtitleError),
            ):
                extractor.parse_whispercpp_json(
                    {"transcription": [{"offsets": offsets, "text": "content"}]}
                )
        with self.assertRaises(extractor.SubtitleError):
            extractor.parse_whispercpp_json({"transcription": ["not an object"]})

    def test_generic_local_json_still_allows_missing_end(self):
        self.assertEqual(
            extractor.parse_json_subtitle(
                {"segments": [{"start": 1, "text": "hello"}]}
            )[0]["end"],
            None,
        )

    def test_backend_failure_preserves_native_logs_in_temp_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            media = root / "sample.wav"
            media.write_bytes(b"fixture")
            runtime = extractor.WhisperCppRuntime(
                root / "cli", root, root / "model.bin", None, False, {}
            )
            result = subprocess.CompletedProcess(
                ["cli"], 1, stdout="native stdout", stderr="native failure"
            )
            with (
                mock.patch.object(extractor.subprocess, "run", return_value=result),
                self.assertRaises(extractor.SubtitleError),
            ):
                extractor.transcribe_with_whispercpp(
                    media, root, "title", "small", "Chinese", runtime
                )
            self.assertEqual(
                (root / "whispercpp" / "stdout.log").read_text(), "native stdout"
            )
            self.assertEqual(
                (root / "whispercpp" / "stderr.log").read_text(), "native failure"
            )


if __name__ == "__main__":
    unittest.main()
