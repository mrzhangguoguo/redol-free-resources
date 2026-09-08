"""Focused tests for the optional whisper.cpp setup helper."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "setup_whispercpp.py"
SPEC = importlib.util.spec_from_file_location("setup_whispercpp_under_test", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Could not load setup helper from {SCRIPT_PATH}")
setup_helper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = setup_helper
SPEC.loader.exec_module(setup_helper)


class CoreMLCheckpointBindingTests(unittest.TestCase):
    def test_runpy_wrapper_binds_converter_to_explicit_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_package = root / "fake-package"
            fake_package.mkdir()
            marker = root / "load-model-argument.txt"
            (fake_package / "whisper.py").write_text(
                """
import os

def load_model(name, *args, **kwargs):
    with open(os.environ["WHISPER_LOAD_MARKER"], "w", encoding="utf-8") as handle:
        handle.write(str(name))
    return object()
""".strip()
                + "\n",
                encoding="utf-8",
            )
            converter = root / "fake-converter.py"
            converter.write_text(
                """
import sys
from whisper import load_model

assert sys.argv[1:] == ["--model", "small"]
load_model("small")
""".strip()
                + "\n",
                encoding="utf-8",
            )
            checkpoint = root / "explicit-small.pt"
            checkpoint.write_bytes(b"checkpoint")
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(fake_package)
            environment["WHISPER_LOAD_MARKER"] = str(marker)

            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    setup_helper.coreml_converter_wrapper_source(),
                    str(converter),
                    "small",
                    str(checkpoint),
                    "--model",
                    "small",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertEqual(marker.read_text(encoding="utf-8"), str(checkpoint.resolve()))


class VenvEntrypointTests(unittest.TestCase):
    def test_venv_symlink_is_preserved_in_uv_pip_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv = root / ".venv"
            python_entrypoint = venv / "bin" / "python"
            python_entrypoint.parent.mkdir(parents=True)
            base_interpreter = root / "base-python"
            base_interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
            base_interpreter.chmod(0o755)
            python_entrypoint.symlink_to(base_interpreter)
            site_packages = root / "site-packages"
            site_packages.mkdir()

            class RecordingRunner:
                def __init__(self) -> None:
                    self.calls: list[list[str]] = []

                def run(self, argv, **kwargs):
                    command = [str(item) for item in argv]
                    self.calls.append(command)
                    if command[:3] == ["uv", "pip", "install"]:
                        return subprocess.CompletedProcess(command, 0, "", "")
                    if command[0] == str(python_entrypoint) and command[1] == "--version":
                        return subprocess.CompletedProcess(command, 0, "Python 3.12.0\n", "")
                    if command[0] == str(python_entrypoint) and command[1] == "-c":
                        if "getsitepackages" in command[2]:
                            output = str(site_packages) + "\n"
                        else:
                            output = json.dumps(setup_helper.EXPECTED_PACKAGE_VERSIONS) + "\n"
                        return subprocess.CompletedProcess(command, 0, output, "")
                    raise AssertionError(f"unexpected command: {command}")

            runner = RecordingRunner()
            installer = object.__new__(setup_helper.Installer)
            installer.venv = venv
            installer.command_runner = runner

            selected = setup_helper._find_venv_python(venv)
            self.assertEqual(selected, python_entrypoint.absolute())
            self.assertTrue(selected.is_symlink())

            installer._prepare_venv()
            pip_commands = [
                command
                for command in runner.calls
                if command[:3] == ["uv", "pip", "install"]
            ]
            self.assertEqual(len(pip_commands), 2)
            for command in pip_commands:
                python_index = command.index("--python")
                self.assertEqual(command[python_index + 1], str(python_entrypoint.absolute()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
