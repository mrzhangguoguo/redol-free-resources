#!/usr/bin/env python3
"""Install a pinned, isolated whisper.cpp + Core ML runtime.

This is deliberately an opt-in setup tool.  The normal subtitle extractor does
not invoke it.  The accelerated ``whispercpp`` path consumes already-installed
artifacts and never downloads a binary, model, or Core ML bundle during
transcription; the preserved OpenAI fallback may download a missing ``.pt``
checkpoint on first use.  All setup packages are installed into a uv-managed
Python 3.12 virtual environment below the runtime root; the user's system
Python is not modified.

The setup is resumable.  Every stage records its input fingerprint, commands,
and output hashes in ``<runtime>/receipts``.  A stage is skipped only when its
receipt and every recorded output still match.  Existing source/config files
are left intact unless the caller explicitly asked this setup to update the
runtime; generated replacements keep a recoverable ``.previous-*`` directory
when a compiled Core ML bundle is replaced.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _datetime
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


REPO_URL = "https://github.com/ggml-org/whisper.cpp.git"
REPO_TAG = "v1.8.7"
REPO_COMMIT = "48f628a84833905ee4a0658ee6d4a5c915ce1997"
DEFAULT_RUNTIME_DIR = Path("~/.cache/bilibili-subtitle-extractor/whispercpp-v1.8.7")
DEFAULT_MODEL = "small"
DEFAULT_CONFIG_NAME = "runtime.json"
COREML_SOURCE_RELATIVE = Path("src/coreml/whisper-encoder.mm")
COREML_SOURCE_OLD = "config.computeUnits = MLComputeUnitsAll;"
COREML_SOURCE_NEW = "config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;"
COREML_COMPUTE_UNITS = "CPUAndNeuralEngine"
RECEIPT_VERSION = 1
MAX_OUTPUT_TAIL = 20_000

PYTHON_PACKAGES: tuple[str, ...] = (
    "torch==2.5.1",
    "coremltools==8.3.0",
    "openai-whisper==20250625",
    "numpy==1.26.4",
)
ANE_PACKAGE = "ane_transformers==0.1.3"
EXPECTED_PACKAGE_VERSIONS: dict[str, str] = {
    "torch": "2.5.1",
    "coremltools": "8.3.0",
    "openai-whisper": "20250625",
    "numpy": "1.26.4",
    "ane_transformers": "0.1.3",
}


def coreml_converter_wrapper_source() -> str:
    """Return the small runpy wrapper that binds conversion to one .pt file.

    The upstream converter calls ``whisper.load_model(args.model)``.  Patching
    that function before ``runpy`` executes the unchanged converter makes an
    explicit ``--pt-model`` authoritative without copying it into the Whisper
    cache or allowing a symbolic model name to download another checkpoint.
    """

    return r'''
import runpy
import sys
from pathlib import Path

import whisper

converter = Path(sys.argv[1]).resolve()
model_name = sys.argv[2]
checkpoint = Path(sys.argv[3]).resolve()
converter_args = sys.argv[4:]
original_load_model = whisper.load_model


def load_model(name, *args, **kwargs):
    if name == model_name:
        return original_load_model(str(checkpoint), *args, **kwargs)
    return original_load_model(name, *args, **kwargs)


whisper.load_model = load_model
sys.argv = [str(converter), *converter_args]
runpy.run_path(str(converter), run_name="__main__")
'''.strip()


class SetupError(RuntimeError):
    """A setup or doctor failure with a user-actionable message."""


def _now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def coreml_converter_wrapper_sha256() -> str:
    return sha256_bytes(coreml_converter_wrapper_source().encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SetupError(f"Could not hash {path}: {exc}") from exc
    return digest.hexdigest()


def _relative_is_ignored(relative: Path, ignored: Sequence[Path]) -> bool:
    return any(relative == item or item in relative.parents for item in ignored)


def tree_sha256(path: Path, *, ignored: Sequence[Path] = ()) -> str:
    """Hash a directory deterministically, including relative names and modes.

    ``.git`` and build output are excluded by callers when they are not part of
    the reproducible source identity.  Symlink targets are recorded without
    following them.
    """

    path = path.resolve()
    if not path.is_dir():
        raise SetupError(f"Expected a directory while hashing: {path}")
    digest = hashlib.sha256()
    ignored = tuple(Path(item) for item in ignored)
    for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        relative = candidate.relative_to(path)
        if _relative_is_ignored(relative, ignored):
            continue
        try:
            mode = stat.S_IFMT(candidate.lstat().st_mode)
        except OSError as exc:
            raise SetupError(f"Could not inspect {candidate}: {exc}") from exc
        digest.update(f"{relative.as_posix()}\0{mode:o}\0".encode("utf-8"))
        if candidate.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(candidate).encode("utf-8"))
        elif candidate.is_file():
            digest.update(b"file\0")
            digest.update(sha256_file(candidate).encode("ascii"))
        elif candidate.is_dir():
            digest.update(b"dir\0")
        else:
            digest.update(b"other\0")
    return digest.hexdigest()


def path_snapshot(path: Path) -> dict[str, Any]:
    """Return a stable receipt entry for a file, directory, or symlink."""

    path = path.expanduser().resolve()
    if not path.exists() and not path.is_symlink():
        raise SetupError(f"Expected setup output does not exist: {path}")
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise SetupError(f"Could not inspect setup output {path}: {exc}") from exc
    entry: dict[str, Any] = {
        "path": str(path),
        "mode": stat.S_IMODE(mode),
        "kind": "directory" if path.is_dir() else "file",
    }
    if path.is_symlink():
        entry["kind"] = "symlink"
        entry["target"] = os.readlink(path)
    elif path.is_file():
        entry["size"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
    elif path.is_dir():
        entry["tree_sha256"] = tree_sha256(path)
    else:
        entry["kind"] = "other"
    return entry


def snapshots_match(recorded: dict[str, Any]) -> bool:
    try:
        current = path_snapshot(Path(str(recorded["path"])))
    except (KeyError, SetupError, ValueError):
        return False
    # Do not compare transient size/mode fields beyond what the recorded hash
    # already establishes; executable mode is nevertheless useful for binary
    # readiness and is kept in the receipt.
    for key in ("path", "kind", "target", "sha256", "tree_sha256"):
        if recorded.get(key) != current.get(key):
            return False
    return True


def atomic_write_bytes(path: Path, payload: bytes, *, mode: int | None = None) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode: int | None = None
    if mode is None and path.exists():
        try:
            existing_mode = stat.S_IMODE(path.stat().st_mode)
        except OSError:
            existing_mode = None
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode if mode is not None else (existing_mode or 0o644))
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any, *, mode: int | None = None) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n").encode(
            "utf-8"
        ),
        mode=mode,
    )


def replace_generated_file(source: Path, destination: Path) -> None:
    """Replace a generated file while keeping an old copy recoverable."""

    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    backup: Path | None = None
    if destination.exists():
        backup = destination.with_name(
            destination.name + ".previous-" + uuid.uuid4().hex[:10]
        )
        destination.rename(backup)
    try:
        os.replace(source, destination)
    except BaseException:
        if backup is not None and not destination.exists():
            backup.rename(destination)
        raise


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SetupError(f"Could not read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SetupError(f"JSON file must contain an object: {path}")
    return payload


def resolve_runtime_dir(value: str | Path | None) -> Path:
    selected = DEFAULT_RUNTIME_DIR if value is None else Path(value)
    return selected.expanduser().resolve()


def default_config_path() -> Path:
    # The config belongs beside the installed skill, where the extractor looks
    # for it by default.  It is intentionally independent of the cache root.
    return (Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_NAME).resolve()


def resolve_config_path(value: str | Path | None) -> Path:
    return (default_config_path() if value is None else Path(value).expanduser()).resolve()


def command_text(argv: Sequence[str]) -> str:
    return shlex.join(str(item) for item in argv)


def redact_command(argv: Sequence[str]) -> list[str]:
    """Keep command receipts safe if a future setup step gains secrets."""

    redacted: list[str] = []
    redact_next = False
    for item in argv:
        value = str(item)
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if value in {"--cookies", "--password", "--token", "--access-token"}:
            redacted.append(value)
            redact_next = True
            continue
        redacted.append(value)
    return redacted


def ensure_command(name: str) -> str:
    selected = shutil.which(name)
    if not selected:
        raise SetupError(f"Required command not found: {name}")
    return selected


def _tail(value: str) -> str:
    if len(value) <= MAX_OUTPUT_TAIL:
        return value
    return "…" + value[-MAX_OUTPUT_TAIL:]


class CommandRunner:
    def __init__(self) -> None:
        self.commands: list[dict[str, Any]] = []

    def run(
        self,
        argv: Sequence[str | Path],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = [str(item) for item in argv]
        cwd_path = str(cwd.resolve()) if cwd is not None else None
        safe = redact_command(command)
        print(f"$ {command_text(safe)}", flush=True)
        result = subprocess.run(
            command,
            cwd=cwd_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        entry = {
            "argv": safe,
            "cwd": cwd_path,
            "returncode": result.returncode,
            "stdout_tail": _tail(result.stdout or ""),
            "stderr_tail": _tail(result.stderr or ""),
        }
        self.commands.append(entry)
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            if len(detail) > 4_000:
                detail = detail[-4_000:]
            raise SetupError(
                f"Command failed ({result.returncode}): {command_text(safe)}"
                + (f"\n{detail}" if detail else "")
            )
        return result


def _git_commit(runner: CommandRunner, repo: Path) -> str:
    result = runner.run(["git", "-C", repo, "rev-parse", "HEAD"])
    return result.stdout.strip()


def _expected_patched_source(runner: CommandRunner, repo: Path) -> str:
    result = runner.run(
        [
            "git",
            "-C",
            repo,
            "show",
            f"{REPO_COMMIT}:{COREML_SOURCE_RELATIVE.as_posix()}",
        ]
    )
    baseline = result.stdout
    if baseline.count(COREML_SOURCE_OLD) != 1:
        raise SetupError(
            "Pinned Core ML source does not contain exactly one expected "
            f"{COREML_SOURCE_OLD!r} line"
        )
    return baseline.replace(COREML_SOURCE_OLD, COREML_SOURCE_NEW)


def _status_paths(status_output: str) -> list[tuple[str, str]]:
    """Parse porcelain status into (XY, path), retaining only tracked rows."""

    paths: list[tuple[str, str]] = []
    for line in status_output.splitlines():
        if len(line) < 3:
            continue
        code = line[:2]
        raw_path = line[3:]
        if code in {"??", "!!"}:
            continue
        # A rename has two paths. Neither is an allowed source-only edit.
        if " -> " in raw_path:
            paths.append((code, raw_path))
        else:
            paths.append((code, raw_path))
    return paths


def _find_venv_python(venv: Path) -> Path:
    candidates = [venv / "bin" / "python", venv / "bin" / "python3"]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            # Keep the venv entrypoint itself. Resolving this symlink points at
            # uv's shared base interpreter and would make ``uv pip --python``
            # install outside the isolated environment.
            return candidate.absolute()
    raise SetupError(f"uv virtual environment has no executable Python: {venv}")


def _find_whisper_cli(build_dir: Path) -> Path:
    candidates = [
        build_dir / "bin" / "whisper-cli",
        build_dir / "Release" / "bin" / "whisper-cli",
        build_dir / "whisper-cli",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    for candidate in sorted(build_dir.rglob("whisper-cli")):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise SetupError(f"cmake completed but whisper-cli was not found under {build_dir}")


def _find_build_dylibs(build_dir: Path) -> list[Path]:
    return sorted(
        {
            candidate.resolve()
            for candidate in build_dir.rglob("*.dylib")
            if candidate.is_file()
        },
        key=lambda item: str(item),
    )


def _site_packages(runner: CommandRunner, python: Path) -> Path:
    result = runner.run(
        [
            python,
            "-c",
            "import site; print(site.getsitepackages()[0])",
        ]
    )
    selected = Path(result.stdout.strip()).expanduser().resolve()
    if not selected.is_dir():
        raise SetupError(f"Python reported a missing site-packages directory: {selected}")
    return selected


def _installed_versions(runner: CommandRunner, python: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata as m, json; "
        "names=['torch','coremltools','openai-whisper','numpy','ane_transformers']; "
        "print(json.dumps({n:m.version(n) for n in names}, sort_keys=True))"
    )
    result = runner.run([python, "-c", code])
    try:
        versions = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise SetupError(f"Could not parse installed Python package versions: {result.stdout}") from exc
    if not isinstance(versions, dict) or any(not isinstance(v, str) for v in versions.values()):
        raise SetupError(f"Unexpected package version response: {versions!r}")
    return {str(key): str(value) for key, value in versions.items()}


def _stage_fingerprint(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def _receipt_path(receipts_dir: Path, stage: str) -> Path:
    return receipts_dir / f"stage-{stage}.json"


def _load_receipt(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return read_json_object(path)
    except SetupError:
        return None


class StageRunner:
    def __init__(self, receipts_dir: Path, runner: CommandRunner, *, force: bool = False) -> None:
        self.receipts_dir = receipts_dir
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.force = force
        self.results: dict[str, dict[str, Any]] = {}

    def run(
        self,
        name: str,
        *,
        inputs: dict[str, Any],
        commands: list[dict[str, Any]],
        output_paths: Callable[[], list[Path]],
        action: Callable[[], dict[str, Any] | None],
    ) -> dict[str, Any]:
        fingerprint_payload = {"stage": name, "inputs": inputs, "commands": commands}
        fingerprint = _stage_fingerprint(fingerprint_payload)
        receipt_file = _receipt_path(self.receipts_dir, name)
        previous = _load_receipt(receipt_file)
        if not self.force and previous is not None and previous.get("fingerprint") == fingerprint:
            recorded_outputs = previous.get("outputs")
            if isinstance(recorded_outputs, list) and recorded_outputs and all(
                isinstance(item, dict) and snapshots_match(item) for item in recorded_outputs
            ):
                print(f"stage {name}: skipped (receipt and hashes match)")
                self.results[name] = previous
                return previous

        print(f"stage {name}: running")
        metadata = action() or {}
        outputs = output_paths()
        if not outputs:
            raise SetupError(f"Stage {name} produced no declared outputs")
        snapshots = [path_snapshot(path) for path in outputs]
        receipt = {
            "receipt_version": RECEIPT_VERSION,
            "stage": name,
            "status": "success",
            "created_at": _now(),
            "fingerprint": fingerprint,
            "inputs": inputs,
            "commands": commands,
            "outputs": snapshots,
            "metadata": metadata,
        }
        atomic_write_json(receipt_file, receipt)
        self.results[name] = receipt
        return receipt


@contextlib.contextmanager
def setup_lock(runtime_dir: Path) -> Iterator[None]:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_dir / ".setup.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise SetupError(
            f"Another setup appears to be running ({lock_path}); remove the lock only after checking."
        ) from exc
    try:
        os.write(fd, f"pid={os.getpid()} started={_now()}\n".encode("utf-8"))
        os.close(fd)
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


class Installer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.runtime_dir = resolve_runtime_dir(args.runtime_dir)
        self.repo = self.runtime_dir / "whisper.cpp"
        self.build_dir = self.repo / "build"
        self.venv = self.runtime_dir / ".venv"
        self.models_dir = self.runtime_dir / "models"
        self.receipts_dir = self.runtime_dir / "receipts"
        self.pt_model = (
            Path(args.pt_model).expanduser().resolve()
            if args.pt_model
            else (Path.home() / ".cache" / "whisper" / f"{args.model}.pt").resolve()
        )
        self.config_path = resolve_config_path(args.config)
        self.command_runner = CommandRunner()
        self.stages = StageRunner(self.receipts_dir, self.command_runner, force=args.force)
        self.venv_python: Path | None = None
        self.site_packages: Path | None = None
        self.whisper_cli: Path | None = None
        self.compiled_encoder = self.models_dir / f"ggml-{args.model}-encoder.mlmodelc"
        self.coreml_package = self.repo / "models" / f"coreml-encoder-{args.model}.mlpackage"

    @property
    def model(self) -> str:
        return str(self.args.model)

    def _check_platform(self) -> None:
        if sys.platform != "darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
            print(
                "warning: this runtime targets Apple Silicon Core ML; "
                f"current platform is {sys.platform}/{platform.machine()}",
                file=sys.stderr,
            )

    def _ensure_tools(self) -> None:
        for name in ("git", "cmake", "uv"):
            ensure_command(name)

    def _tool_versions(self) -> dict[str, Any]:
        """Capture tool/package versions in the final receipt."""

        versions: dict[str, Any] = {}
        for name in ("git", "cmake", "uv"):
            result = self.command_runner.run([name, "--version"])
            output = (result.stdout or result.stderr).strip().splitlines()
            versions[name] = output[0] if output else "unknown"
        if self.venv_python is not None:
            result = self.command_runner.run([self.venv_python, "--version"])
            output = (result.stdout or result.stderr).strip().splitlines()
            versions["python"] = output[0] if output else "unknown"
            versions["packages"] = _installed_versions(self.command_runner, self.venv_python)
        return versions

    def _clone_source(self) -> dict[str, Any]:
        if self.repo.exists():
            if not (self.repo / ".git").exists():
                raise SetupError(
                    f"Runtime source path exists but is not a git checkout: {self.repo}. "
                    "Move it aside and retry; setup will not delete it."
                )
            commit = _git_commit(self.command_runner, self.repo)
            if commit != REPO_COMMIT:
                raise SetupError(
                    f"Existing whisper.cpp checkout is {commit}, expected {REPO_COMMIT}; "
                    "setup will not replace it. Use a new --runtime-dir."
                )
            return {"commit": commit, "action": "reused"}

        self.repo.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.repo.parent / f".{self.repo.name}.clone-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        try:
            self.command_runner.run(
                [
                    "git",
                    "clone",
                    "--branch",
                    REPO_TAG,
                    "--single-branch",
                    REPO_URL,
                    temporary,
                ]
            )
            commit = _git_commit(self.command_runner, temporary)
            if commit != REPO_COMMIT:
                raise SetupError(
                    f"{REPO_TAG} resolved to {commit}, expected pinned commit {REPO_COMMIT}"
                )
            os.replace(temporary, self.repo)
            return {"commit": commit, "action": "cloned"}
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _patch_coreml_source(self) -> dict[str, Any]:
        source = self.repo / COREML_SOURCE_RELATIVE
        if not source.is_file():
            raise SetupError(f"Pinned source file is missing: {source}")
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise SetupError(f"Could not read Core ML source {source}: {exc}") from exc
        if COREML_SOURCE_OLD in text:
            if text.count(COREML_SOURCE_OLD) != 1:
                raise SetupError(
                    f"Expected exactly one active {COREML_SOURCE_OLD!r} in {source}"
                )
            updated = text.replace(COREML_SOURCE_OLD, COREML_SOURCE_NEW)
            mode = stat.S_IMODE(source.stat().st_mode)
            atomic_write_bytes(source, updated.encode("utf-8"), mode=mode)
            return {"action": "replaced", "old": COREML_SOURCE_OLD, "new": COREML_SOURCE_NEW}
        if COREML_SOURCE_NEW in text:
            return {"action": "already_replaced", "new": COREML_SOURCE_NEW}
        raise SetupError(
            f"Core ML source has neither expected compute-unit line in {source}; refusing an unsafe edit"
        )

    def _verify_source_state(self) -> dict[str, Any]:
        """Reject tracked edits other than the one required Core ML patch."""

        source = self.repo / COREML_SOURCE_RELATIVE
        expected = _expected_patched_source(self.command_runner, self.repo)
        try:
            current = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise SetupError(f"Could not read patched source {source}: {exc}") from exc
        if current != expected:
            raise SetupError(
                f"Tracked source differs from the pinned source plus the expected "
                f"{COREML_SOURCE_OLD} replacement: {source}"
            )
        status = self.command_runner.run(
            ["git", "-C", self.repo, "status", "--porcelain=v1", "--untracked-files=all"]
        )
        allowed = COREML_SOURCE_RELATIVE.as_posix()
        unexpected = [
            f"{code} {path}"
            for code, path in _status_paths(status.stdout)
            if path != allowed
        ]
        if unexpected:
            raise SetupError(
                "Unexpected tracked modifications in pinned source checkout; "
                "only the exact Core ML compute-unit replacement is allowed: "
                + ", ".join(unexpected)
            )
        return {
            "source": str(source),
            "expected_patch": {
                "from": COREML_SOURCE_OLD,
                "to": COREML_SOURCE_NEW,
            },
            "tracked_modifications": [
                {"status": code, "path": path}
                for code, path in _status_paths(status.stdout)
            ],
        }

    def _prepare_venv(self) -> dict[str, Any]:
        if self.venv.exists() and not any(self.venv.iterdir()):
            # An empty directory is safe to populate.
            pass
        elif self.venv.exists() and not any(
            (self.venv / "bin" / name).is_file() for name in ("python", "python3")
        ):
            raise SetupError(
                f"Virtual-environment path exists without Python: {self.venv}; "
                "move it aside or choose another --runtime-dir."
            )
        if not self.venv.exists() or not any(
            (self.venv / "bin" / name).is_file() for name in ("python", "python3")
        ):
            self.command_runner.run(["uv", "venv", "--python", "3.12", self.venv])
        self.venv_python = _find_venv_python(self.venv)
        self.command_runner.run(
            ["uv", "pip", "install", "--python", self.venv_python, *PYTHON_PACKAGES]
        )
        self.command_runner.run(
            ["uv", "pip", "install", "--python", self.venv_python, "--no-deps", ANE_PACKAGE]
        )
        versions = _installed_versions(self.command_runner, self.venv_python)
        if versions != EXPECTED_PACKAGE_VERSIONS:
            raise SetupError(
                "Installed Python package versions do not match the pinned recipe: "
                f"expected {EXPECTED_PACKAGE_VERSIONS}, got {versions}"
            )
        self.site_packages = _site_packages(self.command_runner, self.venv_python)
        manifest = self.venv / "setup-manifest.json"
        atomic_write_json(
            manifest,
            {
                "python": str(self.venv_python),
                "python_version": self.command_runner.run(
                    [self.venv_python, "--version"]
                ).stdout.strip(),
                "site_packages": str(self.site_packages),
                "packages": versions,
            },
        )
        return {"python": str(self.venv_python), "site_packages": str(self.site_packages), "packages": versions}

    def _load_model(self) -> dict[str, Any]:
        if self.pt_model.is_file():
            return {"action": "reused", "path": str(self.pt_model)}
        if self.args.pt_model:
            raise SetupError(
                f"Requested --pt-model does not exist: {self.pt_model}; setup will not download a custom path."
            )
        if self.venv_python is None:
            raise SetupError("Python environment is not ready before model setup")
        self.pt_model.parent.mkdir(parents=True, exist_ok=True)
        code = (
            "import sys, whisper; "
            "whisper.load_model(sys.argv[1], download_root=sys.argv[2]); "
            "print('model-ready')"
        )
        self.command_runner.run([self.venv_python, "-c", code, self.model, self.pt_model.parent])
        if not self.pt_model.is_file():
            raise SetupError(f"Whisper model loader completed without creating {self.pt_model}")
        return {"action": "loaded_or_reused", "path": str(self.pt_model)}

    def _convert_ggml(self) -> dict[str, Any]:
        if self.venv_python is None or self.site_packages is None:
            raise SetupError("Python environment is not ready before ggml conversion")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        intermediate = self.models_dir / "ggml-model.bin"
        final = self.models_dir / f"ggml-{self.model}.bin"
        converter = self.repo / "models" / "convert-pt-to-ggml.py"
        if not converter.is_file():
            raise SetupError(f"Pinned ggml conversion script is missing: {converter}")
        self.command_runner.run(
            [self.venv_python, converter, self.pt_model, self.site_packages, self.models_dir]
        )
        if not intermediate.is_file():
            raise SetupError(f"ggml conversion completed without creating {intermediate}")
        replace_generated_file(intermediate, final)
        return {"path": str(final), "sha256": sha256_file(final)}

    def _convert_coreml(self) -> dict[str, Any]:
        if self.venv_python is None:
            raise SetupError("Python environment is not ready before Core ML conversion")
        converter = self.repo / "models" / "convert-whisper-to-coreml.py"
        if not converter.is_file():
            raise SetupError(f"Pinned Core ML conversion script is missing: {converter}")
        self.repo.joinpath("models").mkdir(parents=True, exist_ok=True)
        quantize = "True" if self.args.coreml_quantize else "False"
        package_backup: Path | None = None
        if self.coreml_package.exists():
            package_backup = self.coreml_package.with_name(
                self.coreml_package.name + ".previous-" + uuid.uuid4().hex[:10]
            )
            self.coreml_package.rename(package_backup)
        try:
            self.command_runner.run(
                [
                    self.venv_python,
                    "-c",
                    coreml_converter_wrapper_source(),
                    converter,
                    self.model,
                    self.pt_model,
                    "--model",
                    self.model,
                    "--encoder-only",
                    "True",
                    "--optimize-ane",
                    "True",
                    "--quantize",
                    quantize,
                ],
                cwd=self.repo,
            )
        except BaseException:
            if package_backup is not None and not self.coreml_package.exists():
                package_backup.rename(self.coreml_package)
            raise
        if not self.coreml_package.is_dir():
            raise SetupError(f"Core ML conversion completed without creating {self.coreml_package}")
        return {
            "path": str(self.coreml_package),
            "tree_sha256": tree_sha256(self.coreml_package),
            "checkpoint": str(self.pt_model),
            "wrapper_sha256": coreml_converter_wrapper_sha256(),
            "quantize": self.args.coreml_quantize,
            "compute_precision": "FLOAT16" if self.args.coreml_quantize else "FLOAT32",
            "notes": (
                "quantize=True selects the upstream FLOAT16 path; it is not low-bit integer quantization. "
                "The default FLOAT16 path is used because the FP32 conversion did not produce useful ANE power evidence."
                if self.args.coreml_quantize
                else "FP32 was explicitly requested; Core ML configuration alone does not establish ANE execution."
            ),
        }

    def _compile_coreml(self) -> dict[str, Any]:
        if self.venv_python is None:
            raise SetupError("Python environment is not ready before Core ML compilation")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # Compile through coremltools rather than the optional xcrun/coremlc
        # wrapper.  The helper stages into a sibling directory and preserves a
        # previous compiled bundle if the destination already exists.
        helper = r'''
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
import coremltools as ct

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
destination.parent.mkdir(parents=True, exist_ok=True)
compiled = Path(ct.utils.compile_model(str(source))).resolve()
staging_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent)))
staged = staging_root / destination.name
backup = None
try:
    shutil.copytree(compiled, staged)
    if destination.exists():
        backup = destination.with_name(destination.name + ".previous-" + uuid.uuid4().hex[:10])
        destination.rename(backup)
    try:
        os.replace(staged, destination)
    except BaseException:
        if backup is not None and not destination.exists():
            backup.rename(destination)
        raise
finally:
    shutil.rmtree(staging_root, ignore_errors=True)
print(destination)
'''.strip()
        self.command_runner.run([self.venv_python, "-c", helper, self.coreml_package, self.compiled_encoder])
        if not self.compiled_encoder.is_dir():
            raise SetupError(f"Core ML compile completed without creating {self.compiled_encoder}")
        return {
            "path": str(self.compiled_encoder),
            "tree_sha256": tree_sha256(self.compiled_encoder),
            "compiler": "coremltools.utils.compile_model",
        }

    def _binary_path_from_receipt(self) -> Path | None:
        receipt = _load_receipt(_receipt_path(self.receipts_dir, "build"))
        if receipt:
            metadata = receipt.get("metadata")
            if isinstance(metadata, dict) and isinstance(metadata.get("binary_path"), str):
                return Path(metadata["binary_path"]).expanduser().resolve()
        return None

    def _build(self) -> dict[str, Any]:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        configure = [
            "cmake",
            "-S",
            self.repo,
            "-B",
            self.build_dir,
            "-DWHISPER_COREML=ON",
            "-DWHISPER_COREML_ALLOW_FALLBACK=OFF",
            "-DGGML_METAL=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        build = [
            "cmake",
            "--build",
            self.build_dir,
            "--config",
            "Release",
            "--target",
            "whisper-cli",
            "-j4",
        ]
        self.command_runner.run(configure)
        self.command_runner.run(build)
        self.whisper_cli = _find_whisper_cli(self.build_dir)
        linked_dylibs = _find_build_dylibs(self.build_dir)
        linked_dylib_snapshots = [path_snapshot(path) for path in linked_dylibs]
        marker = self.receipts_dir / "build-binary.json"
        atomic_write_json(
            marker,
            {
                "binary_path": str(self.whisper_cli),
                "sha256": sha256_file(self.whisper_cli),
                "source_commit": _git_commit(self.command_runner, self.repo),
                "linked_dylibs": linked_dylib_snapshots,
            },
        )
        return {
            "binary_path": str(self.whisper_cli),
            "sha256": sha256_file(self.whisper_cli),
            "linked_dylibs": linked_dylib_snapshots,
        }

    def _build_output_paths(self) -> list[Path]:
        marker = self.receipts_dir / "build-binary.json"
        if marker.is_file():
            try:
                payload = read_json_object(marker)
                binary = payload.get("binary_path")
                if isinstance(binary, str):
                    candidate = Path(binary).expanduser().resolve()
                    if candidate.is_file() and os.access(candidate, os.X_OK):
                        paths = [marker, candidate]
                        for item in payload.get("linked_dylibs", []):
                            if isinstance(item, dict) and isinstance(item.get("path"), str):
                                paths.append(Path(item["path"]).expanduser().resolve())
                        return paths
            except SetupError:
                pass
        return [marker] if marker.is_file() else []

    def _write_config(self, runtime_identity: dict[str, Any]) -> dict[str, Any]:
        if self.whisper_cli is None:
            raise SetupError("whisper-cli path is not available before config generation")
        model_path = self.models_dir / f"ggml-{self.model}.bin"
        if not model_path.is_file() or not self.compiled_encoder.is_dir():
            raise SetupError("Model outputs are not ready before config generation")
        existing: dict[str, Any] = {}
        if self.config_path.exists():
            existing = read_json_object(self.config_path)
        merged = dict(existing)
        merged.update(
            {
                "schema_version": 1,
                "backend": "whispercpp",
                "model": self.model,
                "whisper_cli": str(self.whisper_cli.resolve()),
                "model_dir": str(self.models_dir.resolve()),
                "coreml_encoder": str(self.compiled_encoder.resolve()),
                "coreml_enabled": True,
                "require_coreml": True,
            }
        )
        identity = dict(existing.get("runtime_identity")) if isinstance(existing.get("runtime_identity"), dict) else {}
        identity.update(runtime_identity)
        merged["runtime_identity"] = identity
        if self.config_path.exists():
            if self.config_path.is_symlink():
                raise SetupError(
                    f"Refusing to replace symlinked config path: {self.config_path}"
                )
            backup = self.config_path.with_name(
                self.config_path.name + ".previous-" + sha256_file(self.config_path)[:12]
            )
            if not backup.exists():
                shutil.copy2(self.config_path, backup)
        atomic_write_json(self.config_path, merged)
        return {"path": str(self.config_path), "sha256": sha256_file(self.config_path)}

    def install(self) -> int:
        self._check_platform()
        self._ensure_tools()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.stages.run(
            "source",
            inputs={"repo_url": REPO_URL, "tag": REPO_TAG, "commit": REPO_COMMIT},
            commands=[
                {
                    "argv": ["git", "clone", "--branch", REPO_TAG, "--single-branch", REPO_URL, str(self.repo)],
                    "cwd": str(self.runtime_dir),
                }
            ],
            output_paths=lambda: [self.repo / ".git" / "HEAD"],
            action=self._clone_source,
        )

        source_file = self.repo / COREML_SOURCE_RELATIVE
        self.stages.run(
            "patch-coreml",
            inputs={
                "source": path_snapshot(source_file) if source_file.exists() else {"path": str(source_file)},
                "replacement": COREML_SOURCE_NEW,
            },
            commands=[{"operation": "atomic source replacement", "path": str(source_file)}],
            output_paths=lambda: [source_file],
            action=self._patch_coreml_source,
        )

        venv_manifest = self.venv / "setup-manifest.json"
        venv_inputs = {"python": "3.12", "packages": [*PYTHON_PACKAGES, ANE_PACKAGE]}
        self.stages.run(
            "venv",
            inputs=venv_inputs,
            commands=[
                {"argv": ["uv", "venv", "--python", "3.12", str(self.venv)]},
                {"argv": ["uv", "pip", "install", "--python", str(self.venv / "bin" / "python"), *PYTHON_PACKAGES]},
                {"argv": ["uv", "pip", "install", "--python", str(self.venv / "bin" / "python"), "--no-deps", ANE_PACKAGE]},
            ],
            output_paths=lambda: [venv_manifest],
            action=self._prepare_venv,
        )
        if self.venv_python is None:
            self.venv_python = _find_venv_python(self.venv)
        if self.site_packages is None:
            manifest = read_json_object(venv_manifest)
            site = manifest.get("site_packages")
            if not isinstance(site, str):
                raise SetupError(f"venv receipt lacks site_packages: {venv_manifest}")
            self.site_packages = Path(site).expanduser().resolve()

        source_state = self._verify_source_state()
        self.stages.run(
            "build",
            inputs={
                "source_commit": REPO_COMMIT,
                "patched_source": path_snapshot(source_file),
                "expected_source_patch": source_state["expected_patch"],
                "receipt_outputs": "binary-and-linked-dylib-snapshots-v1",
                "flags": [
                    "-DWHISPER_COREML=ON",
                    "-DWHISPER_COREML_ALLOW_FALLBACK=OFF",
                    "-DGGML_METAL=ON",
                    "-DCMAKE_BUILD_TYPE=Release",
                ],
            },
            commands=[
                {"argv": ["cmake", "-S", str(self.repo), "-B", str(self.build_dir), "-DWHISPER_COREML=ON", "-DWHISPER_COREML_ALLOW_FALLBACK=OFF", "-DGGML_METAL=ON", "-DCMAKE_BUILD_TYPE=Release"]},
                {"argv": ["cmake", "--build", str(self.build_dir), "--config", "Release", "--target", "whisper-cli", "-j4"]},
            ],
            output_paths=self._build_output_paths,
            action=self._build,
        )
        binary_from_marker = self._binary_path_from_receipt()
        if binary_from_marker is None or not binary_from_marker.is_file():
            raise SetupError("Build receipt did not identify an executable whisper-cli")
        self.whisper_cli = binary_from_marker

        self.stages.run(
            "model-download",
            inputs={"model": self.model, "path": str(self.pt_model)},
            commands=[{"operation": "whisper.load_model during setup only", "model": self.model}],
            output_paths=lambda: [self.pt_model],
            action=self._load_model,
        )
        self.stages.run(
            "ggml-convert",
            inputs={
                "model_pt": path_snapshot(self.pt_model),
                "converter": path_snapshot(self.repo / "models" / "convert-pt-to-ggml.py"),
                "site_packages": str(self.site_packages),
            },
            commands=[
                {"argv": [str(self.venv_python), str(self.repo / "models" / "convert-pt-to-ggml.py"), str(self.pt_model), str(self.site_packages), str(self.models_dir)]}
            ],
            output_paths=lambda: [self.models_dir / f"ggml-{self.model}.bin"],
            action=self._convert_ggml,
        )
        self.stages.run(
            "coreml-convert",
            inputs={
                "model_pt": path_snapshot(self.pt_model),
                "converter": path_snapshot(self.repo / "models" / "convert-whisper-to-coreml.py"),
                "model": self.model,
                "optimize_ane": True,
                "quantize": bool(self.args.coreml_quantize),
                "wrapper_sha256": coreml_converter_wrapper_sha256(),
            },
            commands=[
                {
                    "argv": [
                        str(self.venv_python),
                        "-c",
                        "runpy wrapper patches whisper.load_model to the explicit checkpoint",
                        str(self.repo / "models" / "convert-whisper-to-coreml.py"),
                        self.model,
                        str(self.pt_model),
                        "--model",
                        self.model,
                        "--encoder-only",
                        "True",
                        "--optimize-ane",
                        "True",
                        "--quantize",
                        "True" if self.args.coreml_quantize else "False",
                    ],
                    "cwd": str(self.repo),
                }
            ],
            output_paths=lambda: [self.coreml_package],
            action=self._convert_coreml,
        )
        self.stages.run(
            "coreml-compile",
            inputs={
                "mlpackage": path_snapshot(self.coreml_package),
                "compute_units": COREML_COMPUTE_UNITS,
                "compiler": "coremltools.utils.compile_model",
            },
            commands=[
                {"operation": "coremltools.utils.compile_model", "source": str(self.coreml_package), "destination": str(self.compiled_encoder)}
            ],
            output_paths=lambda: [self.compiled_encoder],
            action=self._compile_coreml,
        )

        model_path = self.models_dir / f"ggml-{self.model}.bin"
        source_tree = tree_sha256(
            self.repo,
            ignored=(Path(".git"), Path("build"), Path(f"models/coreml-encoder-{self.model}.mlpackage")),
        )
        runtime_identity = {
            "runtime": "whisper.cpp",
            "version": REPO_TAG,
            "commit": REPO_COMMIT,
            "compute_units": COREML_COMPUTE_UNITS,
            "coreml_enabled": True,
            "metal_enabled": True,
            "coreml_fallback": False,
            "model": self.model,
            "coreml_quantize": bool(self.args.coreml_quantize),
            "compute_precision": "FLOAT16" if self.args.coreml_quantize else "FLOAT32",
            "note": "Core ML configuration or load evidence is not proof of ANE occupancy.",
        }
        self.stages.run(
            "config",
            inputs={
                "config_path": str(self.config_path),
                "whisper_cli": path_snapshot(self.whisper_cli),
                "model": path_snapshot(model_path),
                "coreml_encoder": path_snapshot(self.compiled_encoder),
                "runtime_identity": runtime_identity,
                "require_coreml": True,
            },
            commands=[
                {
                    "operation": "atomic runtime.json update",
                    "path": str(self.config_path),
                    "require_coreml": True,
                }
            ],
            output_paths=lambda: [self.config_path],
            action=lambda: self._write_config(runtime_identity),
        )

        previous_final = _load_receipt(self.receipts_dir / "setup.json") or {}
        tool_versions = self._tool_versions()
        build_marker = read_json_object(self.receipts_dir / "build-binary.json")
        linked_dylibs = build_marker.get("linked_dylibs", [])
        if not isinstance(linked_dylibs, list):
            raise SetupError("Build receipt has malformed linked_dylibs snapshots")
        executed_commands = self.command_runner.commands
        if not executed_commands and isinstance(previous_final.get("commands"), list):
            executed_commands = previous_final["commands"]
        final_receipt = {
            "receipt_version": RECEIPT_VERSION,
            "status": "success",
            "created_at": _now(),
            "runtime_dir": str(self.runtime_dir),
            "source_repository": str(self.repo),
            "source_url": REPO_URL,
            "source_tag": REPO_TAG,
            "source_commit": REPO_COMMIT,
            "source_tree_sha256": source_tree,
            "expected_source_patch": source_state["expected_patch"],
            "whisper_cli": str(self.whisper_cli),
            "whisper_cli_sha256": sha256_file(self.whisper_cli),
            "linked_dylibs": linked_dylibs,
            "model": self.model,
            "model_pt": str(self.pt_model),
            "model_pt_sha256": sha256_file(self.pt_model),
            "ggml_model": str(model_path),
            "ggml_model_sha256": sha256_file(model_path),
            "coreml_mlpackage": str(self.coreml_package),
            "coreml_mlpackage_tree_sha256": tree_sha256(self.coreml_package),
            "coreml_encoder": str(self.compiled_encoder),
            "coreml_encoder_tree_sha256": tree_sha256(self.compiled_encoder),
            "config": str(self.config_path),
            "runtime_identity": runtime_identity,
            "coreml_compute_precision": "FLOAT16" if self.args.coreml_quantize else "FLOAT32",
            "coreml_wrapper_sha256": coreml_converter_wrapper_sha256(),
            "require_coreml": True,
            "tool_versions": tool_versions,
            "commands": executed_commands,
            "stage_command_specs": {
                name: receipt.get("commands", [])
                for name, receipt in self.stages.results.items()
            },
            "stages": self.stages.results,
            "notes": [
                "The .pt model is reused from the local OpenAI Whisper cache when present.",
                "Model download, if needed, occurs only during this setup command; transcription performs no downloads.",
                "Core ML configuration/load evidence does not prove ANE occupancy; use hardware instrumentation for that claim.",
                "The official download-coreml-model.sh helper is intentionally not used.",
            ],
        }
        atomic_write_json(self.receipts_dir / "setup.json", final_receipt)
        print(f"setup complete: {self.runtime_dir}")
        print(f"config: {self.config_path}")
        print(f"whisper-cli: {self.whisper_cli}")
        return 0


def _doctor_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def _normalise_distribution_name(name: str) -> str:
    return name.lower().replace("_", "-")


def _metadata_versions(site_packages: Path) -> tuple[dict[str, str], str | None]:
    """Read distribution metadata without importing torch or other packages."""

    try:
        distributions = importlib_metadata.distributions(path=[str(site_packages)])
        versions: dict[str, str] = {}
        for distribution in distributions:
            name = distribution.metadata.get("Name")
            if name:
                versions[_normalise_distribution_name(name)] = distribution.version
        return versions, None
    except Exception as exc:  # metadata can fail on an incomplete venv
        return {}, f"{type(exc).__name__}: {exc}"


def _doctor_hash(
    checks: list[dict[str, Any]],
    name: str,
    path: Path,
    expected: Any,
    *,
    directory: bool = False,
) -> None:
    if not isinstance(expected, str):
        _doctor_check(checks, name, False, "receipt has no expected hash")
        return
    try:
        if directory:
            actual = tree_sha256(path)
        else:
            actual = sha256_file(path)
        _doctor_check(checks, name, actual == expected, f"expected={expected} actual={actual}")
    except SetupError as exc:
        _doctor_check(checks, name, False, str(exc))


def _doctor_cli_version(checks: list[dict[str, Any]], binary: Path) -> None:
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _doctor_check(checks, "whisper-cli --version", False, str(exc))
        return
    output = (result.stdout or result.stderr or "").strip().splitlines()
    detail = output[0] if output else f"exit={result.returncode}"
    _doctor_check(checks, "whisper-cli --version", result.returncode == 0, detail)


def _doctor_linked_dylibs(
    checks: list[dict[str, Any]],
    receipt: dict[str, Any],
) -> None:
    raw_source = receipt.get("source_repository")
    build_dir = Path(raw_source).expanduser().resolve() / "build" if isinstance(raw_source, str) else None
    raw_dylibs = receipt.get("linked_dylibs")
    if not isinstance(raw_dylibs, list) or any(not isinstance(item, dict) for item in raw_dylibs):
        _doctor_check(checks, "linked dylib receipt", False, "missing or malformed linked_dylibs snapshots")
        return
    recorded_paths = {
        str(Path(item["path"]).expanduser().resolve())
        for item in raw_dylibs
        if isinstance(item.get("path"), str)
    }
    current_paths = {
        str(path)
        for path in _find_build_dylibs(build_dir)
    } if build_dir is not None and build_dir.is_dir() else set()
    _doctor_check(
        checks,
        "build linked dylib set",
        current_paths == recorded_paths,
        f"recorded={sorted(recorded_paths)} current={sorted(current_paths)}",
    )
    for item in raw_dylibs:
        path_raw = item.get("path")
        if not isinstance(path_raw, str):
            continue
        path = Path(path_raw).expanduser().resolve()
        _doctor_hash(checks, f"linked dylib hash {path.name}", path, item.get("sha256"))


def doctor(runtime_dir_value: str | Path | None, config_value: str | Path | None, model: str) -> int:
    runtime_dir = resolve_runtime_dir(runtime_dir_value)
    config_path = resolve_config_path(config_value)
    checks: list[dict[str, Any]] = []
    receipt_path = runtime_dir / "receipts" / "setup.json"
    _doctor_check(checks, "runtime directory", runtime_dir.is_dir(), str(runtime_dir))
    _doctor_check(checks, "setup receipt", receipt_path.is_file(), str(receipt_path))
    config: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None
    if config_path.is_file():
        try:
            config = read_json_object(config_path)
            _doctor_check(checks, "config JSON", True, str(config_path))
        except SetupError as exc:
            _doctor_check(checks, "config JSON", False, str(exc))
    else:
        _doctor_check(checks, "config JSON", False, f"missing: {config_path}")

    if receipt_path.is_file():
        try:
            receipt = read_json_object(receipt_path)
        except SetupError as exc:
            _doctor_check(checks, "receipt JSON", False, str(exc))

    if config is not None:
        configured_model = config.get("model", model)
        _doctor_check(checks, "config.model", configured_model == model, repr(configured_model))
        config_paths: dict[str, Path] = {}
        for key in ("whisper_cli", "model_dir", "coreml_encoder"):
            raw = config.get(key)
            absolute = isinstance(raw, str) and Path(raw).expanduser().is_absolute()
            _doctor_check(checks, f"config.{key} absolute", absolute, str(raw))
            if absolute:
                path = Path(raw).expanduser().resolve()
                config_paths[key] = path
                exists = (
                    path.is_dir()
                    if key != "whisper_cli"
                    else path.is_file() and os.access(path, os.X_OK)
                )
                _doctor_check(checks, f"config.{key} ready", exists, str(path))
        _doctor_check(
            checks,
            "require_coreml",
            config.get("require_coreml") is True,
            repr(config.get("require_coreml")),
        )
        identity = config.get("runtime_identity")
        identity_ok = (
            isinstance(identity, dict)
            and identity.get("compute_units") == COREML_COMPUTE_UNITS
        )
        _doctor_check(checks, "runtime_identity.compute_units", identity_ok, repr(identity))

        expected_model = config_paths.get("model_dir", Path("__missing__")) / f"ggml-{model}.bin"
        expected_encoder = config_paths.get("model_dir", Path("__missing__")) / f"ggml-{model}-encoder.mlmodelc"
        _doctor_check(checks, "config model file", expected_model.is_file(), str(expected_model))
        _doctor_check(checks, "config Core ML sibling", expected_encoder.is_dir(), str(expected_encoder))
        if "coreml_encoder" in config_paths:
            _doctor_check(
                checks,
                "config.coreml_encoder sibling",
                config_paths["coreml_encoder"] == expected_encoder,
                str(config_paths["coreml_encoder"]),
            )

    if receipt is not None:
        _doctor_check(
            checks,
            "receipt status",
            receipt.get("status") == "success",
            repr(receipt.get("status")),
        )
        _doctor_check(
            checks,
            "pinned source commit",
            receipt.get("source_commit") == REPO_COMMIT,
            str(receipt.get("source_commit")),
        )
        receipt_paths: dict[str, Path] = {}
        for field, kind in (
            ("whisper_cli", "file"),
            ("ggml_model", "file"),
            ("coreml_encoder", "directory"),
            ("model_pt", "file"),
            ("coreml_mlpackage", "directory"),
        ):
            raw = receipt.get(field)
            path = Path(raw).expanduser().resolve() if isinstance(raw, str) else Path("__missing__")
            receipt_paths[field] = path
            exists = path.is_file() if kind == "file" else path.is_dir()
            _doctor_check(checks, f"receipt.{field}", exists, str(path))

        if config is not None:
            for config_key, receipt_key in (
                ("whisper_cli", "whisper_cli"),
                ("model_dir", "ggml_model"),
                ("coreml_encoder", "coreml_encoder"),
            ):
                raw = config.get(config_key)
                configured = Path(raw).expanduser().resolve() if isinstance(raw, str) else None
                expected = (
                    receipt_paths[receipt_key].parent
                    if config_key == "model_dir"
                    else receipt_paths[receipt_key]
                )
                _doctor_check(
                    checks,
                    f"config.{config_key} matches receipt",
                    configured is not None and configured == expected,
                    f"config={configured} receipt={expected}",
                )

        _doctor_hash(checks, "whisper-cli hash", receipt_paths["whisper_cli"], receipt.get("whisper_cli_sha256"))
        _doctor_hash(checks, "ggml model hash", receipt_paths["ggml_model"], receipt.get("ggml_model_sha256"))
        _doctor_hash(
            checks,
            "OpenAI checkpoint hash",
            receipt_paths["model_pt"],
            receipt.get("model_pt_sha256"),
        )
        _doctor_hash(
            checks,
            "Core ML package tree hash",
            receipt_paths["coreml_mlpackage"],
            receipt.get("coreml_mlpackage_tree_sha256"),
            directory=True,
        )
        _doctor_hash(
            checks,
            "Core ML encoder tree hash",
            receipt_paths["coreml_encoder"],
            receipt.get("coreml_encoder_tree_sha256"),
            directory=True,
        )
        if receipt_paths["whisper_cli"].is_file():
            _doctor_cli_version(checks, receipt_paths["whisper_cli"])
        _doctor_linked_dylibs(checks, receipt)

    venv_python = runtime_dir / ".venv" / "bin" / "python"
    _doctor_check(
        checks,
        "venv Python",
        venv_python.is_file() and os.access(venv_python, os.X_OK),
        str(venv_python),
    )
    manifest_path = runtime_dir / ".venv" / "setup-manifest.json"
    if manifest_path.is_file():
        try:
            manifest = read_json_object(manifest_path)
            site_raw = manifest.get("site_packages")
            site_packages = Path(site_raw).expanduser().resolve() if isinstance(site_raw, str) else None
            if site_packages is None or not site_packages.is_dir():
                _doctor_check(checks, "venv package metadata path", False, str(site_raw))
            else:
                actual_versions, metadata_error = _metadata_versions(site_packages)
                expected_versions = {
                    _normalise_distribution_name(name): version
                    for name, version in EXPECTED_PACKAGE_VERSIONS.items()
                }
                if metadata_error:
                    _doctor_check(checks, "pinned Python distributions", False, metadata_error)
                else:
                    mismatches = {
                        name: {"expected": version, "actual": actual_versions.get(name)}
                        for name, version in expected_versions.items()
                        if actual_versions.get(name) != version
                    }
                    _doctor_check(checks, "pinned Python distributions", not mismatches, repr(mismatches))
        except SetupError as exc:
            _doctor_check(checks, "venv package metadata", False, str(exc))
    else:
        _doctor_check(checks, "venv package metadata", False, f"missing: {manifest_path}")

    ready = all(bool(item["ok"]) for item in checks)
    for item in checks:
        print(f"{'PASS' if item['ok'] else 'FAIL'} {item['name']}: {item['detail']}")
    print(f"doctor: {'ready' if ready else 'not ready'}")
    return 0 if ready else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install the pinned whisper.cpp v1.8.7 Apple Silicon runtime in an isolated uv Python 3.12 environment. "
            "No setup work is performed by the normal extractor."
        )
    )
    parser.add_argument("--runtime-dir", help=f"runtime root (default: {DEFAULT_RUNTIME_DIR})")
    parser.add_argument(
        "--config",
        "--backend-config",
        dest="config",
        help="absolute or relative runtime.json path (default: installed skill/runtime.json)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI Whisper model name (default: small)")
    parser.add_argument("--pt-model", help="existing local OpenAI Whisper .pt file; custom missing paths are not downloaded")
    parser.add_argument("--force", action="store_true", help="rerun stages even when matching receipts exist")
    parser.add_argument(
        "--coreml-quantize",
        "--quantize",
        dest="coreml_quantize",
        action="store_true",
        help="pass --quantize True to the upstream Core ML converter (FLOAT16 path; default is True)",
    )
    parser.add_argument(
        "--no-coreml-quantize",
        "--no-quantize",
        dest="coreml_quantize",
        action="store_false",
        help="use the upstream converter's FP32 path (ANE occupancy is not expected)",
    )
    parser.set_defaults(coreml_quantize=True)
    parser.add_argument("--doctor", action="store_true", help="check installed runtime/config readiness without changing files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model or "/" in args.model or "\\" in args.model:
        print("error: --model must be a simple model name", file=sys.stderr)
        return 2
    if args.doctor:
        return doctor(args.runtime_dir, args.config, args.model)
    try:
        installer = Installer(args)
        with setup_lock(installer.runtime_dir):
            return installer.install()
    except SetupError as exc:
        print(f"setup error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("setup interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
