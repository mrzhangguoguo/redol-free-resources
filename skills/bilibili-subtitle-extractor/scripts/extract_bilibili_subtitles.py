#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import html
import json
import math
import mimetypes
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

BILI_VIEW_API = "https://api.bilibili.com/x/web-interface/view"
BILI_PLAYER_API = "https://api.bilibili.com/x/player/v2"
BILI_REFERER = "https://www.bilibili.com/"
BILI_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
ALLOWED_BILIBILI_HOSTS = {
    "b23.tv",
    "www.b23.tv",
    "bilibili.com",
    "www.bilibili.com",
    "m.bilibili.com",
    "api.bilibili.com",
}
ALLOWED_BILIBILI_SUFFIXES = (
    ".bilibili.com",
    ".hdslb.com",
)
BROWSER_COOKIE_SOURCES = (
    ("chrome", Path.home() / "Library/Application Support/Google/Chrome"),
    ("chromium", Path.home() / "Library/Application Support/Chromium"),
    ("brave", Path.home() / "Library/Application Support/BraveSoftware/Brave-Browser"),
    ("edge", Path.home() / "Library/Application Support/Microsoft Edge"),
    ("vivaldi", Path.home() / "Library/Application Support/Vivaldi"),
)

BVID_RE = re.compile(r"(BV[0-9A-Za-z]{10})")
SUBTITLE_EXTENSIONS = {".json", ".srt", ".vtt"}
MEDIA_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".avi",
    ".flac",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}
TIME_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.\d]*|\d{1,2}:\d{2}[,.\d]*)\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.\d]*|\d{1,2}:\d{2}[,.\d]*)"
)
TAG_RE = re.compile(r"<[^>]+>")
SUPPORTED_BACKENDS = {"auto", "openai", "whispercpp"}
WHISPER_CPP_AUDIO_EXTENSIONS = {".flac", ".mp3", ".ogg", ".wav"}
COREML_LOADED_RE = re.compile(r"core\s*ml model loaded", re.IGNORECASE)
COREML_LOADING_RE = re.compile(r"loading core\s*ml model", re.IGNORECASE)
COREML_FAILED_RE = re.compile(r"(?:failed|error).*core\s*ml", re.IGNORECASE)
WHISPER_TIMING_RE = re.compile(
    r"whisper_print_timings:\s+(?P<name>[A-Za-z_]+)\s+time\s*=\s+(?P<ms>[0-9.]+)\s*ms",
    re.IGNORECASE,
)


@dataclass
class TranscriptBundle:
    title: str
    bvid: str | None
    source_label: str
    segments: list[dict[str, Any]]


@dataclass
class WhisperCppRuntime:
    """Validated local whisper.cpp runtime and model paths.

    The extractor intentionally keeps this as a description of already-installed
    files. It never downloads a binary, model, or Core ML bundle while running.
    """

    whisper_cli: Path
    model_dir: Path
    model_path: Path
    coreml_encoder: Path | None
    require_coreml: bool
    runtime_identity: Any
    config_path: Path | None = None


class MetricsRecorder:
    """Capture timing and backend evidence without leaking cookie arguments."""

    def __init__(
        self,
        requested_backend: str,
        model: str,
        language: str,
        output_path: Path | None = None,
    ) -> None:
        self.output_path = output_path
        self.started_wall = time.perf_counter()
        self.started_cpu = _cpu_snapshot()
        self.data: dict[str, Any] = {
            "schema_version": 1,
            "status": "running",
            "requested_backend": requested_backend,
            "resolved_backend": None,
            "model": model,
            "language": language,
            "stages": {},
            "stage_attempts": {},
            "commands": [],
            "commands_sanitized": True,
            "fallback_reason": None,
            "runtime_identity": None,
            "coreml_load_evidence": {
                "requested": False,
                "status": "not_requested",
                "lines": [],
                "ane_occupancy_proven": False,
            },
        }

    def set_backend(
        self,
        resolved_backend: str,
        *,
        fallback_reason: str | None = None,
    ) -> None:
        self.data["resolved_backend"] = resolved_backend
        self.data["fallback_reason"] = fallback_reason

    def set_runtime(self, runtime: WhisperCppRuntime) -> None:
        self.data["runtime_identity"] = _json_safe(
            runtime.runtime_identity
            if runtime.runtime_identity is not None
            else {}
        )
        self.data["runtime_identity_details"] = {
            "whisper_cli": str(runtime.whisper_cli),
            "model_dir": str(runtime.model_dir),
            "model": str(runtime.model_path),
            "coreml_encoder": str(runtime.coreml_encoder)
            if runtime.coreml_encoder
            else None,
            "require_coreml": runtime.require_coreml,
        }
        self.data["coreml_load_evidence"]["requested"] = runtime.require_coreml

    def record_command(self, command: list[str]) -> None:
        self.data["commands"].append(sanitize_command(command))

    def record_coreml_output(
        self,
        output: str,
        *,
        required: bool,
    ) -> None:
        lines = []
        status = "not_observed"
        for line in output.splitlines():
            if COREML_LOADING_RE.search(line) or COREML_LOADED_RE.search(line) or COREML_FAILED_RE.search(line):
                clean = line.strip()
                lines.append(clean)
                if COREML_LOADED_RE.search(line):
                    status = "loaded"
                elif COREML_FAILED_RE.search(line):
                    status = "failed"
        evidence = self.data["coreml_load_evidence"]
        evidence["requested"] = required
        evidence["status"] = status if lines else ("not_observed" if required else "not_required")
        evidence["lines"] = lines
        # A load message only proves that the Core ML model initialized. It
        # cannot establish ANE occupancy, so keep this explicit and false.
        evidence["ane_occupancy_proven"] = False

    def record_backend_timings(self, output: str) -> None:
        timings: dict[str, float] = {}
        for line in output.splitlines():
            match = WHISPER_TIMING_RE.search(line)
            if match:
                timings[match.group("name").lower()] = float(match.group("ms"))
        if timings:
            self.data["backend_timings_ms"] = timings

    def _record_stage(self, name: str, entry: dict[str, Any]) -> None:
        self.data["stages"][name] = entry
        self.data["stage_attempts"].setdefault(name, []).append(dict(entry))

    def skip_stage(self, name: str, reason: str) -> None:
        self._record_stage(name, {
            "status": "skipped",
            "reason": reason,
            "wall_seconds": 0.0,
            "cpu_seconds": 0.0,
            "user_cpu_seconds": 0.0,
            "system_cpu_seconds": 0.0,
            "average_cores": 0.0,
        })

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started_wall = time.perf_counter()
        started_cpu = _cpu_snapshot()
        status = "success"
        try:
            yield
        except BaseException:
            status = "error"
            raise
        finally:
            elapsed_wall = max(0.0, time.perf_counter() - started_wall)
            elapsed_cpu = _cpu_delta(started_cpu, _cpu_snapshot())
            self._record_stage(name, {
                "status": status,
                "wall_seconds": elapsed_wall,
                "cpu_seconds": elapsed_cpu["cpu_seconds"],
                "user_cpu_seconds": elapsed_cpu["user_cpu_seconds"],
                "system_cpu_seconds": elapsed_cpu["system_cpu_seconds"],
                "average_cores": (
                    elapsed_cpu["cpu_seconds"] / elapsed_wall
                    if elapsed_wall > 0
                    else 0.0
                ),
            })

    def finish(self, *, success: bool, error_message: str | None = None) -> None:
        elapsed_wall = max(0.0, time.perf_counter() - self.started_wall)
        elapsed_cpu = _cpu_delta(self.started_cpu, _cpu_snapshot())
        self.data["status"] = "success" if success else "error"
        self.data["wall_seconds"] = elapsed_wall
        self.data["cpu_seconds"] = elapsed_cpu["cpu_seconds"]
        self.data["user_cpu_seconds"] = elapsed_cpu["user_cpu_seconds"]
        self.data["system_cpu_seconds"] = elapsed_cpu["system_cpu_seconds"]
        self.data["average_cores"] = (
            elapsed_cpu["cpu_seconds"] / elapsed_wall if elapsed_wall > 0 else 0.0
        )
        if error_message:
            self.data["error"] = error_message

    def write(self) -> None:
        if not self.output_path:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


class SubtitleError(RuntimeError):
    pass


def _cpu_snapshot() -> dict[str, float]:
    """Return process plus already-reaped child CPU counters.

    ``os.times`` is available on the supported platforms and lets metrics
    include ffmpeg/whisper child CPU time without an extra dependency.
    """

    values = os.times()
    user = float(values.user + values.children_user)
    system = float(values.system + values.children_system)
    return {
        "user_cpu_seconds": user,
        "system_cpu_seconds": system,
        "cpu_seconds": user + system,
    }


def _cpu_delta(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, float]:
    return {
        key: max(0.0, float(after.get(key, 0.0) - before.get(key, 0.0)))
        for key in ("user_cpu_seconds", "system_cpu_seconds", "cpu_seconds")
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def sanitize_command(command: list[str]) -> list[str]:
    """Redact cookie values from a command receipt before persisting it."""

    sanitized: list[str] = []
    redact_next = False
    cookie_flags = {"--cookies", "--cookies-from-browser", "-cookies"}
    for token in command:
        token_text = str(token)
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if token_text in cookie_flags:
            sanitized.append(token_text)
            redact_next = True
            continue
        if "cookie" in token_text.lower():
            sanitized.append("<redacted>")
            continue
        sanitized.append(token_text)
    return sanitized


def log(level: str, message: str) -> None:
    print(f"[{level}] {message}", flush=True)


def info(message: str) -> None:
    log("INFO", message)


def warn(message: str) -> None:
    log("WARN", message)


def error(message: str) -> None:
    log("ERROR", message)


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".")
    return cleaned or "untitled"


def normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = TAG_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds + 0.5))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_timecode(raw: str) -> float:
    text = raw.strip().replace(",", ".")
    parts = text.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    raise SubtitleError(f"Unrecognized timestamp: {raw}")


def ensure_command(name: str) -> None:
    if shutil.which(name):
        return
    raise SubtitleError(f"Required command not found: {name}")


def default_backend_config_path() -> Path | None:
    """Return the installed skill's optional runtime.json, if present."""

    configured = os.environ.get("BILIBILI_SUBTITLE_EXTRACTOR_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[1] / "runtime.json"
    return candidate if candidate.exists() else None


def _load_backend_config_with_path(
    config_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, Path | None]:
    """Load a local backend config without creating or downloading anything.

    An explicitly supplied path is an error when it is absent or malformed.
    With no path, the sibling ``runtime.json`` is optional; its absence is the
    normal reason for auto mode to use the OpenAI Whisper implementation.
    """

    explicit = config_path is not None
    selected = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else default_backend_config_path()
    )
    if selected is None:
        return None, None
    if not selected.exists():
        raise SubtitleError(f"Backend config not found: {selected}")
    if not selected.is_file():
        raise SubtitleError(f"Backend config is not a file: {selected}")
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SubtitleError(f"Could not read backend config {selected}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SubtitleError("Backend config must contain a JSON object.")
    if explicit and not payload:
        raise SubtitleError(f"Backend config is empty: {selected}")
    return payload, selected


def load_backend_config(config_path: str | Path | None = None) -> dict[str, Any] | None:
    """Return the optional backend config object for callers that need it."""

    payload, _ = _load_backend_config_with_path(config_path)
    return payload


def _absolute_path_from_config(
    config: dict[str, Any], key: str, *, file_kind: str
) -> Path:
    raw = config.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise SubtitleError(f"Backend config requires {key!r}.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise SubtitleError(f"Backend config {key!r} must be an absolute path: {raw}")
    path = path.resolve()
    if file_kind == "file" and not path.is_file():
        raise SubtitleError(f"Backend config {key!r} does not point to a file: {path}")
    if file_kind == "directory" and not path.is_dir():
        raise SubtitleError(f"Backend config {key!r} does not point to a directory: {path}")
    if file_kind == "executable" and (not path.is_file() or not os.access(path, os.X_OK)):
        raise SubtitleError(f"Backend config {key!r} is not an executable file: {path}")
    return path


def _model_filename(model: str) -> str:
    token = str(model).strip()
    if not token:
        raise SubtitleError("Whisper model must not be empty.")
    if token.endswith(".bin"):
        return Path(token).name
    if token.startswith("ggml-"):
        return f"{token}.bin"
    return f"ggml-{token}.bin"


def validate_whispercpp_config(
    config: dict[str, Any],
    model: str,
    *,
    require_coreml: bool | None = None,
    config_path: Path | None = None,
) -> WhisperCppRuntime:
    """Validate an installed whisper.cpp runtime and derive its model paths."""

    if not isinstance(config, dict):
        raise SubtitleError("Backend config must contain a JSON object.")
    whisper_cli = _absolute_path_from_config(config, "whisper_cli", file_kind="executable")
    model_dir = _absolute_path_from_config(config, "model_dir", file_kind="directory")
    model_path = (model_dir / _model_filename(model)).resolve()
    if not model_path.is_file():
        raise SubtitleError(
            f"Whisper.cpp model not found for {model!r}: {model_path}. "
            "Set model_dir to an installed model directory; downloads are not performed."
        )
    configured_require_coreml = config.get("require_coreml", False)
    if configured_require_coreml is not None and not isinstance(configured_require_coreml, bool):
        raise SubtitleError("Backend config require_coreml must be a boolean.")
    strict_coreml = (
        bool(configured_require_coreml)
        if require_coreml is None
        else bool(require_coreml)
    )
    if strict_coreml and config.get("coreml_enabled") is False:
        raise SubtitleError("Backend config disables Core ML but strict Core ML was requested.")
    coreml_encoder: Path | None = None
    if strict_coreml:
        coreml_encoder = model_dir / f"{model_path.stem}-encoder.mlmodelc"
        if not coreml_encoder.is_dir():
            raise SubtitleError(
                f"Core ML encoder bundle not found for {model!r}: {coreml_encoder}."
            )
    runtime_identity = config.get("runtime_identity")
    if runtime_identity is None:
        runtime_identity = {}
    if not isinstance(runtime_identity, (dict, str, int, float, bool, list)):
        raise SubtitleError("Backend config runtime_identity must be JSON-compatible.")
    if isinstance(runtime_identity, dict):
        runtime_identity = dict(runtime_identity)
        runtime_identity.setdefault("coreml_enabled", config.get("coreml_enabled"))
        runtime_identity.setdefault("compute_units", config.get("compute_units"))
        runtime_identity.setdefault("require_coreml", strict_coreml)
    return WhisperCppRuntime(
        whisper_cli=whisper_cli,
        model_dir=model_dir,
        model_path=model_path,
        coreml_encoder=coreml_encoder,
        require_coreml=strict_coreml,
        runtime_identity=runtime_identity,
        config_path=config_path,
    )


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def resolve_transcription_backend(
    requested_backend: str,
    model: str,
    backend_config: str | Path | dict[str, Any] | None = None,
    metrics: MetricsRecorder | None = None,
) -> tuple[str, WhisperCppRuntime | None]:
    """Resolve auto/openai/whispercpp with strict explicit-backend errors."""

    requested = (requested_backend or "auto").lower().strip()
    if requested not in SUPPORTED_BACKENDS:
        raise SubtitleError(
            f"Unsupported transcription backend {requested_backend!r}; "
            f"choose one of {', '.join(sorted(SUPPORTED_BACKENDS))}."
        )
    if requested == "openai":
        if metrics:
            metrics.set_backend("openai")
        return "openai", None

    config: dict[str, Any] | None
    config_path: Path | None
    try:
        if isinstance(backend_config, dict):
            config, config_path = backend_config, None
        else:
            config, config_path = _load_backend_config_with_path(backend_config)
    except SubtitleError as exc:
        if requested == "whispercpp":
            raise
        reason = f"whisper.cpp backend config unavailable: {exc}"
        warn(f"{reason}; using OpenAI Whisper fallback.")
        if metrics:
            metrics.set_backend("openai", fallback_reason=reason)
        return "openai", None

    if requested == "auto" and not is_apple_silicon():
        reason = "validated Core ML auto mode requires Darwin arm64"
        warn(f"{reason}; using OpenAI Whisper fallback.")
        if metrics:
            metrics.set_backend("openai", fallback_reason=reason)
        return "openai", None
    if config is None:
        reason = "no whisper.cpp backend config was found"
        if requested == "whispercpp":
            raise SubtitleError(
                f"{reason}; pass --backend-config with installed whisper_cli and model_dir."
            )
        warn(f"{reason}; using OpenAI Whisper fallback.")
        if metrics:
            metrics.set_backend("openai", fallback_reason=reason)
        return "openai", None
    if requested == "auto" and config.get("require_coreml") is False:
        reason = "backend config does not require Core ML for auto mode"
        warn(f"{reason}; using OpenAI Whisper fallback.")
        if metrics:
            metrics.set_backend("openai", fallback_reason=reason)
        return "openai", None

    try:
        runtime = validate_whispercpp_config(
            config,
            model,
            # auto is specifically the validated Core ML + Metal path. An
            # explicit whispercpp request may intentionally benchmark Metal-only.
            require_coreml=True if requested == "auto" else None,
            config_path=config_path,
        )
    except SubtitleError as exc:
        if requested == "whispercpp":
            raise
        reason = f"whisper.cpp runtime validation failed: {exc}"
        warn(f"{reason}; using OpenAI Whisper fallback.")
        if metrics:
            metrics.set_backend("openai", fallback_reason=reason)
        return "openai", None
    if metrics:
        metrics.set_backend("whispercpp")
        metrics.set_runtime(runtime)
    return "whispercpp", runtime


def summarize_command_error(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout).strip()


def is_bilibili_cookie_retry_worthy(detail: str) -> bool:
    lowered = detail.lower()
    return (
        "http error 412" in lowered
        or "precondition failed" in lowered
        or "requires login" in lowered
        or "members only" in lowered
    )


def available_browser_cookie_sources() -> list[str]:
    available: list[str] = []
    for browser_name, browser_path in BROWSER_COOKIE_SOURCES:
        if browser_path.exists():
            available.append(browser_name)
    return available


def run_yt_dlp_download(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True)


def load_json_document(url: str, *, headers: dict[str, str] | None = None, retries: int = 0, description: str = "JSON resource") -> dict[str, Any]:
    request_headers = {
        "User-Agent": BILI_USER_AGENT,
        "Accept": "application/json, text/plain, */*",
    }
    if headers:
        request_headers.update(headers)
    request = Request(url, headers=request_headers)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=20) as response:
                body = response.read().decode("utf-8")
            return json.loads(body)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                warn(f"{description} failed ({exc}). Retrying {attempt + 1}/{retries}...")
                time.sleep(1 + attempt)
                continue
            break
    raise SubtitleError(f"{description} failed: {last_error}")


def load_bilibili_api(url: str, *, params: dict[str, Any], description: str) -> dict[str, Any]:
    full_url = f"{url}?{urlencode(params)}"
    payload = load_json_document(
        full_url,
        headers={"Referer": BILI_REFERER},
        retries=2,
        description=description,
    )
    if payload.get("code") not in (0, None):
        raise SubtitleError(
            f"{description} returned code={payload.get('code')} message={payload.get('message') or payload.get('msg')}"
        )
    return payload


def is_allowed_bilibili_host(host: str) -> bool:
    normalized = host.lower().strip(".")
    return normalized in ALLOWED_BILIBILI_HOSTS or normalized.endswith(ALLOWED_BILIBILI_SUFFIXES)


def resolve_bilibili_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.split("@")[-1].split(":")[0].lower().strip(".")
    if not is_allowed_bilibili_host(host):
        raise SubtitleError(f"Unsupported video host: {host}")
    if host not in {"b23.tv", "www.b23.tv"}:
        return url
    request = Request(url, headers={"User-Agent": BILI_USER_AGENT})
    with urlopen(request, timeout=20) as response:
        resolved = response.geturl()
    resolved_host = urlparse(resolved).netloc.split("@")[-1].split(":")[0].lower().strip(".")
    if not is_allowed_bilibili_host(resolved_host):
        raise SubtitleError(f"Resolved short link to unsupported host: {resolved_host}")
    return resolved


def looks_like_bvid(value: str) -> bool:
    return bool(re.fullmatch(r"BV[0-9A-Za-z]{10}", value.strip()))


def extract_bvid(value: str) -> str | None:
    match = BVID_RE.search(value)
    if not match:
        return None
    return match.group(1)


def looks_like_bilibili_url(value: str) -> bool:
    parsed = urlparse(value)
    host = parsed.netloc.split("@")[-1].split(":")[0].lower().strip(".")
    return parsed.scheme in {"http", "https"} and is_allowed_bilibili_host(host)


def detect_input(value: str) -> tuple[str, Any]:
    candidate = Path(value).expanduser()
    if candidate.exists():
        ext = candidate.suffix.lower()
        mime, _ = mimetypes.guess_type(candidate.name)
        if ext in SUBTITLE_EXTENSIONS:
            info("Input detected: Subtitle file")
            return "subtitle_file", candidate.resolve()
        if ext in MEDIA_EXTENSIONS or (mime and mime.startswith(("audio/", "video/"))):
            info("Input detected: Local media file")
            return "local_media", candidate.resolve()
        raise SubtitleError(f"Unsupported local file type: {candidate}")
    if looks_like_bvid(value):
        info("Input detected: BV number")
        return "bvid", value.strip()
    if looks_like_bilibili_url(value):
        info("Input detected: Bilibili URL")
        resolved_url = resolve_bilibili_url(value.strip())
        bvid = extract_bvid(resolved_url) or extract_bvid(value.strip())
        if not bvid:
            raise SubtitleError("Could not extract a BV number from the provided Bilibili URL.")
        return "bilibili_url", {"url": resolved_url, "bvid": bvid}
    raise SubtitleError("Could not detect input type. Provide a Bilibili URL, BV number, subtitle file, or media file.")


def fetch_video_metadata(bvid: str) -> tuple[str, int]:
    info(f"BV detected: {bvid}")
    info("Fetching cid...")
    payload = load_bilibili_api(BILI_VIEW_API, params={"bvid": bvid}, description="Bilibili view API")
    data = payload.get("data") or {}
    cid = data.get("cid")
    title = normalize_text(data.get("title") or bvid)
    if not cid:
        raise SubtitleError("Bilibili view API did not return cid.")
    info(f"cid found: {cid}")
    return title, int(cid)


def classify_subtitle(item: dict[str, Any]) -> tuple[int, str]:
    combined = " ".join(
        str(item.get(key, "")) for key in ("lan", "lan_doc", "type", "id")
    ).lower()
    has_chinese = any(token in combined for token in ("zh", "chinese", "中文", "汉语", "国语", "普通话"))
    has_english = any(token in combined for token in ("en", "english", "英文"))
    has_manual_hint = any(token in combined for token in ("人工", "manual", "official"))
    has_ai_hint = any(token in combined for token in (" ai", "ai ", "自动", "生成", "machine", "auto"))

    if has_chinese and has_manual_hint:
        return 0, "Chinese human subtitle"
    if has_chinese and has_ai_hint:
        return 1, "Chinese AI subtitle"
    if has_chinese:
        return 2, "Chinese subtitle"
    if has_english:
        return 3, "English subtitle"
    return 99, "Other subtitle"


def pick_subtitle(subtitles: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked = sorted(
        subtitles,
        key=lambda item: (
            classify_subtitle(item)[0],
            str(item.get("lan_doc", "")),
            str(item.get("lan", "")),
        ),
    )
    return ranked[0] if ranked else None


def normalize_subtitle_url(url: str) -> str:
    if url.startswith("//"):
        url = f"https:{url}"
    if url.startswith("/"):
        url = f"https://api.bilibili.com{url}"
    parsed = urlparse(url)
    host = parsed.netloc.split("@")[-1].split(":")[0].lower().strip(".")
    if parsed.scheme not in {"http", "https"}:
        raise SubtitleError(f"Unsupported subtitle URL scheme: {parsed.scheme or 'missing'}")
    if not is_allowed_bilibili_host(host):
        raise SubtitleError(f"Unsupported subtitle host: {host}")
    return url


def normalize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for segment in segments:
        start = segment.get("start")
        if start is None:
            continue
        try:
            start_value = float(start)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start_value) or start_value < 0:
            continue
        text = normalize_text(segment.get("text", ""))
        if not text:
            continue
        end = segment.get("end")
        end_value: float | None = None
        if end is not None:
            try:
                end_value = float(end)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(end_value) or end_value < start_value:
                continue
        normalized.append(
            {
                "start": start_value,
                "end": end_value,
                "text": text,
            }
        )
    normalized.sort(key=lambda item: item["start"])
    return normalized


def _cpp_timestamp_to_seconds(value: Any, *, offset_ms: bool = False) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and ":" in value:
        try:
            return parse_timecode(value)
        except (SubtitleError, ValueError):
            return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric / 1000.0 if offset_ms else numeric


def parse_whispercpp_json(payload: Any) -> list[dict[str, Any]]:
    """Normalize whisper.cpp ``--output-json`` transcription offsets.

    whisper.cpp writes ``offsets.from/to`` in milliseconds (its internal
    timestamps are 10 ms units), while generic subtitle JSON uses seconds.
    Keep the conversion local so callers cannot accidentally mix the units.
    """

    if not isinstance(payload, dict):
        raise SubtitleError("Unsupported whisper.cpp JSON format.")
    raw_segments = payload.get("transcription")
    if raw_segments is None:
        raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise SubtitleError("Whisper.cpp JSON did not contain a transcription array.")
    segments: list[dict[str, Any]] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            raise SubtitleError("Whisper.cpp transcription entries must be JSON objects.")
        offsets = item.get("offsets")
        timestamps = item.get("timestamps")
        if isinstance(offsets, dict):
            start = _cpp_timestamp_to_seconds(offsets.get("from"), offset_ms=True)
            end = _cpp_timestamp_to_seconds(offsets.get("to"), offset_ms=True)
        elif isinstance(timestamps, dict):
            start = _cpp_timestamp_to_seconds(timestamps.get("from"))
            end = _cpp_timestamp_to_seconds(timestamps.get("to"))
        else:
            # A few wrappers expose generic seconds fields. This fallback is
            # deliberately seconds-based and is never applied to offsets.
            start = _cpp_timestamp_to_seconds(item.get("start"))
            end = _cpp_timestamp_to_seconds(item.get("end"))
        if (
            start is None or end is None
            or not math.isfinite(start) or not math.isfinite(end)
            or start < 0 or end < start
        ):
            raise SubtitleError("Whisper.cpp output contains missing or invalid timestamps.")
        segments.append({"start": start, "end": end, "text": item.get("text", "")})
    return normalize_segments(segments)


# Short alias useful to callers/tests that use the upstream project spelling.
parse_cpp_json = parse_whispercpp_json


def parse_json_subtitle(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("transcription"), list):
            return parse_whispercpp_json(payload)
        if isinstance(payload.get("body"), list):
            return normalize_segments(
                [
                    {
                        "start": item.get("from"),
                        "end": item.get("to"),
                        "text": item.get("content", ""),
                    }
                    for item in payload["body"]
                ]
            )
        if isinstance(payload.get("segments"), list):
            return normalize_segments(
                [
                    {
                        "start": item.get("start"),
                        "end": item.get("end"),
                        "text": item.get("text", ""),
                    }
                    for item in payload["segments"]
                ]
            )
        if isinstance(payload.get("events"), list):
            return normalize_segments(
                [
                    {
                        "start": item.get("start"),
                        "end": item.get("end"),
                        "text": item.get("text", ""),
                    }
                    for item in payload["events"]
                ]
            )
    if isinstance(payload, list):
        return normalize_segments(payload)
    raise SubtitleError("Unsupported JSON subtitle format.")


def parse_srt_or_vtt(text: str) -> list[dict[str, Any]]:
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    segments: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].upper() == "WEBVTT":
            continue
        if lines[0].startswith(("NOTE", "STYLE", "REGION")):
            continue
        timestamp_index = 0
        if "-->" not in lines[0]:
            if len(lines) < 2 or "-->" not in lines[1]:
                continue
            timestamp_index = 1
        match = TIME_RE.search(lines[timestamp_index])
        if not match:
            continue
        content_lines = lines[timestamp_index + 1 :]
        text_value = normalize_text(" ".join(content_lines))
        if not text_value:
            continue
        segments.append(
            {
                "start": parse_timecode(match.group("start")),
                "end": parse_timecode(match.group("end")),
                "text": text_value,
            }
        )
    return normalize_segments(segments)


def parse_subtitle_file(path: Path, title_override: str | None) -> TranscriptBundle:
    title = normalize_text(title_override or path.stem)
    ext = path.suffix.lower()
    if ext == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        segments = parse_json_subtitle(payload)
        source_label = "本地字幕文件导入"
    elif ext in {".srt", ".vtt"}:
        segments = parse_srt_or_vtt(path.read_text(encoding="utf-8-sig"))
        source_label = "本地字幕文件导入"
    else:
        raise SubtitleError(f"Unsupported subtitle file extension: {ext}")
    if not segments:
        raise SubtitleError("Subtitle file parsed successfully but contained no subtitle segments.")
    return TranscriptBundle(title=title, bvid=None, source_label=source_label, segments=segments)


def fetch_official_subtitle(bvid: str, cid: int, title: str) -> TranscriptBundle | None:
    info("Trying official Bilibili subtitles...")
    payload = load_bilibili_api(
        BILI_PLAYER_API,
        params={"bvid": bvid, "cid": cid},
        description="Bilibili player API",
    )
    subtitles = (((payload.get("data") or {}).get("subtitle") or {}).get("subtitles")) or []
    if not subtitles:
        warn("No official subtitle found.")
        return None
    chosen = pick_subtitle(subtitles)
    if not chosen or not chosen.get("subtitle_url"):
        warn("Official subtitle list exists, but no usable subtitle_url was found.")
        return None
    priority, label = classify_subtitle(chosen)
    if priority == 99:
        warn("Only unsupported official subtitle variants were returned.")
        return None
    subtitle_url = normalize_subtitle_url(str(chosen["subtitle_url"]))
    info(f"Selected official subtitle: {label}")
    subtitle_payload = load_json_document(
        subtitle_url,
        headers={"Referer": BILI_REFERER},
        description="Official subtitle download",
    )
    segments = parse_json_subtitle(subtitle_payload)
    if not segments:
        warn("Official subtitle JSON downloaded, but no subtitle segments were found.")
        return None
    return TranscriptBundle(
        title=title,
        bvid=bvid,
        source_label="B站原始字幕",
        segments=segments,
    )


def find_downloaded_audio(download_dir: Path, stdout: str) -> Path | None:
    for line in reversed([entry.strip() for entry in stdout.splitlines() if entry.strip()]):
        candidate = Path(line)
        if candidate.exists():
            return candidate.resolve()
    audio_files = sorted(download_dir.glob("*.mp3"), key=lambda path: path.stat().st_mtime)
    if audio_files:
        return audio_files[-1].resolve()
    return None


def download_audio(
    url: str,
    temp_dir: Path,
    cookies: Path | None,
    metrics: MetricsRecorder | None = None,
) -> Path:
    ensure_command("yt-dlp")
    ensure_command("ffmpeg")
    download_dir = temp_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    base_cmd = [
        "yt-dlp",
        "--no-playlist",
        "--print",
        "after_move:filepath",
        "-x",
        "--audio-format",
        "mp3",
        "-o",
        str(download_dir / "%(title)s.%(ext)s"),
    ]
    cmd = list(base_cmd)
    if cookies:
        cmd.extend(["--cookies", str(cookies)])
    cmd.append(url)
    info("Falling back to audio download.")
    if metrics:
        metrics.record_command(cmd)
    result = run_yt_dlp_download(cmd)
    if result.returncode != 0:
        detail = summarize_command_error(result)
        if not cookies and is_bilibili_cookie_retry_worthy(detail):
            browser_sources = available_browser_cookie_sources()
            if browser_sources:
                warn("yt-dlp hit Bilibili anti-bot checks without cookies. Retrying with browser cookies...")
                retry_failures: list[str] = []
                for browser_name in browser_sources:
                    retry_cmd = list(base_cmd)
                    retry_cmd.extend(["--cookies-from-browser", browser_name, url])
                    info(f"Retrying yt-dlp with browser cookies from {browser_name}...")
                    if metrics:
                        metrics.record_command(retry_cmd)
                    retry_result = run_yt_dlp_download(retry_cmd)
                    if retry_result.returncode == 0:
                        audio_path = find_downloaded_audio(download_dir, retry_result.stdout)
                        if audio_path:
                            return audio_path
                        retry_failures.append(
                            f"{browser_name}: download succeeded but no audio file was found"
                        )
                        continue
                    retry_detail = summarize_command_error(retry_result)
                    first_line = retry_detail.splitlines()[0] if retry_detail else "unknown error"
                    warn(f"Browser cookie retry failed for {browser_name}: {first_line}")
                    retry_failures.append(f"{browser_name}: {first_line}")
                raise SubtitleError(
                    "yt-dlp download failed without cookies, and browser-cookie retries did not succeed. "
                    "Retry with --cookies /absolute/path/to/cookies.txt if your browser session is not usable.\n"
                    + "\n".join(retry_failures)
                )
        raise SubtitleError(
            "yt-dlp download failed. The video may require login cookies. "
            "Retry with --cookies /absolute/path/to/cookies.txt.\n"
            f"{detail}"
        )
    audio_path = find_downloaded_audio(download_dir, result.stdout)
    if not audio_path:
        raise SubtitleError("yt-dlp reported success, but no downloaded audio file was found.")
    return audio_path


def _whispercpp_language(language: str) -> str:
    aliases = {
        "chinese": "zh",
        "中文": "zh",
        "普通话": "zh",
        "mandarin": "zh",
        "english": "en",
        "英文": "en",
        "japanese": "ja",
        "日语": "ja",
        "korean": "ko",
        "韩语": "ko",
        "auto": "auto",
    }
    value = str(language).strip()
    return aliases.get(value.lower(), value.lower())


def transcribe_with_openai(
    media_path: Path,
    temp_dir: Path,
    title: str,
    model: str,
    language: str,
    metrics: MetricsRecorder | None = None,
    threads: int | None = None,
) -> TranscriptBundle:
    ensure_command("whisper")
    ensure_command("ffmpeg")
    output_dir = temp_dir / "whisper"
    output_dir.mkdir(parents=True, exist_ok=True)
    info(f"Running Whisper {model}...")
    cmd = [
        "whisper",
        str(media_path),
        "--model",
        model,
        "--language",
        language,
        "--output_format",
        "json",
        "--output_dir",
        str(output_dir),
        "--verbose",
        "False",
    ]
    if threads is not None:
        if threads <= 0:
            raise SubtitleError("--threads must be a positive integer.")
        cmd.extend(["--threads", str(threads)])
    if metrics:
        metrics.record_command(cmd)
        metrics.skip_stage("audio_conversion", "OpenAI Whisper performs media decoding internally.")
    stage = metrics.stage("transcription") if metrics else contextlib.nullcontext()
    with stage:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or "unknown error"
            raise SubtitleError(f"Whisper failed: {detail}")
        json_candidates = sorted(output_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
        if not json_candidates:
            raise SubtitleError("Whisper completed but no JSON output was found.")
        try:
            payload = json.loads(json_candidates[-1].read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SubtitleError(f"Whisper JSON output could not be read: {exc}") from exc
        segments = parse_json_subtitle(payload)
        if not segments:
            raise SubtitleError("Whisper output contained no subtitle segments.")
    return TranscriptBundle(
        title=title,
        bvid=None,
        source_label=f"Whisper {model} 转写",
        segments=segments,
    )


def prepare_whispercpp_audio(
    media_path: Path,
    temp_dir: Path,
    metrics: MetricsRecorder | None = None,
) -> Path:
    """Convert unsupported media containers to 16 kHz mono WAV for whisper.cpp."""

    if media_path.suffix.lower() in WHISPER_CPP_AUDIO_EXTENSIONS:
        if metrics:
            metrics.skip_stage("audio_conversion", "Input format is supported by whisper.cpp.")
        return media_path
    stage = metrics.stage("audio_conversion") if metrics else contextlib.nullcontext()
    with stage:
        ensure_command("ffmpeg")
        output_dir = temp_dir / "whispercpp-audio"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{sanitize_filename(media_path.stem)}.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]
        if metrics:
            metrics.record_command(cmd)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or "unknown error"
            raise SubtitleError(f"ffmpeg audio conversion failed: {detail}")
        if not output_path.is_file():
            raise SubtitleError(
                f"ffmpeg reported success, but converted audio was not found: {output_path}"
            )
    return output_path


def transcribe_with_whispercpp(
    media_path: Path,
    temp_dir: Path,
    title: str,
    model: str,
    language: str,
    runtime: WhisperCppRuntime,
    threads: int | None = None,
    metrics: MetricsRecorder | None = None,
) -> TranscriptBundle:
    audio_path = prepare_whispercpp_audio(media_path, temp_dir, metrics)
    output_dir = temp_dir / "whispercpp"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "transcript"
    cmd = [
        str(runtime.whisper_cli),
        "--model",
        str(runtime.model_path),
        "--file",
        str(audio_path),
        "--language",
        _whispercpp_language(language),
        "--output-json",
        "--output-file",
        str(output_base),
    ]
    if threads is not None:
        if threads <= 0:
            raise SubtitleError("--threads must be a positive integer.")
        cmd.extend(["--threads", str(threads)])
    if metrics:
        metrics.record_command(cmd)
        metrics.set_runtime(runtime)
    info(f"Running whisper.cpp {model}...")
    stage = metrics.stage("transcription") if metrics else contextlib.nullcontext()
    with stage:
        result = subprocess.run(cmd, capture_output=True, text=True)
        (output_dir / "stdout.log").write_text(result.stdout or "", encoding="utf-8")
        (output_dir / "stderr.log").write_text(result.stderr or "", encoding="utf-8")
        combined_output = "\n".join(
            part for part in (result.stdout or "", result.stderr or "") if part
        )
        if metrics:
            metrics.record_coreml_output(combined_output, required=runtime.require_coreml)
            metrics.record_backend_timings(combined_output)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip() or "unknown error"
            raise SubtitleError(f"whisper.cpp failed: {detail}")
        if runtime.require_coreml and metrics:
            evidence_status = metrics.data["coreml_load_evidence"]["status"]
            if evidence_status != "loaded":
                raise SubtitleError(
                    "whisper.cpp completed without explicit Core ML model load evidence; "
                    "strict Core ML mode refuses to claim the optimized run."
                )
        elif runtime.require_coreml and not metrics:
            if not COREML_LOADED_RE.search(combined_output):
                raise SubtitleError(
                    "whisper.cpp completed without explicit Core ML model load evidence."
                )
        json_path = output_base.with_suffix(".json")
        if not json_path.is_file():
            candidates = sorted(output_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
            if candidates:
                json_path = candidates[-1]
            else:
                raise SubtitleError("whisper.cpp completed but no JSON output was found.")
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise SubtitleError(f"whisper.cpp JSON output could not be read: {exc}") from exc
        segments = parse_whispercpp_json(payload)
        if not segments:
            raise SubtitleError("whisper.cpp output contained no subtitle segments.")
    return TranscriptBundle(
        title=title,
        bvid=None,
        source_label=f"whisper.cpp {model} 转写",
        segments=segments,
    )


def transcribe_media(
    media_path: Path,
    temp_dir: Path,
    title: str,
    model: str,
    language: str,
    backend: str = "auto",
    backend_config: str | Path | dict[str, Any] | None = None,
    threads: int | None = None,
    metrics: MetricsRecorder | None = None,
) -> TranscriptBundle:
    requested_backend = (backend or "auto").lower().strip()
    resolved_backend, runtime = resolve_transcription_backend(
        requested_backend,
        model,
        backend_config,
        metrics,
    )
    if resolved_backend == "openai":
        return transcribe_with_openai(
            media_path,
            temp_dir,
            title,
            model,
            language,
            metrics,
            threads,
        )
    if runtime is None:  # defensive guard for type checkers and malformed callers
        raise SubtitleError("whisper.cpp backend resolved without a validated runtime.")
    try:
        return transcribe_with_whispercpp(
            media_path,
            temp_dir,
            title,
            model,
            language,
            runtime,
            threads,
            metrics,
        )
    except (OSError, SubtitleError) as exc:
        if requested_backend != "auto":
            raise
        reason = f"whisper.cpp runtime execution failed: {exc}"
        warn(f"{reason}; using OpenAI Whisper fallback.")
        if metrics:
            metrics.data["runtime_execution_error"] = str(exc)
            metrics.set_backend("openai", fallback_reason=reason)
        return transcribe_with_openai(
            media_path,
            temp_dir,
            title,
            model,
            language,
            metrics,
            threads,
        )


def transcribe_with_whisper(
    media_path: Path,
    temp_dir: Path,
    title: str,
    model: str,
    language: str,
    backend: str = "openai",
    backend_config: str | Path | dict[str, Any] | None = None,
    threads: int | None = None,
    metrics: MetricsRecorder | None = None,
) -> TranscriptBundle:
    """Backward-compatible entry point with optional backend controls."""

    return transcribe_media(
        media_path,
        temp_dir,
        title,
        model,
        language,
        backend,
        backend_config,
        threads,
        metrics,
    )


def build_transcript_markdown(bundle: TranscriptBundle) -> str:
    bvid = bundle.bvid or "无"
    lines = [
        "# 视频字幕",
        "",
        "## 基本信息",
        f"- 标题：{bundle.title}",
        f"- BV号：{bvid}",
        f"- 字幕来源：{bundle.source_label}",
        "",
        "---",
        "",
    ]
    for segment in bundle.segments:
        lines.append(f"[{format_timestamp(segment['start'])}]")
        lines.append(segment["text"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_transcript_markdown(bundle: TranscriptBundle, output_dir: Path) -> tuple[Path, Path]:
    safe_title = sanitize_filename(bundle.title)
    transcript_path = output_dir / f"{safe_title}_字幕.md"
    organized_path = output_dir / f"{safe_title}_整理版.md"
    transcript_path.write_text(build_transcript_markdown(bundle), encoding="utf-8")
    return transcript_path.resolve(), organized_path.resolve()


def build_fallback_url(input_kind: str, input_payload: Any) -> str | None:
    if input_kind == "bilibili_url":
        return str(input_payload["url"])
    if input_kind == "bvid":
        return f"https://www.bilibili.com/video/{input_payload}/"
    return None


def process_bilibili_input(
    input_kind: str,
    input_payload: Any,
    temp_dir: Path,
    cookies: Path | None,
    whisper_model: str,
    language: str,
    backend: str = "auto",
    backend_config: str | Path | dict[str, Any] | None = None,
    threads: int | None = None,
    force_transcribe: bool = False,
    metrics: MetricsRecorder | None = None,
) -> TranscriptBundle:
    bvid = input_payload if input_kind == "bvid" else input_payload.get("bvid")
    title: str | None = None
    if bvid:
        try:
            title, cid = fetch_video_metadata(bvid)
            if force_transcribe:
                info("Force transcribe requested; skipping official subtitle lookup.")
            else:
                official = fetch_official_subtitle(bvid, cid, title)
                if official:
                    if metrics:
                        metrics.set_backend("official_subtitle")
                        metrics.data["subtitle_source"] = official.source_label
                        metrics.skip_stage(
                            "audio_conversion",
                            "Official subtitles bypass audio processing.",
                        )
                        metrics.skip_stage(
                            "transcription",
                            "Official subtitles bypass ASR.",
                        )
                    return official
        except SubtitleError as exc:
            warn(str(exc))
    fallback_url = build_fallback_url(input_kind, input_payload)
    if not fallback_url:
        raise SubtitleError("No fallback Bilibili URL available for audio download.")
    audio_path = download_audio(fallback_url, temp_dir, cookies, metrics)
    fallback_title = title or normalize_text(audio_path.stem)
    bundle = transcribe_media(
        audio_path,
        temp_dir,
        fallback_title,
        whisper_model,
        language,
        backend,
        backend_config,
        threads,
        metrics,
    )
    if bvid:
        bundle.bvid = bvid
    return bundle


def process_local_media(
    media_path: Path,
    temp_dir: Path,
    whisper_model: str,
    language: str,
    title_override: str | None,
    backend: str = "auto",
    backend_config: str | Path | dict[str, Any] | None = None,
    threads: int | None = None,
    metrics: MetricsRecorder | None = None,
) -> TranscriptBundle:
    title = normalize_text(title_override or media_path.stem)
    return transcribe_media(
        media_path,
        temp_dir,
        title,
        whisper_model,
        language,
        backend,
        backend_config,
        threads,
        metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Bilibili subtitles with official-subtitle priority and Whisper fallback."
    )
    parser.add_argument("input", help="Bilibili URL, BV number, local subtitle file, or local media file")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where Markdown files should be written. Defaults to current directory.",
    )
    parser.add_argument(
        "--cookies",
        help="Optional cookies.txt path for yt-dlp when Bilibili requires login.",
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="Whisper model name. Defaults to small.",
    )
    parser.add_argument(
        "--language",
        default="Chinese",
        help="Whisper language name. Defaults to Chinese.",
    )
    parser.add_argument(
        "--backend",
        choices=sorted(SUPPORTED_BACKENDS),
        default="auto",
        help="ASR backend: auto, openai, or whispercpp. Defaults to auto.",
    )
    parser.add_argument(
        "--backend-config",
        help="Optional JSON config with absolute whisper_cli/model_dir paths.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Optional whisper.cpp computation thread count.",
    )
    parser.add_argument(
        "--force-transcribe",
        action="store_true",
        help="For Bilibili URLs/BV numbers, skip the official subtitle lookup and transcribe.",
    )
    parser.add_argument(
        "--metrics-json",
        help="Write a JSON receipt with backend, timing, command, and runtime evidence.",
    )
    parser.add_argument(
        "--title",
        help="Optional title override for local subtitle/media inputs.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary download/transcription artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_json).expanduser().resolve() if args.metrics_json else None
    metrics = MetricsRecorder(
        args.backend,
        args.whisper_model,
        args.language,
        metrics_path,
    )
    if args.threads is not None and args.threads <= 0:
        message = "--threads must be a positive integer."
        metrics.finish(success=False, error_message=message)
        try:
            metrics.write()
        except OSError as exc:
            error(f"Could not write metrics JSON: {exc}")
        error(message)
        return 1
    cookies = Path(args.cookies).expanduser().resolve() if args.cookies else None

    try:
        with metrics.stage("full_run"):
            if cookies and not cookies.exists():
                raise SubtitleError(f"Cookies file not found: {cookies}")
            input_kind, input_payload = detect_input(args.input)
            temp_context: contextlib.AbstractContextManager[str]
            if args.keep_temp:
                temp_path = output_dir / f".bilibili-subtitle-extractor-{int(time.time())}"
                temp_path.mkdir(parents=True, exist_ok=True)
                temp_context = contextlib.nullcontext(str(temp_path))
            else:
                temp_context = tempfile.TemporaryDirectory(
                    prefix=".bilibili-subtitle-extractor-",
                    dir=str(output_dir),
                )

            with temp_context as temp_dir_str:
                temp_dir = Path(temp_dir_str)
                if input_kind == "subtitle_file":
                    bundle = parse_subtitle_file(input_payload, args.title)
                    metrics.set_backend("local_subtitle")
                    metrics.data["subtitle_source"] = bundle.source_label
                    metrics.skip_stage(
                        "audio_conversion",
                        "Local subtitle input bypasses audio processing.",
                    )
                    metrics.skip_stage("transcription", "Local subtitle input bypasses ASR.")
                elif input_kind == "local_media":
                    bundle = process_local_media(
                        input_payload,
                        temp_dir,
                        args.whisper_model,
                        args.language,
                        args.title,
                        args.backend,
                        args.backend_config,
                        args.threads,
                        metrics,
                    )
                else:
                    bundle = process_bilibili_input(
                        input_kind,
                        input_payload,
                        temp_dir,
                        cookies,
                        args.whisper_model,
                        args.language,
                        args.backend,
                        args.backend_config,
                        args.threads,
                        args.force_transcribe,
                        metrics,
                    )

            transcript_path, organized_path = write_transcript_markdown(bundle, output_dir)
        metrics.finish(success=True)
        try:
            metrics.write()
        except OSError as exc:
            error(f"Could not write metrics JSON: {exc}")
            return 1
        info("Transcript Markdown generated.")
        info(f"Transcript markdown: {transcript_path}")
        info(f"Suggested organized markdown path: {organized_path}")
        print(
            json.dumps(
                {
                    "title": bundle.title,
                    "bvid": bundle.bvid,
                    "subtitle_source": bundle.source_label,
                    "transcript_markdown": str(transcript_path),
                    "organized_markdown": str(organized_path),
                    "backend": metrics.data.get("resolved_backend"),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except SubtitleError as exc:
        metrics.finish(success=False, error_message=str(exc))
        try:
            metrics.write()
        except OSError as write_exc:
            error(f"Could not write metrics JSON: {write_exc}")
        error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
