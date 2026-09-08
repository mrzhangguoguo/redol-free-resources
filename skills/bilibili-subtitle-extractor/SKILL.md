---
name: bilibili-subtitle-extractor
description: Extract subtitles from a Bilibili URL, BV number, local subtitle file, or local media file. Prefer official or imported subtitles, and use the optional configured whisper.cpp Apple Silicon backend only when ASR is required.
---

# Bilibili Subtitle Extractor

Use this skill when the user wants a timestamped Bilibili transcript or a
faithful article derived from one. The extractor preserves the original
official-subtitle-first workflow and adds an optional, pre-installed
whisper.cpp backend for Apple Silicon.

## Run the extractor

```bash
python3 ~/.codex/skills/bilibili-subtitle-extractor/scripts/extract_bilibili_subtitles.py \
  "<input>" --output-dir "<output-dir>"
```

`<input>` may be a Bilibili URL, a `BV` number, a local `.json`, `.srt`, or
`.vtt` subtitle file, or a local audio/video file. The output directory
defaults to the current directory. Existing named options remain available:

```text
--cookies /absolute/path/to/cookies.txt
--whisper-model small
--language Chinese
--title "自定义标题"
--keep-temp
```

The ASR experiment options are:

```text
--backend auto|openai|whispercpp
--backend-config /absolute/path/to/runtime.json
--threads N
--metrics-json /absolute/path/to/metrics.json
--force-transcribe
```

The default model is multilingual Whisper `small`, and the default language
is `Chinese`. `--threads` is optional and must be a positive integer. The
backend is `auto` unless specified otherwise.

`auto` selects a validated configured Core ML encoder plus Metal decoder only
on Darwin arm64. If the configured runtime is absent, invalid, or fails during
execution, it warns, records the reason when metrics are enabled, and uses the
original OpenAI Whisper implementation. `openai` always selects that
implementation.
`whispercpp` is strict: an unavailable or failing configured runtime is an
error and must not silently fall back. This makes an explicit benchmark
failure visible. A configured `whispercpp` runtime may be Metal-only or may
require Core ML according to `runtime.json`; read
[references/optimization.md](references/optimization.md) when choosing it.

The accelerated backend consumes already-installed files described by the
optional `runtime.json` beside the skill, or by `--backend-config`. Paths for
`whisper_cli` and `model_dir` must be absolute. The optimized `whispercpp`
path never downloads a binary, model, or Core ML bundle during transcription.
The preserved OpenAI fallback may download its missing `.pt` checkpoint on
first use, so verify that cache separately when offline work could fall back
to OpenAI. Use the optional setup helper and its `--doctor` check before
transcribing; setup, hashes, receipts, and the exact command are documented
in [references/runtime-setup.md](references/runtime-setup.md). Do not mutate
system Python packages. Read [references/optimization.md](references/optimization.md)
for Apple Silicon validation and benchmark procedure; do not load it for an
ordinary official-subtitle or local-subtitle import.

For Bilibili audio download, `yt-dlp` and `ffmpeg` must remain on `PATH`.
The OpenAI fallback also requires the existing `whisper` CLI. The setup helper
prepares the accelerated runtime; it does not replace these media utilities.
Local subtitle imports need none of those executables.

## Input precedence and bypasses

Apply the following routing before selecting an ASR backend:

- A local subtitle file is parsed and normalized directly. It never downloads
  media or runs Whisper, including when `--force-transcribe` is present.
- A local audio/video file is sent directly to the selected ASR backend.
- A Bilibili URL or `BV` number first fetches metadata and tries official
  subtitles. Prefer Chinese human subtitles, then Chinese AI subtitles, then
  other Chinese subtitles, then English subtitles. An official subtitle
  result bypasses audio download and ASR.
- `--force-transcribe` is an explicit experiment flag for a Bilibili URL or BV
  input when the benchmark must bypass official subtitles. Do not add it to a
  normal extraction merely to prefer ASR, and do not use it to override the
  local-subtitle bypass.
- If official subtitles are unavailable or the URL workflow is forced, use
  `yt-dlp` audio download and the selected ASR backend. Preserve the existing
  browser-cookie recovery for Bilibili `412 Precondition Failed`, login, or
  members-only errors when no explicit `--cookies` file was supplied.
- Bilibili API requests retry twice before the URL workflow gives up on
  official subtitles. If `yt-dlp` still fails after the cookie retry, report
  the real download error and that cookies may be required; do not fabricate a
  transcript.

Transcription results use normalized `start`, `end`, and `text` segments.
Malformed whisper.cpp timestamps fail explicitly; local subtitle imports retain
their existing parsing behavior and Markdown layout. Read the `字幕来源` field and report
it exactly (for example, `B站原始字幕`, `本地字幕文件导入`, `Whisper small 转写`,
or `whisper.cpp small 转写`). A Core ML configuration or load message
does not prove Neural Engine occupancy; never relabel it as an ANE transcript.

## Required workflow and output boundary

1. Determine the requested output directory. Ordinary use produces both the
   raw transcript and the organized article; produce only the raw transcript
   when the user explicitly asks for raw-only output, without asking an extra
   preference question when none was given.
2. Run the extractor and read its real output and error. Do not invent a
   transcript when Bilibili, `yt-dlp`, ffmpeg, or ASR fails.
3. The script creates one raw Markdown file, `<title>_字幕.md`, containing
   metadata and timestamped text. It prints a suggested sibling path,
   `<title>_整理版.md`, but does not create or claim that organized article.
4. For ordinary use, read the raw transcript and create the organized article
   as a separate agent step. Skip that step only for an explicit raw-only
   request. Preserve facts, numbers, examples, uncertainty, and ordering where
   they matter. Remove filler or repeated fragments only when the transcript
   supports the edit. Do not add content absent from the transcript. Keep ambiguous amounts in their original
   wording, and preserve which items a stated price covers.
5. On an ASR failure, report the actual error and leave the organized article
   absent. If `--metrics-json` was supplied, report the receipt path and its
   status; it records requested/resolved backend, stage timings, sanitized
   commands, fallback reason, and runtime evidence.

The raw Markdown layout is:

```markdown
# 视频字幕
## 基本信息
- 标题：xxx
- BV号：xxx
- 字幕来源：B站原始字幕 / 本地字幕文件导入 / Whisper small 转写
---
[00:00]
字幕内容
```

Suggested article sections are `背景`, `核心问题`, `主要观点`, `论证过程`,
`案例与细节`, and `总结`. Use only the sections supported by the transcript.

When reporting a raw-only run, use language such as:

```text
已生成字幕：/absolute/path/视频标题_字幕.md
整理版建议路径：/absolute/path/视频标题_整理版.md（尚未创建）
字幕来源：B站原始字幕 / 本地字幕文件导入 / Whisper small 转写
```

For ordinary use, report the real organized-article path after creating it. For
an explicit raw-only run, report only the suggested path and state that it was
not created. Keep the subtitle source distinct from the ASR backend.

## Optional Obsidian handoff

Use this only when the user requests an Obsidian note, a vault folder, or
backlinks. Load and follow the `obsidian-cli` skill before touching the vault.

1. Confirm the active vault with `obsidian vault info=path`; use the requested
   `vault=<name>` when needed, or stop with the actual mismatch.
2. After the organized article exists, create it in the requested folder,
   usually `好文章收集/视频标题.md`, with known frontmatter such as
   `source`, `bv`, `created`, `type: bilibili-summary`, `subtitle_source`, and
   relevant tags.
3. Search the user-named folders for related notes, especially
   `好文章收集` and `公众号/果叔日记` when they are part of the request. Add
   only strong, explainable wikilinks under `## 相关笔记`.
4. Before adding reciprocal links, search the target notes for the new title.
   Add deduplicated links inside this marker pair:

   ```markdown
   ## 关联文章
   <!-- bilibili-extractor:backlinks:start -->
   - [[好文章收集/视频标题|视频标题]]：关联原因。
   <!-- bilibili-extractor:backlinks:end -->
   ```

5. Verify with `obsidian read path="..."`, `obsidian links path="..."`,
   `obsidian unresolved`, and `obsidian backlinks path="..."` when the index
   has caught up. If backlinks lag after a fresh write, read the reciprocal
   notes and inspect their `obsidian links` output, then state that evidence
   precisely.

Do not create or modify an Obsidian article from a missing or failed raw
transcript.
