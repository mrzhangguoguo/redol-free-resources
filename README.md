# Redol Free Resources

Public release assets for free Redol resources.

## Bilibili Subtitle Extractor v2.0.0

Prefer existing Bilibili captions; when transcription is needed, use the configured Apple Silicon whisper.cpp Core ML / Metal runtime, with an explicit OpenAI Whisper fallback.

- [Download v2.0.0](https://github.com/mrzhangguoguo/redol-free-resources/releases/tag/bilibili-subtitle-extractor-v2.0.0)
- [Browse the Skill source](skills/bilibili-subtitle-extractor/SKILL.md)
- [Measured results and limitations](benchmarks/bilibili-subtitle-extractor-m3.md)

The portable ZIP excludes machine-local `runtime.json`, models, virtual environments, cookies, and media. Run the bundled setup helper on each target Mac to generate its runtime configuration. The existing `resources-v1.0.0` collection remains the repository's latest collection release so unrelated resource downloads remain compatible.

Source files are publicly available. This repository does not currently declare a repository-wide open-source license.
