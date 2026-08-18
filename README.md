# Jarvis

Local Linux voice assistant with Portuguese transcription, wake-word detection and a permission-controlled command registry.

## Overview

Jarvis listens for “Jarvis”, transcribes the following command and executes only actions registered by the local application. Ollama can classify unclear speech, but it never receives shell access.

## Features

- OpenWakeWord activation
- Faster-Whisper Portuguese transcription
- Piper local text-to-speech
- Configurable audio sources, models and Ollama endpoint
- Public mode with visible confirmation before external actions
- Optional Codex dictation
- Unit tests and GitHub Actions validation

## Requirements

Linux with Python 3.11+, PipeWire/PulseAudio, `parec`, `pactl`, `pw-play`, `ffmpeg`, `wtype` and the model files listed in `config.example.toml`.

## Getting started

```bash
git clone https://github.com/lucaspwalter/jarvis.git
cd jarvis
python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp config.example.toml config.toml
```

Edit `config.toml` with the audio source and model paths. Keep personal configuration out of Git; `config.toml` is ignored.

## Ollama

Install Ollama using its official instructions, start the local service, and download the model:

```bash
ollama serve
ollama pull qwen2.5:0.5b
```

The default endpoint is `http://127.0.0.1:11434/api/generate`. Change `ollama.model` or `ollama.url` in `config.toml` when necessary.

Ollama is used only after the deterministic command matcher fails. In public mode it classifies the request against known intents; it cannot invent or execute shell commands.

## Usage

```bash
.venv/bin/python jarvis.py
```

Say “Jarvis” followed by a registered command. Run tests with:

```bash
.venv/bin/python -m unittest discover -s tests -q
```

## Public and unsafe modes

Public mode asks for confirmation in a terminal before actions with external effects:

```bash
JARVIS_PUBLIC_MODE=1 .venv/bin/python jarvis.py
```

`JARVIS_UNSAFE=1` skips that prompt only for actions already registered locally. It does not enable arbitrary shell execution.

## License

MIT. See [LICENSE](LICENSE).
