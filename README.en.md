# Jarvis

English documentation for the local Linux voice assistant. See [README.md](README.md) for the complete guide.

The public build uses configurable paths and audio sources, a local Ollama classifier for unclear commands, and a fixed action registry. Ollama never receives direct shell access. Public mode asks for confirmation before external actions.

```bash
cp config.example.toml config.toml
ollama serve
ollama pull qwen2.5:0.5b
.venv/bin/python jarvis.py
```

Read [README.md](README.md) for requirements, configuration, tests and security details.
