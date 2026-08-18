# Jarvis

Local voice assistant for Linux with wake-word detection, Portuguese speech
transcription and desktop command execution.

## Public mode

Public mode uses Ollama to interpret every command. Actions with external side
effects open a terminal authorization prompt and run only after `y`/`yes`.

```bash
JARVIS_PUBLIC_MODE=1 .venv/bin/python jarvis.py
```

### Unsafe mode

Use only on a trusted computer. It skips the visual confirmation for actions
already registered by Jarvis. It does not give Ollama unrestricted shell access.

```bash
JARVIS_PUBLIC_MODE=1 JARVIS_UNSAFE=1 .venv/bin/python jarvis.py
```

Ollama never receives direct shell access. The local action registry remains the
execution boundary.
