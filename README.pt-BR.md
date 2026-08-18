# Jarvis

Assistente de voz local para Linux, com transcrição em português, palavra de ativação e registro controlado de comandos.

## Uso rápido

```bash
cp config.example.toml config.toml
ollama serve
ollama pull qwen2.5:0.5b
.venv/bin/python jarvis.py
```

Edite `config.toml` para informar modelos e fontes de áudio. Esse arquivo é ignorado pelo Git.

O Ollama interpreta apenas comandos não reconhecidos pelo parser local. Ele escolhe uma intenção conhecida e nunca recebe acesso direto ao shell. No modo público, ações externas exigem confirmação no terminal.

Consulte o [README principal](README.md) para requisitos, instalação, testes e segurança.
