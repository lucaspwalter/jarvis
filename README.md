# Jarvis

Assistente de voz local para Linux, com ativação por palavra-chave, transcrição em português e execução de comandos do desktop.

## Visão geral

O Jarvis escuta a palavra de ativação, captura o comando, transcreve a fala e executa ações configuradas. O projeto também oferece ditado para uma janela já aberta do Codex.

## Funcionalidades

- Ativação por “Jarvis” usando OpenWakeWord
- Transcrição em português com Faster-Whisper
- Resposta de voz local com Piper TTS
- Comandos para aplicativos, mídia, monitores e desempenho do PC
- Ditado para o Codex aberto, sem criar outra janela
- Integração com Waybar e PipeWire/PulseAudio
- Testes automatizados para interpretação de comandos

## Tecnologias

- Python 3
- Faster-Whisper
- OpenWakeWord
- Piper TTS
- NumPy, SciPy e scikit-learn
- PipeWire/PulseAudio, `parec`, `pw-play` e `wtype`

## Instalação

```bash
git clone https://github.com/lucaspwalter/jarvis.git
cd jarvis
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Os modelos de wake word, Piper e Whisper devem estar disponíveis nos caminhos configurados em `jarvis.py`.

## Uso

```bash
.venv/bin/python jarvis.py
```

Diga “Jarvis” e, depois, o comando. Para ditar no Codex já aberto, diga “Jarvis, digite para mim”, aguarde a resposta e fale o texto.

## Estrutura

```text
jarvis.py              Assistente, captura, transcrição e comandos
write_codex.py         Digitação na janela existente do Codex
tests/test_commands.py Testes de interpretação
requirements.txt       Dependências Python
```

## Licença

Ainda não definida.
