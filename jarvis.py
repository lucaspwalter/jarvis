#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import os
import re
import signal
import subprocess
import tempfile
import time
import unicodedata
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel
from piper.voice import PiperVoice
from scipy.signal import butter, sosfilt


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
WAKE_MODELS_DIR = MODELS_DIR / "openwakeword"
PIPER_MODEL = MODELS_DIR / "piper" / "pt_BR-cadu-medium.onnx"
WHISPER_SMALL_ROOT = Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots"
WHISPER_MEDIUM_ROOT = Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-medium/snapshots"
LISTENING_STATE = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "jarvis-listening"
AUDIO_SOURCE = "jarvis_mic"
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_280
FRAME_BYTES = FRAME_SAMPLES * 2
WAKE_THRESHOLD = 0.15
WAKE_CONFIRM_FRAMES = 2
INPUT_GAIN = 1.60
SILENCE_RMS = 110.0
SILENCE_SECONDS = 0.12
MAX_COMMAND_SECONDS = 9.0
MIN_COMMAND_SECONDS = 0.5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("jarvis")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def add_honorific(text: str) -> str:
    if "senhor" in normalize(text).split():
        return text
    punctuation = text[-1] if text.endswith((".", "!", "?")) else "."
    body = text[:-1] if text.endswith((".", "!", "?")) else text
    return f"Senhor, {body}{punctuation}"


def detached(command: list[str]) -> None:
    subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


@dataclass(frozen=True)
class CommandResult:
    response: str
    action: list[str] | None = None


def expand_command_variations(script_commands: list[tuple[str, list[str], str]]) -> list[tuple[str, list[str], str]]:
    expansions = {
        "resfriar normal": ("restaurar|normalizar|reverter|voltar|retornar|desfazer|cancelar|parar|deixar|colocar", "limites originais|resfriamento normal|temperatura normal|modo normal|configuração normal|configuracao normal|desempenho original|parâmetros originais|parametros originais|estado original|controle normal|configurações originais|configuracoes originais"),
        "performace": ("ativar|ative|iniciar|inicie|ligar|ligue|usar|use|colocar|coloque", "performance|modo performance|modo desempenho|modo jogo|preparar pc para jogar|pc para jogar|alto desempenho|jogos|jogar|perfil gamer|perfil de jogos|potência máxima|potencia maxima"),
        "atualizarlayout": ("atualizar|atualize|sincronizar|sincronize|recarregar|recarregue|aplicar|aplique|refazer|refaça|refazer|corrigir", "layout|o layout|configurações|configuracoes|configuração visual|configuracao visual|barra|waybar|tela|ambiente|aparência|aparencia|dotfiles"),
        "3monitor": ("ativar|ative|iniciar|inicie|ligar|ligue|usar|use|conectar|conecte|adicionar|adicione", "terceiro monitor|terceira monitora|três monitores|tres monitores|3 monitores|tela três|tela tres|monitor extra|monitor adicional|notebook como monitor|notebook de monitor|monitor do notebook"),
        "pc-remoto": ("abrir|abra|iniciar|inicie|conectar|conecte|acessar|acesse|usar|use|ligar|ligue", "pc remoto|computador remoto|acesso remoto|pc no notebook|computador no notebook|desktop remoto|meu pc remotamente|meu computador|sessão remota|sessao remota|moonlight|stream do pc"),
        "autoclicker": ("ativar|ative|iniciar|inicie|ligar|ligue|abrir|abra|executar|execute|usar|use", "autoclicker|auto clicker|auto clique|cliques automáticos|cliques automaticos|cliques repetidos|cliques rápidos|cliques rapidos|clicador automático|clicador automatico|automação de cliques|automacao de cliques|clicar automaticamente|cliques contínuos|cliques continuos"),
        "consumo": ("mostrar|mostre|ver|veja|consultar|consulte|exibir|exiba|informar|informe|checar|verificar", "consumo|consumo do pc|status do pc|estado do computador|cpu e ram|cpu ram|memória e cpu|memoria e cpu|temperatura do pc|recursos do pc|uso do computador|desempenho do pc"),
        "notebook": ("abrir|abra|iniciar|inicie|conectar|conecte|acessar|acesse|entrar|entre|usar|use", "notebook|o notebook|meu notebook|computador portátil|computador portatil|ssh do notebook|conexão do notebook|conexao do notebook|sessão do notebook|sessao do notebook|terminal do notebook|acesso ao notebook"),
        "parar3monitor": ("parar|pare|desativar|desative|desligar|desligue|encerrar|encerre|fechar|feche|finalizar|finalize", "terceiro monitor|terceira monitora|três monitores|tres monitores|3 monitores|tela três|tela tres|monitor extra|monitor adicional|monitor remoto|moonlight|transmissão remota|transmissao remota"),
        "pesados": ("mostrar|mostre|ver|veja|listar|liste|exibir|exiba|consultar|consulte|encontrar|encontre", "processos pesados|programas pesados|aplicativos pesados|processos que mais usam|programas que mais usam|maior consumo|maior uso de cpu|maior uso de memória|maior uso de memoria|tarefas pesadas|processos do pc|aplicativos do pc"),
        "padrao": ("restaurar|restaure|voltar|volte|retornar|retorne|ativar|ative|usar|use|colocar|coloque|iniciar|inicie", "padrão|padrao|modo padrão|modo padrao|sessão padrão|sessao padrao|modo normal|sessão normal|sessao normal|configuração padrão|configuracao padrao|perfil padrão|perfil padrao|estado padrão|estado padrao|configuração original|configuracao original"),
        "resfriar": ("ativar|ative|iniciar|inicie|ligar|ligue|usar|use|reduzir|reduza|baixar|baixe|diminuir|diminua", "resfriamento|modo frio|resfriar pc|esfriar pc|temperatura|temperatura do pc|calor do pc|calor|temperatura baixa|resfriamento do computador|modo de resfriamento|controle térmico|controle termico"),
        "rede": ("mostrar|mostre|ver|veja|consultar|consulte|exibir|exiba|informar|informe|checar|verificar", "rede|status da rede|estado da rede|internet|status da internet|conexão|conexao|conexão de rede|conexao de rede|ip|ip local|gateway|tailscale"),
    }
    expanded = []
    fillers = ("", "por favor", "senhor", "agora", "neste momento", "no computador", "aqui", "para mim", "por gentileza", "sem demora")
    typo_fillers = ("por favo", "senho", "agoraa", "neste mometo", "no computado", "aki", "pra mim", "por gentilezaa", "sem demoro", "favor")
    for script, aliases, response in script_commands:
        verbs_targets = expansions.get(script)
        if verbs_targets:
            verbs, targets = (part.split("|") for part in verbs_targets)
            generated = [normalize(f"{filler} {verb} {target}") for filler in fillers for verb in verbs for target in targets]
            generated += [normalize(f"{filler} {verb} {target}") for filler in typo_fillers for verb in verbs for target in targets]
            aliases = sorted(set(aliases + generated))
        expanded.append((script, aliases, response))
    return expanded


MEDIA_VARIATIONS = {
    normalize(f"{filler} {verb} {target}")
    for filler in ("", "por favor", "senhor", "agora", "neste momento", "no computador", "aqui", "para mim", "por gentileza", "sem demora")
    for verb in ("pause", "pausar", "pausa", "despause", "despausar", "continue", "continuar", "retome", "retomar", "toque", "tocar")
    for target in ("a música", "a musica", "a mídia", "a midia", "o vídeo", "o video", "o som", "a reprodução", "a reproducao", "a faixa", "a transmissão", "a transmissao", "o conteúdo", "o conteudo", "o áudio", "o audio", "o stream")
}
MEDIA_ERROR_VARIATIONS = {
    normalize(f"{filler} {verb} {target}")
    for filler in ("", "por favo", "senho", "agoraa", "neste mometo", "aki", "pra mim", "sem demoro", "favor", "por gentilezaa")
    for verb in ("pouse", "pousar", "pousa", "pousei", "pesar", "pozar", "pausar", "pausa", "pausei", "despouse", "despousar", "dispause", "dispausar", "despausa", "despausar", "retoma", "retomei", "continuar", "continaur", "tocaar")
    for target in ("a música", "a musica", "a mídia", "a midia", "o vídeo", "o video", "o som", "a reprodução", "a reproducao", "a faixa", "a transmissão", "a transmissao", "o conteúdo", "o conteudo", "o áudio", "o audio", "o stream")
}
CODEX_WRITE_VARIATIONS = {
    normalize(f"{filler} {phrase} {suffix}")
    for filler in ("", "por favor", "senhor", "agora", "rapidamente", "neste momento", "para mim", "por gentileza", "sem demora")
    for phrase in ("digite para mim", "digita para mim", "digitar para mim", "pode digitar para mim", "escreva para mim", "escreve para mim", "escrever para mim")
    for suffix in ("", "agora", "por favor", "senhor", "para mim", "no computador", "neste momento", "sem demora", "por gentileza", "rapidamente", "aí", "ai", "aqui", "neste instante", "quando puder", "se puder", "por gentileza senhor", "agora senhor", "para mim agora", "no terminal", "no computador agora")
}


def codex_write_payload(text: str) -> str | None:
    match = re.search(
        r"\b(?:digite|digitar|digita|escreva|escrever|escreve)\s+para\s+mim\s*[:,;-]?\s*(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    payload = match.group(1).strip().strip(".,;:-")
    return payload if re.search(r"[\wÀ-ÿ]", payload) else None


def is_codex_write_request(text: str) -> bool:
    return bool(re.search(r"\b(?:digite|digitar|digita|escreva|escrever|escreve)\s+para\s+mim\b", text, flags=re.IGNORECASE))


def remove_dictation_prompt(text: str) -> str:
    return re.sub(
        r"^(?:senhor[, ]*)?(?:pode|podem)\s+(?:ditar|digitar)\s*[,.;:!-]*\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).strip()


EXTRA_COMMAND_SPECS = [
    ("mostrar ip local", "ip -brief address", "Mostrando IP local."),
    ("mostrar ip publico", "curl -4 -s https://ifconfig.me; echo", "Mostrando IP público."),
    ("testar internet", "ping -c 4 1.1.1.1", "Testando internet."),
    ("testar latencia", "ping -c 4 1.1.1.1", "Testando latência."),
    ("mostrar espaco em disco", "df -h", "Mostrando espaço em disco."),
    ("mostrar processos cpu", "ps -eo pid,comm,%cpu --sort=-%cpu | head -15", "Mostrando processos por CPU."),
    ("mostrar memoria", "free -h", "Mostrando memória."),
    ("mostrar temperatura", "sensors", "Mostrando temperatura."),
    ("mostrar versao do kernel", "uname -a", "Mostrando versão do kernel."),
    ("mostrar tempo ligado", "uptime", "Mostrando tempo ligado."),
    ("abrir github", "xdg-open https://github.com", "Abrindo GitHub."),
    ("abrir youtube", "xdg-open https://youtube.com", "Abrindo YouTube."),
    ("abrir discord", "discord", "Abrindo Discord."),
    ("abrir vscode", "code", "Abrindo VS Code."),
    ("abrir downloads", "dolphin $HOME/Downloads", "Abrindo Downloads."),
    ("capturar tela", "grim $HOME/Imagens/captura-$(date +%Y%m%d-%H%M%S).png", "Capturando tela."),
    ("bloquear computador", "loginctl lock-session", "Bloqueando computador."),
    ("proxima musica", "playerctl next", "Próxima música."),
    ("musica anterior", "playerctl previous", "Música anterior."),
    ("mostrar processos memoria", "ps -eo pid,comm,%mem --sort=-%mem | head -15", "Mostrando processos por memória."),
    ("mostrar gateway", "ip route | head -5", "Mostrando gateway."),
    ("mostrar dns", "resolvectl status | sed -n '1,35p'", "Mostrando DNS."),
    ("mostrar interfaces de rede", "ip -brief link", "Mostrando interfaces de rede."),
    ("mostrar conexoes de rede", "ss -tuln", "Mostrando conexões de rede."),
    ("listar dispositivos usb", "lsusb", "Listando dispositivos USB."),
    ("listar dispositivos pci", "lspci", "Listando dispositivos PCI."),
    ("mostrar audio", "wpctl status", "Mostrando dispositivos de áudio."),
    ("mostrar bluetooth", "bluetoothctl devices", "Mostrando dispositivos Bluetooth."),
    ("mostrar bateria", "upower -i $(upower -e | grep BAT | head -1)", "Mostrando bateria."),
    ("abrir configuracoes", "systemsettings", "Abrindo configurações."),
    ("abrir gerenciador de processos", "ksysguard", "Abrindo gerenciador de processos."),
    ("abrir monitor do sistema", "missioncenter", "Abrindo monitor do sistema."),
    ("abrir documentos", "dolphin $HOME/Documentos", "Abrindo Documentos."),
    ("abrir imagens", "dolphin $HOME/Imagens", "Abrindo Imagens."),
    ("abrir projetos", "dolphin $HOME/Documentos/projetos", "Abrindo Projetos."),
    ("mostrar servicos ativos", "systemctl --user --type=service --state=running", "Mostrando serviços ativos."),
    ("mostrar servicos falhos", "systemctl --user --failed", "Mostrando serviços falhos."),
    ("mostrar logs do jarvis", "journalctl --user -u jarvis.service -n 40 --no-pager", "Mostrando logs do Jarvis."),
    ("reiniciar jarvis", "systemctl --user restart jarvis.service", "Reiniciando Jarvis."),
    ("mostrar atualizacoes", "checkupdates", "Mostrando atualizações."),
    ("atualizar sistema", "sudo pacman -Syu", "Atualizando sistema."),
    ("ver status do git", "cd $HOME/Documentos/projetos/jarvis && git status", "Mostrando status do Git."),
    ("executar testes", "cd $HOME/Documentos/projetos/jarvis && .venv/bin/python -m unittest discover -s tests -q", "Executando testes."),
    ("abrir projeto jarvis", "kitty --directory $HOME/Documentos/projetos/jarvis", "Abrindo projeto Jarvis."),
    ("mostrar branches", "cd $HOME/Documentos/projetos/jarvis && git branch -a", "Mostrando branches."),
    ("mostrar commits", "cd $HOME/Documentos/projetos/jarvis && git log --oneline -15", "Mostrando commits."),
    ("enviar projeto", "cd $HOME/Documentos/projetos/jarvis && git push", "Enviando projeto."),
    ("mostrar pontos de montagem", "findmnt", "Mostrando pontos de montagem."),
    ("mostrar portas abertas", "ss -tuln", "Mostrando portas abertas."),
    ("listar processos", "ps aux --sort=-%cpu | head -20", "Listando processos."),
    ("reiniciar audio", "systemctl --user restart pipewire pipewire-pulse", "Reiniciando áudio."),
    ("recarregar waybar", "pkill -SIGUSR2 waybar", "Recarregando Waybar."),
    ("reiniciar waybar", "systemctl --user restart waybar.service", "Reiniciando Waybar."),
    ("mostrar janelas", "hyprctl clients", "Mostrando janelas abertas."),
    ("recarregar hyprland", "hyprctl reload", "Recarregando Hyprland."),
    ("reiniciar wallpaper", "systemctl --user restart hyprpaper.service", "Reiniciando wallpaper."),
    ("suspender computador", "systemctl suspend", "Suspendendo computador."),
    ("reiniciar computador", "systemctl reboot", "Reiniciando computador."),
    ("desligar computador", "systemctl poweroff", "Desligando computador."),
    ("abrir spotify", "spotify", "Abrindo Spotify."),
    ("abrir steam", "steam", "Abrindo Steam."),
    ("abrir prism launcher", "prismlauncher", "Abrindo PrismLauncher."),
    ("abrir retroarch", "retroarch", "Abrindo RetroArch."),
    ("abrir controle de audio", "pavucontrol", "Abrindo controle de áudio."),
    ("mostrar telas", "hyprctl monitors all", "Mostrando telas."),
    ("mostrar clipboard", "wl-paste", "Mostrando clipboard."),
    ("copiar clipboard", "wl-copy", "Clipboard pronto para receber texto."),
    ("silenciar microfone", "pactl set-source-mute jarvis_mic 1", "Silenciando microfone."),
    ("ativar microfone", "pactl set-source-mute jarvis_mic 0", "Ativando microfone."),
    ("mostrar volume do microfone", "pactl get-source-volume jarvis_mic", "Mostrando volume do microfone."),
    ("microfone cem por cento", "pactl set-source-volume jarvis_mic 100%", "Microfone em 100 por cento."),
    ("microfone oitenta por cento", "pactl set-source-volume jarvis_mic 80%", "Microfone em 80 por cento."),
    ("microfone vinte por cento", "pactl set-source-volume jarvis_mic 20%", "Microfone em 20 por cento."),
    ("capturar area da tela", "grim -g \"$(slurp)\" $HOME/Imagens/area-$(date +%Y%m%d-%H%M%S).png", "Capturando área da tela."),
    ("gravar tela", "wf-recorder -f $HOME/Vídeos/gravação-$(date +%Y%m%d-%H%M%S).mp4", "Gravando tela."),
    ("parar gravacao", "pkill -INT wf-recorder", "Parando gravação."),
    ("mostrar calendario", "cal -3", "Mostrando calendário."),
    ("mostrar data", "date", "Mostrando data."),
    ("mostrar relogio", "date '+%H:%M:%S'", "Mostrando relógio."),
    ("ler clipboard", "wl-paste", "Lendo clipboard."),
    ("limpar clipboard", "printf '' | wl-copy", "Limpando clipboard."),
    ("abrir lixeira", "dolphin trash:/", "Abrindo lixeira."),
    ("esvaziar lixeira", "gio trash --empty", "Esvaziando lixeira."),
    ("compactar projeto", "cd $HOME/Documentos/projetos/jarvis && tar -czf $HOME/jarvis-backup.tar.gz --exclude=.venv --exclude=.git .", "Compactando projeto."),
    ("fazer backup do jarvis", "cd $HOME/Documentos/projetos/jarvis && tar -czf $HOME/jarvis-backup.tar.gz --exclude=.venv --exclude=.git .", "Fazendo backup do Jarvis."),
    ("sincronizar projeto", "cd $HOME/Documentos/projetos/jarvis && git pull --ff-only", "Sincronizando projeto."),
    ("mostrar versao python", "python --version", "Mostrando versão do Python."),
    ("mostrar pacotes python", "cd $HOME/Documentos/projetos/jarvis && .venv/bin/pip list", "Mostrando pacotes Python."),
    ("validar python", "cd $HOME/Documentos/projetos/jarvis && .venv/bin/python -m py_compile jarvis.py write_codex.py", "Validando Python."),
    ("verificar codigo", "cd $HOME/Documentos/projetos/jarvis && git diff --check", "Verificando código."),
    ("mostrar diferencas", "cd $HOME/Documentos/projetos/jarvis && git diff", "Mostrando diferenças."),
    ("abrir readme", "xdg-open $HOME/Documentos/projetos/jarvis/README.md", "Abrindo README."),
    ("abrir arquivos recentes", "dolphin --select $HOME/Documentos/projetos/jarvis", "Abrindo arquivos recentes."),
    ("minimizar janela", "hyprctl dispatch movetoworkspacesilent special", "Minimizando janela."),
    ("fechar janela atual", "hyprctl dispatch killactive", "Fechando janela atual."),
    ("tela cheia", "hyprctl dispatch fullscreen", "Alternando tela cheia."),
    ("mostrar workspace", "hyprctl activeworkspace", "Mostrando workspace atual."),
    ("mostrar janela atual", "hyprctl activewindow", "Mostrando janela atual."),
    ("abrir pasta home", "dolphin $HOME", "Abrindo pasta pessoal."),
]
EXTRA_FILLERS = ("", "por favor", "senhor", "agora", "rapidamente", "neste momento", "no computador", "aqui", "para mim", "sem demora")
EXTRA_VERBS = ("mostrar", "mostre", "mostra", "ver", "veja", "exibir", "exiba", "consultar", "consulte", "abrir", "abra", "iniciar", "inicie", "iniciar", "reiniciar", "reinicie", "executar", "execute", "fazer", "faça", "rodar", "rode", "testar", "teste", "testa", "me diga", "diga")

def extra_command_result(command: str) -> CommandResult | None:
    matches = []
    targets = [re.sub(r"^(mostrar|abrir|testar|executar|rodar|reiniciar|silenciar|ativar|capturar|gravar|parar|validar)\s+", "", phrase) for phrase, _, _ in EXTRA_COMMAND_SPECS]
    for phrase, shell, response in EXTRA_COMMAND_SPECS:
        target = re.sub(r"^(mostrar|abrir|testar|executar|rodar|reiniciar|silenciar|ativar|capturar|gravar|parar|validar)\s+", "", phrase)
        verb = phrase.split()[0]
        stem = verb[: max(3, len(verb) - 2)]
        target_match = re.search(rf"(?<!\w){re.escape(target)}(?!\w)", command)
        verb_match = re.search(rf"\b{re.escape(stem)}", command)
        if target_match and (targets.count(target) == 1 or verb_match):
            matches.append((len(target.split()), len(target), shell, response))
    if not matches:
        return None
    best_score = max((words, chars) for words, chars, _, _ in matches)
    best = [(shell, response) for words, chars, shell, response in matches if (words, chars) == best_score]
    if len({shell for shell, _ in best}) > 1:
        return CommandResult("Comando ambíguo. Diga novamente.")
    shell, response = best[0]
    return CommandResult(response, ["kitty", "-e", "bash", "-lc", f"{shell}; read -rp 'Enter para fechar...' _"])


def interpret_command(text: str, now: datetime | None = None) -> CommandResult:
    payload = codex_write_payload(text)
    if payload:
        return CommandResult("Digitando.", [str(BASE_DIR / "write_codex.py"), payload])
    command = normalize(text)
    command = re.sub(r"^(ei +)?jarvis\b", "", command).strip(" ,")
    command = re.sub(r"\bdesat(?:ive|iva|ivar|ime)\b", "desative", command)
    command = re.sub(r"\bdespousar\b", "despausar", command)
    command = re.sub(r"\bdisposa\b", "despausar", command)
    command = re.sub(r"\b(dispause|dispausar|despouse)\b", "despausar", command)
    command = re.sub(r"\b(pousar|pousa|pouse)\b", "pausar", command)
    command = re.sub(r"\b(pose|poze)\b", "pause", command)
    command = re.sub(r"\bpalos\b", "pause", command)
    command = re.sub(r"\bamidia\b", "midia", command)
    command = command.replace("fadi fox", "firefox").replace("fe de foco", "firefox").replace("grom", "chrome")
    now = now or datetime.now()

    applications = {
        "google chrome": ["google-chrome-stable"],
        "google": ["google-chrome-stable"],
        "chrome": ["google-chrome-stable"],
        "firefox": ["firefox"],
        "terminal": ["kitty"],
        "dolphin": ["dolphin"],
        "arquivos": ["dolphin"],
        "gerenciador de arquivos": ["dolphin"],
    }
    open_verbs = r"abra|abre|abrir|abri|inicie|iniciar|inicia|execute|executa|rode|rodar"
    for name, executable in applications.items():
        if re.search(rf"\b({open_verbs}) (o )?{re.escape(name)}\b", command) or command == name or (name in command and len(command.split()) <= 4):
            return CommandResult(f"Abrindo {name}.", executable)

    extra_result = extra_command_result(command)
    if extra_result:
        return extra_result

    script_commands = [
        ("resfriar normal", ["resfriar normal", "resfriar normalmente", "temperatura normal", "restaurar limites", "limites originais"], "Restaurando limites originais."),
        ("performace", ["performance", "performace", "modo desempenho", "modo performance", "preparar pc para jogar", "pc para jogar", "modo jogo"], "Preparando o PC para jogar."),
        ("atualizarlayout", ["atualizar layout", "atualize o layout", "sincronizar configurações", "sincronizar configuracoes", "recarregar configurações", "recarregar configuracoes"], "Sincronizando configurações."),
        ("3monitor", ["3 monitor", "terceiro monitor", "terceira monitora", "terceiro monitora", "terceiro monitores", "três monitor", "tres monitor", "três monitores", "tres monitores", "tela três", "tela tres", "notebook como monitor", "ativar monitor", "ative monitor", "ativar terceiro monitor", "ative terceiro monitor", "ativar terceira monitora", "ative terceira monitora", "ligar terceiro monitor", "ligue terceiro monitor", "usar terceiro monitor"], "Ativando notebook como monitor."),
        ("pc-remoto", ["pc remoto", "pc-remoto", "acesso remoto", "abrir pc remoto", "abrir pc no notebook", "conectar ao pc"], "Abrindo PC remoto."),
        ("autoclicker", ["autoclicker", "auto clicker", "auto clique", "cliques automáticos", "cliques automaticos", "iniciar cliques"], "Iniciando cliques automáticos."),
        ("consumo", ["consumo", "consumo do pc", "status do pc", "cpu ram", "cpu e ram", "temperatura do pc", "recursos do pc"], "Mostrando consumo do PC."),
        ("notebook", ["notebook", "abrir notebook", "conectar notebook", "ssh notebook", "acessar notebook"], "Abrindo conexão com notebook."),
        ("parar3monitor", ["parar 3 monitor", "parar terceiro monitor", "parar três monitor", "parar tres monitor", "parar monitores", "desativar 3 monitor", "desativar terceiro monitor", "desative o terceiro monitor", "desligar terceiro monitor", "desligue o terceiro monitor", "encerrar moonlight", "fechar moonlight", "desligar monitor remoto"], "Encerrando monitor remoto."),
        ("pesados", ["pesados", "processos pesados", "programas pesados", "processos que mais usam", "ver processos"], "Mostrando processos pesados."),
        ("padrao", ["padrão", "padrao", "modo padrão", "modo padrao", "sessão padrão", "sessao padrao", "restaurar sessão", "restaurar sessao", "modo normal"], "Restaurando sessão padrão."),
        ("resfriar", ["resfriar", "esfriar", "resfriar pc", "esfriar pc", "reduzir temperatura", "baixar temperatura", "reduzir calor", "modo frio"], "Reduzindo temperatura sem fechar aplicativos."),
        ("rede", ["rede", "status da rede", "status da internet", "ver minha rede", "internet", "conexão", "conexao"], "Mostrando status da rede."),
    ]
    if not hasattr(interpret_command, "_script_commands"):
        interpret_command._script_commands = expand_command_variations(script_commands)
    script_commands = interpret_command._script_commands
    if any(verb in command for verb in ("desativ", "deslig", "parar", "pare", "encerrar", "fechar")) and any(word in command for word in ("monitor", "monitora")):
        return CommandResult("Encerrando monitor remoto.", [str(Path.home() / ".local/bin/parar3monitor")])
    if not hasattr(interpret_command, "_script_patterns"):
        interpret_command._script_patterns = [
            (script, re.compile(rf"(?<!\w)(?:{'|'.join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True) if alias)})(?!\w)"), response)
            for script, aliases, response in script_commands
        ]
    matches = []
    for script, pattern, response in interpret_command._script_patterns:
        match = pattern.search(command)
        if match:
            alias = match.group(0)
            matches.append((len(alias.split()), len(alias), script, response))
    if matches:
        best_score = max((words, chars) for words, chars, _, _ in matches)
        best = [(script, response) for words, chars, script, response in matches if (words, chars) == best_score]
        if len({script for script, _ in best}) > 1:
            return CommandResult("Comando ambíguo. Diga novamente.")
        script, response = best[0]
        return CommandResult(response, [str(Path.home() / ".local/bin" / script)])

    if any(alias in command for alias in ("codex", "abrir codex", "iniciar codex", "abrir o codex")):
        return CommandResult("Abrindo Codex no modo YOLO.", ["kitty", "--directory", str(Path.home()), "-e", "codex", "--dangerously-bypass-approvals-and-sandbox"])
    if any(alias in command for alias in ("copilot", "abrir copilot", "iniciar copilot", "abrir o copilot")):
        return CommandResult("Abrindo Copilot.", ["kitty", "-e", "copilot"])

    if "que horas" in command or command == "horas":
        return CommandResult(f"Agora são {now:%H} horas e {now:%M} minutos.")
    if "que dia" in command or "qual a data" in command or command == "data":
        return CommandResult(f"Hoje é dia {now:%d/%m/%Y}.")
    if any(phrase in command for phrase in (
        "qual a mulher mais linda do mundo",
        "quem e a mulher mais linda do mundo",
        "qual e a mulher mais bonita do mundo",
        "quem e a mulher mais bonita do mundo",
        "a mulher mais linda do mundo",
        "com a mulher mais linda do mundo",
        "a mulher mais bonita do mundo",
        "com a mulher mais bonita do mundo",
        "qual a mulher mais chata do mundo",
        "quem e a mulher mais chata do mundo",
        "qual e a mulher mais chata do mundo",
        "a mulher mais chata do mundo",
        "com a mulher mais chata do mundo",
    )):
        return CommandResult("É a Julia Borges, com certeza.")
    if re.search(r"\b(aumente|aumentar|suba|mais) (o )?volume\b", command):
        return CommandResult("Aumentando volume.", ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%+", "-l", "1.0"])
    if re.search(r"\b(diminua|diminuir|abaixe|menos) (o )?volume\b", command):
        return CommandResult("Diminuindo volume.", ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%-"])
    if "silencie" in command or "mute" in command or "mudo" in command:
        return CommandResult("Alternando silêncio.", ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"])
    if any(variation in command for variation in MEDIA_VARIATIONS | MEDIA_ERROR_VARIATIONS) or re.search(r"\b(pause|pausar|pausa|continue|continuar|despause|despausar|retome|retomar|toque|tocar)\b.*\b(m[iu]sica|m[ií]dia|video|vídeo|som|reprodu[cç][aã]o|midia)\b", command):
        return CommandResult("Controlando reprodução.", ["playerctl", "play-pause"])
    if command in {"pause", "pausar", "pausa", "continue", "continuar", "despause", "despausar", "retome", "retomar"}:
        return CommandResult("Controlando reprodução.", ["playerctl", "play-pause"])
    if "proxima musica" in command:
        return CommandResult("Próxima música.", ["playerctl", "next"])
    if "musica anterior" in command:
        return CommandResult("Música anterior.", ["playerctl", "previous"])
    if "tela preta" in command:
        return CommandResult("Alternando tela preta.", [str(Path.home() / ".config/hypr/scripts/tela_preta")])
    if "o que voce pode fazer" in command or "seus comandos" in command:
        return CommandResult(
            "Posso abrir aplicativos e terminal, pesquisar, informar hora e data, controlar volume e música, "
            "ativar performance, atualizar layout, usar notebook como monitor, abrir PC remoto, iniciar autoclicker, "
            "mostrar consumo e processos pesados, conectar notebook, parar monitor remoto, restaurar sessão padrão, "
            "resfriar o PC, consultar rede, abrir Codex e abrir Copilot."
        )

    search = re.sub(r"^(pesquise|pesquisar|procure|procurar)( por)?\s+", "", command).strip()
    if search and search != command:
        return CommandResult(
            f"Pesquisando por {search}.",
            ["xdg-open", f"https://www.google.com/search?q={quote_plus(search)}"],
        )
    if command:
        return CommandResult(
            "Não entendi. Diga novamente.",
            None,
        )
    return CommandResult("Não entendi. Diga novamente.")


class AudioStream:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [
                "parec",
                f"--device={AUDIO_SOURCE}",
                "--client-name=Jarvis",
                "--stream-name=Wake word",
                "--format=s16le",
                f"--rate={SAMPLE_RATE}",
                "--channels=1",
                "--latency-msec=30",
                "--raw",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def read_frame(self) -> np.ndarray:
        if self.process.stdout is None:
            raise RuntimeError("Captura de áudio indisponível.")
        data = self.process.stdout.read(FRAME_BYTES)
        if len(data) != FRAME_BYTES:
            error = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
            raise RuntimeError(f"Captura de áudio encerrada: {error.strip()}")
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        safe_gain = min(INPUT_GAIN, 30000.0 / peak) if peak > 0 else INPUT_GAIN
        return np.clip(samples * safe_gain, -32768, 32767).astype(np.int16)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def enhance_command_audio(audio: np.ndarray) -> np.ndarray:
    """Keep speech band and normalize quiet far-field commands before Whisper."""
    if audio.size < 32:
        return audio
    samples = audio.astype(np.float32)
    sos = butter(4, [80, 6800], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    filtered = sosfilt(sos, samples)
    rms = float(np.sqrt(np.mean(filtered * filtered)))
    if rms > 0:
        filtered *= min(1.8, 5000.0 / rms)
    return np.clip(filtered, -32768, 32767).astype(np.int16)


class Jarvis:
    def __init__(self) -> None:
        wake_model = WAKE_MODELS_DIR / "hey_jarvis_v0.1.onnx"
        self.wake = WakeWordModel(
            wakeword_models=[str(wake_model)],
            inference_framework="onnx",
            melspec_model_path=str(WAKE_MODELS_DIR / "melspectrogram.onnx"),
            embedding_model_path=str(WAKE_MODELS_DIR / "embedding_model.onnx"),
        )
        small_path = self._find_whisper_model(WHISPER_SMALL_ROOT)
        medium_path = self._find_whisper_model(WHISPER_MEDIUM_ROOT)
        self.whisper_small = WhisperModel(str(small_path), device="cpu", compute_type="int8", cpu_threads=12)
        self.whisper_medium = WhisperModel(str(medium_path), device="cpu", compute_type="int8", cpu_threads=12)
        LOG.info("Modelos de transcrição: small=%s medium=%s", small_path, medium_path)
        self.voice: PiperVoice | None = PiperVoice.load(PIPER_MODEL)
        self.running = True

    @staticmethod
    def _find_whisper_model(model_root: Path) -> Path:
        snapshots = sorted(model_root.glob("*/model.bin"))
        if snapshots:
            return snapshots[-1].parent
        raise FileNotFoundError(f"Modelo Whisper ausente: {model_root}")

    def speak(self, text: str) -> None:
        text = add_honorific(text)
        if self.voice is None:
            self.voice = PiperVoice.load(PIPER_MODEL)
        with tempfile.NamedTemporaryFile(prefix="jarvis-", suffix=".wav", delete=False) as temporary:
            raw_path = Path(temporary.name)
        with tempfile.NamedTemporaryFile(prefix="jarvis-robot-", suffix=".wav", delete=False) as temporary:
            processed_path = Path(temporary.name)
        try:
            with wave.open(str(raw_path), "wb") as wav_file:
                self.voice.synthesize_wav(text, wav_file)
            effect = (
                "asetrate=22050*0.92,aresample=22050,atempo=1.15,"
                "highpass=f=110,lowpass=f=6500,"
                "acrusher=bits=12:mix=0.12:mode=lin,"
                "tremolo=f=24:d=0.06,"
                "acompressor=threshold=-20dB:ratio=2:attack=8:release=70,volume=0.96"
            )
            filtered = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_path), "-af", effect, str(processed_path)],
                check=False,
            )
            playback_path = processed_path if filtered.returncode == 0 else raw_path
            subprocess.run(["pw-play", str(playback_path)], check=False)
        finally:
            raw_path.unlink(missing_ok=True)
            processed_path.unlink(missing_ok=True)

    @staticmethod
    def set_listening(listening: bool) -> None:
        if listening:
            LISTENING_STATE.touch()
        else:
            LISTENING_STATE.unlink(missing_ok=True)
        subprocess.run(["pkill", "-RTMIN+8", "waybar"], check=False)

    @staticmethod
    def duck_app_audio() -> list[tuple[str, str]]:
        result = subprocess.run(["pactl", "list", "sink-inputs"], capture_output=True, text=True, check=False)
        saved: list[tuple[str, str]] = []
        blocks = re.split(r"\n(?=Sink Input #)", result.stdout)
        for block in blocks:
            match_id = re.search(r"Sink Input #(\d+)", block)
            if not match_id:
                continue
            # This virtual stream feeds the echo-cancel sink; muting it mutes Jarvis too.
            if "echo-cancel-playback" in block or "Echo-Cancel Playback" in block:
                continue
            stream_id = match_id.group(1)
            match = re.search(r"\n\s+Volume:.*?(\d+)%", block, flags=re.DOTALL)
            if not match:
                continue
            saved.append((stream_id, match.group(1)))
            subprocess.run(["pactl", "set-sink-input-volume", stream_id, "0%"], check=False)
        return saved

    @staticmethod
    def restore_app_audio(saved: list[tuple[str, str]]) -> None:
        for stream_id, volume in saved:
            subprocess.run(["pactl", "set-sink-input-volume", stream_id, f"{volume}%"], check=False)

    @staticmethod
    def record_command(
        stream: AudioStream,
        prefix: list[np.ndarray],
        *,
        silence_seconds: float = SILENCE_SECONDS,
        max_seconds: float = MAX_COMMAND_SECONDS,
    ) -> np.ndarray:
        frames = list(prefix)
        silent_frames = 0
        # Wake-word audio stays in prefix; wait for command speech after activation.
        speech_seen = False
        max_frames = math.ceil(max_seconds * SAMPLE_RATE / FRAME_SAMPLES)
        silence_limit = math.ceil(silence_seconds * SAMPLE_RATE / FRAME_SAMPLES)
        minimum_frames = math.ceil(MIN_COMMAND_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)

        for _ in range(max_frames):
            frame = stream.read_frame()
            frames.append(frame)
            rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
            if rms >= SILENCE_RMS:
                speech_seen = True
                silent_frames = 0
            elif speech_seen:
                silent_frames += 1
            if speech_seen and len(frames) >= minimum_frames and silent_frames >= silence_limit:
                break
        return np.concatenate(frames) if frames else np.empty(0, dtype=np.int16)

    @staticmethod
    def command_is_understood(text: str) -> bool:
        if not text:
            return False
        return not interpret_command(text).response.startswith("Não entendi")

    def transcribe_with(self, model: WhisperModel, audio: np.ndarray) -> str:
        audio = enhance_command_audio(audio)
        float_audio = audio.astype(np.float32) / 32768.0
        segments, _ = model.transcribe(
            float_audio,
            language="pt",
            beam_size=1,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 60, "speech_pad_ms": 120},
            condition_on_previous_text=False,
            initial_prompt="Jarvis.",
            hotwords="Jarvis Firefox Chrome terminal Codex Copilot",
            temperature=0.0,
            without_timestamps=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def transcribe(self, audio: np.ndarray) -> str:
        started = time.monotonic()
        small_text = self.transcribe_with(self.whisper_small, audio)
        stripped_small = re.sub(r"^(ei +)?jarvis\b", "", small_text, flags=re.IGNORECASE).strip(" ,.")
        if not stripped_small:
            LOG.info("Transcrição escolhida: small (somente ativação, %.2fs)", time.monotonic() - started)
            return small_text
        if self.command_is_understood(small_text):
            LOG.info("Transcrição escolhida: small (%.2fs)", time.monotonic() - started)
            return small_text
        medium_text = self.transcribe_with(self.whisper_medium, audio)
        LOG.info("Transcrição escolhida: medium (%.2fs)", time.monotonic() - started)
        return medium_text or small_text

    def execute(self, text: str) -> None:
        LOG.info("Comando reconhecido: %s", text)
        result = interpret_command(text)
        if result.action:
            try:
                detached(result.action)
            except OSError as error:
                LOG.error("Falha ao executar %s: %s", result.action, error)
                self.speak("Não consegui executar esse comando.")
                return
        self.speak(result.response)

    def run(self) -> None:
        self.set_listening(False)
        LOG.info("Jarvis pronto. Fonte=%s limite=%.2f", AUDIO_SOURCE, WAKE_THRESHOLD)
        while self.running:
            stream = AudioStream()
            try:
                self.wake.reset()
                pre_roll: deque[np.ndarray] = deque(maxlen=math.ceil(1.2 * SAMPLE_RATE / FRAME_SAMPLES))
                wake_window: deque[bool] = deque(maxlen=4)
                while self.running:
                    frame = stream.read_frame()
                    pre_roll.append(frame)
                    predictions = self.wake.predict(frame)
                    score = max(float(value) for value in predictions.values())
                    wake_window.append(score >= WAKE_THRESHOLD)
                    if sum(wake_window) < WAKE_CONFIRM_FRAMES:
                        continue
                    LOG.info("Ativação detectada. score=%.3f", score)
                    ducked_audio = self.duck_app_audio()
                    try:
                        self.set_listening(True)
                        try:
                            audio = self.record_command(stream, list(pre_roll))
                        finally:
                            self.set_listening(False)
                        text = self.transcribe(audio)
                        if text and normalize(re.sub(r"^(ei +)?jarvis\b", "", text, flags=re.IGNORECASE)).strip(" ,."):
                            if is_codex_write_request(text) and codex_write_payload(text) is None:
                                self.speak("Pode ditar.")
                                time.sleep(0.2)
                                try:
                                    self.set_listening(True)
                                    dictated_audio = self.record_command(stream, [], silence_seconds=0.75, max_seconds=15.0)
                                finally:
                                    self.set_listening(False)
                                dictated_text = remove_dictation_prompt(self.transcribe(dictated_audio))
                                if dictated_text:
                                    LOG.info("Ditado reconhecido: %s", dictated_text)
                                    self.execute(f"digite para mim: {dictated_text}")
                                else:
                                    self.speak("Não entendi. Diga novamente.")
                            else:
                                self.execute(text)
                        else:
                            LOG.info("Ativação ignorada: nenhum comando audível.")
                    finally:
                        self.restore_app_audio(ducked_audio)
                    time.sleep(0.5)
                    break
            finally:
                stream.close()


def self_test() -> None:
    required = [
        WAKE_MODELS_DIR / "hey_jarvis_v0.1.onnx",
        WAKE_MODELS_DIR / "melspectrogram.onnx",
        WAKE_MODELS_DIR / "embedding_model.onnx",
        PIPER_MODEL,
        PIPER_MODEL.with_suffix(".onnx.json"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Arquivos ausentes: " + ", ".join(missing))
    sources = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True, check=True).stdout
    if AUDIO_SOURCE not in sources:
        raise RuntimeError(f"Fonte {AUDIO_SOURCE} não encontrada.")
    assistant = Jarvis()
    test_text = assistant.transcribe(np.zeros(SAMPLE_RATE, dtype=np.int16))
    if test_text:
        LOG.warning("Silêncio produziu texto inesperado: %s", test_text)
    LOG.info("Autoteste concluído.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistente de voz local Jarvis")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    assistant = Jarvis()

    def stop(_signum: int, _frame: object) -> None:
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    try:
        assistant.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
