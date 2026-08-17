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
WHISPER_MODEL = Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots"
LISTENING_STATE = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")) / "jarvis-listening"
AUDIO_SOURCE = "jarvis_mic"
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_280
FRAME_BYTES = FRAME_SAMPLES * 2
WAKE_THRESHOLD = 0.30
INPUT_GAIN = 1.60
SILENCE_RMS = 110.0
SILENCE_SECONDS = 0.50
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
    for script, aliases, response in script_commands:
        verbs_targets = expansions.get(script)
        if verbs_targets:
            verbs, targets = (part.split("|") for part in verbs_targets)
            generated = [normalize(f"{verb} {target}") for verb in verbs for target in targets]
            aliases = sorted(set(aliases + generated))
        expanded.append((script, aliases, response))
    return expanded


def interpret_command(text: str, now: datetime | None = None) -> CommandResult:
    command = normalize(text)
    command = re.sub(r"^(ei +)?jarvis\b", "", command).strip(" ,")
    command = command.replace("fadi fox", "firefox").replace("fe de foco", "firefox").replace("grom", "chrome")
    now = now or datetime.now()

    applications = {
        "google chrome": ["google-chrome-stable"],
        "chrome": ["google-chrome-stable"],
        "firefox": ["firefox"],
        "terminal": ["kitty"],
        "arquivos": ["dolphin"],
        "gerenciador de arquivos": ["dolphin"],
    }
    open_verbs = r"abra|abre|abrir|abri|inicie|iniciar|inicia|execute|executa|rode|rodar"
    for name, executable in applications.items():
        if re.search(rf"\b({open_verbs}) (o )?{re.escape(name)}\b", command) or command == name:
            return CommandResult(f"Abrindo {name}.", executable)

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
    script_commands = expand_command_variations(script_commands)
    if any(verb in command for verb in ("desativ", "deslig", "parar", "pare", "encerrar", "fechar")) and any(word in command for word in ("monitor", "monitora")):
        return CommandResult("Encerrando monitor remoto.", [str(Path.home() / ".local/bin/parar3monitor")])
    for script, aliases, response in script_commands:
        if any(alias in command for alias in aliases):
            return CommandResult(response, [str(Path.home() / ".local/bin" / script)])

    if any(alias in command for alias in ("codex", "abrir codex", "iniciar codex", "abrir o codex")):
        return CommandResult("Abrindo Codex.", ["kitty", "-e", "codex"])
    if any(alias in command for alias in ("copilot", "abrir copilot", "iniciar copilot", "abrir o copilot")):
        return CommandResult("Abrindo Copilot.", ["kitty", "-e", "copilot"])

    if "que horas" in command or command == "horas":
        return CommandResult(f"Agora são {now:%H} horas e {now:%M} minutos.")
    if "que dia" in command or "qual a data" in command or command == "data":
        return CommandResult(f"Hoje é dia {now:%d/%m/%Y}.")
    if re.search(r"\b(aumente|aumentar|suba|mais) (o )?volume\b", command):
        return CommandResult("Aumentando volume.", ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%+", "-l", "1.0"])
    if re.search(r"\b(diminua|diminuir|abaixe|menos) (o )?volume\b", command):
        return CommandResult("Diminuindo volume.", ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", "5%-"])
    if "silencie" in command or "mute" in command or "mudo" in command:
        return CommandResult("Alternando silêncio.", ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"])
    if any(term in command for term in ("pause a musica", "pausar musica", "continue a musica", "toque a musica")):
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
            "Ainda não conheço esse comando. Vou pesquisar.",
            ["xdg-open", f"https://www.google.com/search?q={quote_plus(command)}"],
        )
    return CommandResult("Não entendi o comando.")


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
        self.whisper = WhisperModel(
            str(self._find_whisper_model()), device="cpu", compute_type="int8", cpu_threads=10
        )
        self.voice: PiperVoice | None = None
        self.running = True

    @staticmethod
    def _find_whisper_model() -> Path:
        snapshots = sorted(WHISPER_MODEL.glob("*/model.bin"))
        if not snapshots:
            raise FileNotFoundError("Modelo faster-whisper-small não encontrado.")
        return snapshots[-1].parent

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
                "asetrate=22050*0.90,aresample=22050,atempo=1.388889,"
                "highpass=f=110,lowpass=f=6500,"
                "chorus=0.6:0.8:15|22:0.32|0.22:0.30|0.22:1.8|2.2,"
                "flanger=delay=2:depth=2:regen=20:width=45:speed=0.35,"
                "acrusher=bits=10:mix=0.28:mode=lin,"
                "tremolo=f=32:d=0.12,"
                "aecho=0.8:0.4:24:0.18,"
                "acompressor=threshold=-20dB:ratio=3:attack=8:release=70,volume=0.92"
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
    def record_command(stream: AudioStream, prefix: list[np.ndarray]) -> np.ndarray:
        frames = list(prefix)
        silent_frames = 0
        # Wake-word audio stays in prefix; wait for command speech after activation.
        speech_seen = False
        max_frames = math.ceil(MAX_COMMAND_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)
        silence_limit = math.ceil(SILENCE_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)
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

    def transcribe(self, audio: np.ndarray) -> str:
        audio = enhance_command_audio(audio)
        float_audio = audio.astype(np.float32) / 32768.0
        segments, _ = self.whisper.transcribe(
            float_audio,
            language="pt",
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250, "speech_pad_ms": 220},
            condition_on_previous_text=False,
            initial_prompt="Comandos em português: Jarvis, que horas são? Jarvis, abra o terminal. Jarvis, ative o terceiro monitor.",
            hotwords="Jarvis que horas são abra abrir ative desative terceiro monitor terminal Firefox Chrome Codex volume",
            temperature=0.0,
            without_timestamps=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

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
                while self.running:
                    frame = stream.read_frame()
                    pre_roll.append(frame)
                    predictions = self.wake.predict(frame)
                    score = max(float(value) for value in predictions.values())
                    if score < WAKE_THRESHOLD:
                        continue
                    LOG.info("Ativação detectada. score=%.3f", score)
                    ducked_audio = self.duck_app_audio()
                    try:
                        self.set_listening(True)
                        try:
                            audio = self.record_command(stream, list(pre_roll))
                        finally:
                            self.set_listening(False)
                        stream.close()
                        text = self.transcribe(audio)
                        if text:
                            self.execute(text)
                        else:
                            self.speak("Não entendi o comando.")
                    finally:
                        self.restore_app_audio(ducked_audio)
                    time.sleep(0.1)
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
