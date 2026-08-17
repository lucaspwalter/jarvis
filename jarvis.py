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
WAKE_THRESHOLD = 0.35
SILENCE_RMS = 180.0
SILENCE_SECONDS = 1.1
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
    return f"{body}, senhor{punctuation}"


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


def interpret_command(text: str, now: datetime | None = None) -> CommandResult:
    command = normalize(text)
    command = re.sub(r"^(ei +)?jarvis\b", "", command).strip(" ,")
    now = now or datetime.now()

    applications = {
        "google chrome": ["google-chrome-stable"],
        "chrome": ["google-chrome-stable"],
        "firefox": ["firefox"],
        "terminal": ["kitty"],
        "arquivos": ["dolphin"],
        "gerenciador de arquivos": ["dolphin"],
    }
    for name, executable in applications.items():
        if re.search(rf"\b(abra|abrir|inicie|iniciar) (o )?{re.escape(name)}\b", command):
            return CommandResult(f"Abrindo {name}.", executable)

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
        return CommandResult("Posso abrir aplicativos, pesquisar, informar hora e data, controlar volume, música e tela preta.")

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
                "--latency-msec=80",
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
        return np.frombuffer(data, dtype=np.int16).copy()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


class Jarvis:
    def __init__(self) -> None:
        wake_model = WAKE_MODELS_DIR / "hey_jarvis_v0.1.onnx"
        self.wake = WakeWordModel(
            wakeword_models=[str(wake_model)],
            inference_framework="onnx",
            melspec_model_path=str(WAKE_MODELS_DIR / "melspectrogram.onnx"),
            embedding_model_path=str(WAKE_MODELS_DIR / "embedding_model.onnx"),
        )
        self.whisper: WhisperModel | None = None
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
                "flanger=delay=2:depth=1.5:regen=12:width=35:speed=0.3,"
                "acrusher=bits=12:mix=0.14:mode=lin,"
                "aecho=0.8:0.4:26:0.13,"
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
    def record_command(stream: AudioStream, prefix: list[np.ndarray]) -> np.ndarray:
        frames = list(prefix)
        silent_frames = 0
        speech_seen = True
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
        if self.whisper is None:
            whisper_path = self._find_whisper_model()
            self.whisper = WhisperModel(str(whisper_path), device="cpu", compute_type="int8", cpu_threads=6)
        float_audio = audio.astype(np.float32) / 32768.0
        segments, _ = self.whisper.transcribe(
            float_audio,
            language="pt",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt="Comando em português para Jarvis, assistente do computador.",
            hotwords="Jarvis Chrome Firefox terminal arquivos volume música tela preta",
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
                pre_roll: deque[np.ndarray] = deque(maxlen=math.ceil(3 * SAMPLE_RATE / FRAME_SAMPLES))
                while self.running:
                    frame = stream.read_frame()
                    pre_roll.append(frame)
                    predictions = self.wake.predict(frame)
                    score = max(float(value) for value in predictions.values())
                    if score < WAKE_THRESHOLD:
                        continue
                    LOG.info("Ativação detectada. score=%.3f", score)
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
                    time.sleep(0.4)
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
