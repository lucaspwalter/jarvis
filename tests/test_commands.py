import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis import add_honorific, enhance_command_audio, expand_command_variations, interpret_command


class CommandTests(unittest.TestCase):
    def test_open_chrome(self):
        result = interpret_command("Jarvis, abra o Google Chrome")
        self.assertEqual(result.action, ["google-chrome-stable"])

    def test_hey_jarvis_open_firefox(self):
        result = interpret_command("Ei Jarvis, abra o Firefox")
        self.assertEqual(result.action, ["firefox"])

    def test_open_apps_as_spoken(self):
        self.assertEqual(interpret_command("abri o terminal").action, ["kitty"])
        self.assertEqual(interpret_command("abre fadi fox").action, ["firefox"])

    def test_dashboard_commands_variations(self):
        self.assertEqual(interpret_command("ative o modo desempenho").action[-1], "/home/lucas/.local/bin/performace")
        self.assertEqual(interpret_command("mostre os processos pesados").action[-1], "/home/lucas/.local/bin/pesados")
        self.assertEqual(interpret_command("conecte ao notebook").action[-1], "/home/lucas/.local/bin/notebook")
        self.assertEqual(interpret_command("abra o pc remoto").action[-1], "/home/lucas/.local/bin/pc-remoto")
        self.assertEqual(interpret_command("ative o terceiro monitor").action[-1], "/home/lucas/.local/bin/3monitor")
        self.assertEqual(interpret_command("desative o terceiro monitor").action[-1], "/home/lucas/.local/bin/parar3monitor")
        self.assertEqual(interpret_command("desatime terceiro monitor").action[-1], "/home/lucas/.local/bin/parar3monitor")
        self.assertEqual(interpret_command("ative terceira monitora").action[-1], "/home/lucas/.local/bin/3monitor")
        self.assertEqual(interpret_command("mostre o status da rede").action[-1], "/home/lucas/.local/bin/rede")
        self.assertEqual(interpret_command("abra o Codex").action, ["kitty", "-e", "codex"])

    def test_capabilities_includes_dashboard_commands(self):
        response = interpret_command("o que você pode fazer").response
        for capability in ("performance", "autoclicker", "processos pesados", "resfriar o PC"):
            self.assertIn(capability, response)

    def test_each_command_has_at_least_100_variations(self):
        commands = expand_command_variations([("performace", [], ""), ("3monitor", [], ""), ("parar3monitor", [], ""), ("resfriar", [], "")])
        self.assertTrue(all(len(aliases) >= 100 for _, aliases, _ in commands))

    def test_audio_enhancement_preserves_shape(self):
        import numpy as np
        audio = np.zeros(1600, dtype=np.int16)
        enhanced = enhance_command_audio(audio)
        self.assertEqual(enhanced.shape, audio.shape)

    def test_time(self):
        result = interpret_command("que horas são", datetime(2026, 8, 17, 16, 45))
        self.assertEqual(result.response, "Agora são 16 horas e 45 minutos.")

    def test_volume(self):
        result = interpret_command("aumente o volume")
        self.assertEqual(result.action[:2], ["wpctl", "set-volume"])

    def test_pause_media(self):
        for phrase in ("pause a mídia", "pausar o vídeo", "pause o som", "continue a música"):
            self.assertEqual(interpret_command(phrase).action, ["playerctl", "play-pause"])

    def test_search(self):
        result = interpret_command("pesquise por PipeWire no Linux")
        self.assertIn("pipewire", result.response)
        self.assertIn("pipewire+no+linux", result.action[-1])

    def test_honorific(self):
        self.assertEqual(add_honorific("Sistema online."), "Senhor, Sistema online.")
        self.assertEqual(add_honorific("Como posso ajudar, senhor?"), "Como posso ajudar, senhor?")


if __name__ == "__main__":
    unittest.main()
