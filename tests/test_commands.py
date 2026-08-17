import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis import CODEX_WRITE_VARIATIONS, MEDIA_ERROR_VARIATIONS, MEDIA_VARIATIONS, add_honorific, codex_write_payload, enhance_command_audio, expand_command_variations, interpret_command, is_codex_write_request, remove_dictation_prompt


class CommandTests(unittest.TestCase):
    def test_open_chrome(self):
        result = interpret_command("Jarvis, abra o Google Chrome")
        self.assertEqual(result.action, ["google-chrome-stable"])
        self.assertEqual(interpret_command("abra o Google").action, ["google-chrome-stable"])

    def test_hey_jarvis_open_firefox(self):
        result = interpret_command("Ei Jarvis, abra o Firefox")
        self.assertEqual(result.action, ["firefox"])

    def test_open_apps_as_spoken(self):
        self.assertEqual(interpret_command("abri o terminal").action, ["kitty"])
        self.assertEqual(interpret_command("abre fadi fox").action, ["firefox"])
        self.assertEqual(interpret_command("abra o Dolphin").action, ["dolphin"])
        self.assertEqual(interpret_command("abra os arquivos").action, ["dolphin"])

    def test_extra_commands(self):
        self.assertEqual(interpret_command("mostre meu ip local").response, "Mostrando IP local.")
        self.assertEqual(interpret_command("por favor abra o GitHub").response, "Abrindo GitHub.")
        self.assertEqual(interpret_command("mostre os logs do Jarvis").response, "Mostrando logs do Jarvis.")
        self.assertEqual(interpret_command("reinicie o Jarvis").response, "Reiniciando Jarvis.")
        self.assertEqual(interpret_command("execute os testes").response, "Executando testes.")
        self.assertEqual(interpret_command("mostre os commits").response, "Mostrando commits.")
        self.assertEqual(interpret_command("abra o Steam").response, "Abrindo Steam.")
        self.assertEqual(interpret_command("silencie o microfone").response, "Silenciando microfone.")
        self.assertEqual(interpret_command("valide o Python").response, "Validando Python.")
        self.assertEqual(interpret_command("mostre a janela atual").response, "Mostrando janela atual.")

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
        self.assertEqual(interpret_command("abra o Codex").action, ["kitty", "--directory", "/home/lucas", "-e", "codex", "--dangerously-bypass-approvals-and-sandbox"])

    def test_capabilities_includes_dashboard_commands(self):
        response = interpret_command("o que você pode fazer").response
        for capability in ("performance", "autoclicker", "processos pesados", "resfriar o PC"):
            self.assertIn(capability, response)

    def test_each_command_has_at_least_100_variations(self):
        commands = expand_command_variations([("performace", [], ""), ("3monitor", [], ""), ("parar3monitor", [], ""), ("resfriar", [], "")])
        self.assertTrue(all(len(aliases) >= 2000 for _, aliases, _ in commands))
        self.assertGreaterEqual(len(MEDIA_VARIATIONS), 1000)
        self.assertGreaterEqual(len(MEDIA_ERROR_VARIATIONS), 1000)
        self.assertGreaterEqual(len(CODEX_WRITE_VARIATIONS), 1000)

    def test_write_into_existing_codex(self):
        result = interpret_command("Jarvis, digite para mim: abra o terminal amanhã")
        self.assertEqual(result.action[0].endswith("write_codex.py"), True)
        self.assertEqual(result.action[1], "abra o terminal amanhã")
        self.assertTrue(is_codex_write_request("Jarvis, digite para mim"))
        self.assertIsNone(codex_write_payload("Jarvis, digite para mim"))
        self.assertIsNone(codex_write_payload("Jarvis, escreva para mim."))
        self.assertEqual(remove_dictation_prompt("Senhor, pode digitar. Mande para o GitHub"), "Mande para o GitHub")

    def test_audio_enhancement_preserves_shape(self):
        import numpy as np
        audio = np.zeros(1600, dtype=np.int16)
        enhanced = enhance_command_audio(audio)
        self.assertEqual(enhanced.shape, audio.shape)

    def test_time(self):
        result = interpret_command("que horas são", datetime(2026, 8, 17, 16, 45))
        self.assertEqual(result.response, "Agora são 16 horas e 45 minutos.")

    def test_emylee(self):
        self.assertEqual(interpret_command("qual a mulher mais linda do mundo").response, "É a Emylee, com certeza.")
        self.assertEqual(interpret_command("com a mulher mais linda do mundo").response, "É a Emylee, com certeza.")

    def test_volume(self):
        result = interpret_command("aumente o volume")
        self.assertEqual(result.action[:2], ["wpctl", "set-volume"])

    def test_pause_media(self):
        for phrase in ("pause a mídia", "pausar o vídeo", "pause o som", "continue a música", "despausar a mídia", "despousar", "disposa do vídeo", "pousar", "pose o vídeo", "dispause o vídeo"):
            self.assertEqual(interpret_command(phrase).action, ["playerctl", "play-pause"])

    def test_search(self):
        result = interpret_command("pesquise por PipeWire no Linux")
        self.assertIn("pipewire", result.response)
        self.assertIn("pipewire+no+linux", result.action[-1])

    def test_unknown_command_asks_repeat(self):
        result = interpret_command("faça uma coisa desconhecida")
        self.assertIsNone(result.action)
        self.assertIn("Diga novamente", result.response)

    def test_honorific(self):
        self.assertEqual(add_honorific("Sistema online."), "Senhor, Sistema online.")
        self.assertEqual(add_honorific("Como posso ajudar, senhor?"), "Como posso ajudar, senhor?")


if __name__ == "__main__":
    unittest.main()
