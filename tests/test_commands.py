import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jarvis import interpret_command


class CommandTests(unittest.TestCase):
    def test_open_chrome(self):
        result = interpret_command("Jarvis, abra o Google Chrome")
        self.assertEqual(result.action, ["google-chrome-stable"])

    def test_hey_jarvis_open_firefox(self):
        result = interpret_command("Ei Jarvis, abra o Firefox")
        self.assertEqual(result.action, ["firefox"])

    def test_time(self):
        result = interpret_command("que horas são", datetime(2026, 8, 17, 16, 45))
        self.assertEqual(result.response, "Agora são 16 horas e 45 minutos.")

    def test_volume(self):
        result = interpret_command("aumente o volume")
        self.assertEqual(result.action[:2], ["wpctl", "set-volume"])

    def test_search(self):
        result = interpret_command("pesquise por PipeWire no Linux")
        self.assertIn("pipewire", result.response)
        self.assertIn("pipewire+no+linux", result.action[-1])


if __name__ == "__main__":
    unittest.main()
