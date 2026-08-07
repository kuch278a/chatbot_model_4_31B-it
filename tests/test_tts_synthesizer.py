import unittest
from src.tts.synthesizer import VoiceSynthesizer

class TestVoiceSynthesizer(unittest.TestCase):
    """Unit tests for VoiceSynthesizer class verifying voice selection and text preparation."""

    def setUp(self):
        self.synthesizer = VoiceSynthesizer()

    def test_get_available_voices(self):
        """Test retrieving all voices and filtering by language locale."""
        voices = self.synthesizer.get_available_voices()
        self.assertGreater(len(voices), 0)

        amharic_voices = self.synthesizer.get_available_voices(lang="am-ET")
        self.assertTrue(all(v["lang"] == "am-ET" for v in amharic_voices))
        self.assertGreater(len(amharic_voices), 0)

    def test_voice_selection_mekdes_exact(self):
        """Test exact string match for Microsoft Mekdes Neural voice."""
        selected = self.synthesizer.select_voice(
            lang="am-ET",
            preferred_name="Microsoft Mekdes Online (Natural) - Amharic (Ethiopia)"
        )
        self.assertEqual(selected["name"], "Microsoft Mekdes Online (Natural) - Amharic (Ethiopia)")

    def test_voice_selection_mekdes_fallback(self):
        """Test fallback prioritizes Amharic Mekdes neural voice when requesting am-ET."""
        selected = self.synthesizer.select_voice(lang="am-ET")
        self.assertIn("Mekdes", selected["name"])
        self.assertEqual(selected["lang"], "am-ET")

    def test_clean_text_for_speech(self):
        """Test markdown, HTML, and code block formatting removal prior to speech."""
        raw_text = "<b>ሰላም!</b> **አማኒ** ነኝ:: `print('hello')` ```code block```"
        cleaned = VoiceSynthesizer.clean_text_for_speech(raw_text)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("`", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertIn("ሰላም!", cleaned)
        self.assertIn("አማኒ", cleaned)

    def test_synthesize_metadata(self):
        """Test synthesis parameter generation for Amharic speech."""
        res = self.synthesizer.synthesize(text="ሰላም፥ እንዴት ነዎት?", lang="am-ET", rate=1.0)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["lang"], "am-ET")
        self.assertIn("Mekdes", res["voice"])
        self.assertEqual(res["wpm"], 117)

if __name__ == "__main__":
    unittest.main()
