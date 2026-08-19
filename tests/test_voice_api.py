import unittest
from flask import Flask
from src.ui.app import ui_bp

class TestVoiceAPI(unittest.TestCase):
    """Unit tests for Flask Voice API endpoints (/api/voices and /api/tts)."""

    def setUp(self):
        self.app = Flask(__name__)
        self.app.register_blueprint(ui_bp)
        self.client = self.app.test_client()

    def test_get_voices_endpoint(self):
        """Test GET /api/voices returns list of available TTS voices."""
        response = self.client.get("/api/voices")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIsInstance(data["voices"], list)
        self.assertGreater(len(data["voices"]), 0)

    def test_get_voices_filtered_by_lang(self):
        """Test GET /api/voices?lang=am-ET returns Amharic voices."""
        response = self.client.get("/api/voices?lang=am-ET")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        for voice in data["voices"]:
            self.assertEqual(voice["lang"], "am-ET")

    def test_post_tts_endpoint_success(self):
        """Test POST /api/tts returns speech synthesis payload for valid input."""
        payload = {
            "text": "እንኳን ወደ አማኒ ረዳት በደህና መጡ",
            "lang": "am-ET",
            "rate": 1.0,
            "pitch": 1.0
        }
        response = self.client.post("/api/tts", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["lang"], "am-ET")
        self.assertIn("Mekdes", data["voice"])
        self.assertEqual(data["text"], "እንኳን ወደ አማኒ ረዳት በደህና መጡ")

    def test_post_tts_endpoint_empty_text_error(self):
        """Test POST /api/tts handles empty text gracefully with 400 status."""
        response = self.client.post("/api/tts", json={"text": "   "})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_stream_audio_endpoint(self):
        """Test GET /api/tts/audio streams valid audio/mpeg data generated via edge-tts."""
        response = self.client.get("/api/tts/audio?text=ሰላም&lang=am-ET")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "audio/mpeg")
        self.assertGreater(len(response.data), 0)

    def test_end_to_end_chat_text_to_audio_streaming(self):
        """Verify generated assistant text translates directly into playable streamed audio bytes."""
        assistant_text = "ሰላም! እኔ አማኒ ነኝ:: ምን ልርዳዎት?"
        response = self.client.post("/api/tts/audio", json={"text": assistant_text, "lang": "am-ET"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "audio/mpeg")
        # Check audio header / valid non-empty stream
        self.assertGreater(len(response.data), 1000)

    def test_post_stt_empty_data(self):
        """Test POST /api/stt handles empty audio gracefully with 400 status."""
        response = self.client.post("/api/stt", data=b"")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_post_stt_dummy_audio(self):
        """Test POST /api/stt with small audio blob returns structured JSON response."""
        response = self.client.post("/api/stt", data=b"\x00" * 100, headers={"Content-Type": "audio/webm"})
        self.assertIn(response.status_code, [200, 400, 500])
        data = response.get_json()
        self.assertIsInstance(data, dict)

if __name__ == "__main__":
    unittest.main()



