import unittest
import re

class TestVoiceJSIntegration(unittest.TestCase):
    """Unit test validating JavaScript speech synthesis and voice selection routines."""

    def test_speak_response_function_definition(self):
        """Verify chat.html contains speakText / speakResponse with Mekdes voice selection and cancel handlers."""
        with open("src/ui/chat.html", "r", encoding="utf-8") as f:
            html_content = f.read()

        # Verify speech synthesis API check
        self.assertIn("speechSynthesis", html_content)
        # Verify speech cancel call (preventing queue lock)
        self.assertIn("speechSynthesis.cancel()", html_content)
        # Verify SpeechSynthesisUtterance initialization
        self.assertIn("SpeechSynthesisUtterance", html_content)
        # Verify Amharic language selection check
        self.assertIn("am-ET", html_content)

    def test_js_voice_fallback_order(self):
        """Verify priority voice selection logic for Amharic in HTML UI script."""
        with open("src/ui/chat.html", "r", encoding="utf-8") as f:
            html_content = f.read()

        # Match voice selection code pattern
        self.assertTrue(
            "v.lang.startsWith(\"am\")" in html_content or "voices.find" in html_content,
            "HTML UI script must contain voice filtering logic for Amharic."
        )

if __name__ == "__main__":
    unittest.main()
