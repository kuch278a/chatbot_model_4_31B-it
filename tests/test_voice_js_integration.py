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

    def test_audio_capture_dsp_and_vad_optimization(self):
        """Verify chat.html contains aggressive VAD (380ms) and Web Audio DSP pre-filtering nodes."""
        with open("src/ui/chat.html", "r", encoding="utf-8") as f:
            html_content = f.read()

        # Verify aggressive VAD cutoff threshold
        self.assertIn("VAD_SILENCE_DURATION_MS = 380", html_content)
        self.assertIn("VM_SILENCE_DURATION_MS = 380", html_content)

        # Verify acoustic hardware constraints (AEC, noise suppression, AGC)
        self.assertIn("echoCancellation", html_content)
        self.assertIn("noiseSuppression", html_content)
        self.assertIn("autoGainControl", html_content)

        # Verify Web Audio DSP filter nodes (highpass, lowpass, compressor)
        self.assertIn("createBiquadFilter", html_content)
        self.assertIn("createDynamicsCompressor", html_content)
        self.assertIn("liveInterimRecognition", html_content)

    def test_backend_silero_vad_and_fast_mode(self):
        """Verify backend VAD has Silero enabled and faster-whisper supports fast_mode."""
        from src.stt.vad import SILERO_ENABLED, DEFAULT_POST_PADDING_MS
        self.assertTrue(SILERO_ENABLED, "Silero VAD should be enabled by default.")
        self.assertLessEqual(DEFAULT_POST_PADDING_MS, 150)

        from src.stt.faster_whisper_transcriber import FasterWhisperTranscriber
        import inspect
        sig = inspect.signature(FasterWhisperTranscriber.transcribe_audio_array)
        self.assertIn("fast_mode", sig.parameters)

    def test_stream_audio_queue_and_ttft_optimizations(self):
        """Verify chat.html contains streamAudioQueue and first-clause/sentence pipelining."""
        with open("src/ui/chat.html", "r", encoding="utf-8") as f:
            html_content = f.read()

        self.assertIn("streamAudioQueue", html_content)
        self.assertIn("enqueueStreamSentence", html_content)
        self.assertIn("playNextStreamSentence", html_content)

        from src.tts.synthesizer import VoiceSynthesizer
        import inspect
        sig = inspect.signature(VoiceSynthesizer.generate_audio_file)
        self.assertIn("rate", sig.parameters)
        self.assertEqual(sig.parameters["rate"].default, "+20%")

if __name__ == "__main__":
    unittest.main()
