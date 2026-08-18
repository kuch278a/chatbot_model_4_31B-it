import unittest
from flask import Flask
from src.ui.app import create_ui_app

class TestUIRoutes(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        create_ui_app(self.app)
        self.client = self.app.test_client()

    def test_index_route(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Amani AI", html)
        self.assertIn("brand-icon", html)
        self.assertIn("userInput", html)
        self.assertIn("sendBtn", html)
        self.assertIn("micBtn", html)
        self.assertIn("themeToggleBtn", html)
        self.assertIn("toggleTheme", html)
        self.assertIn("ሰላም፣ እንዴት ልርዳዎት?", html)
        # Verify Gemini logo & welcome landing page cards were removed
        self.assertNotIn("sparkle-icon", html)
        self.assertNotIn("suggestion-grid", html)

    def test_css_route(self):
        response = self.client.get("/style.css")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "text/css")
        css = response.get_data(as_text=True)
        self.assertIn("--gemini-bg", css)
        self.assertIn("data-theme", css)
        self.assertIn(".brand-icon", css)

if __name__ == "__main__":
    unittest.main()
