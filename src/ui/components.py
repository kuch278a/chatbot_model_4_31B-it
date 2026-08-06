import os

UI_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HTML_PATH = os.path.join(UI_DIR, "chat.html")
STYLE_CSS_PATH = os.path.join(UI_DIR, "style.css")


def load_chat_html() -> str:
    """Reads and returns the content of chat.html."""
    if os.path.exists(CHAT_HTML_PATH):
        with open(CHAT_HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<!-- chat.html not found -->"


def load_style_css() -> str:
    """Reads and returns the content of style.css."""
    if os.path.exists(STYLE_CSS_PATH):
        with open(STYLE_CSS_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "/* style.css not found */"


class UIComponents:
    """UI Component manager for rendering frontend components."""

    @staticmethod
    def render_chat_page() -> str:
        return load_chat_html()

    @staticmethod
    def render_styles() -> str:
        return load_style_css()
