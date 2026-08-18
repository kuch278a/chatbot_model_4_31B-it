from flask import Flask
from src.ui.app import create_ui_app

app = Flask(__name__)
create_ui_app(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
