from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
    app.config["MODEL_DIR"] = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
