"""
Main Flask application for WebTester backend API.

Initializes the app, registers Blueprints, and runs the server.

Author: Roo (auto-generated)
"""

from flask import Flask
from flask_cors import CORS
from .routes.generation_routes import generation_bp

def create_app():
    """
    Create and configure the Flask app.
    """
    app = Flask(__name__)

    # Enable CORS for all routes (adjust in production)
    CORS(app)

    # Register Blueprints
    app.register_blueprint(generation_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    # Run the app on localhost:5000 by default
    app.run(host='0.0.0.0', port=5000, debug=True)