from flask import Flask
from webtester_api.routes.generation_routes import generation_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(generation_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)