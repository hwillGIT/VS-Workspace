"""
API routes for code generation.

Defines the /generate_code endpoint which accepts a prompt
and returns generated code.

Author: Roo (auto-generated)
"""

from flask import Blueprint, request, jsonify
from webtester.backend.webtester_api.services.code_generation_service import generate_code

# Create a Blueprint for code generation routes
generation_bp = Blueprint('generation_bp', __name__)

@generation_bp.route('/generate_code', methods=['POST'])
def generate_code_route():
    """
    POST endpoint to generate code from a prompt.

    Expects JSON payload:
    {
        "prompt": "Describe the code you want"
    }

    Returns JSON response:
    {
        "generated_code": "The generated code as a string"
    }
    """
    try:
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        prompt = data['prompt']
        generated_code = generate_code(prompt)

        return jsonify({"generated_code": generated_code}), 200

    except Exception as e:
        # Log the error in real implementation
        return jsonify({"error": str(e)}), 500