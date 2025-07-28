from flask import Blueprint, request, jsonify
from webtester_api.services.code_generation_service import CodeGenerationService

generation_bp = Blueprint('generation', __name__)
code_gen_service = CodeGenerationService()

@generation_bp.route('/code', methods=['POST'])
def generate_code():
    data = request.json
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    try:
        generated_code = code_gen_service.generate_code(prompt)
        return jsonify({'generated_code': generated_code})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@generation_bp.route('/text', methods=['POST'])
def generate_text():
    data = request.json
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    try:
        generated_text = code_gen_service.generate_text(prompt)
        return jsonify({'generated_text': generated_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500