from flask import Flask, send_file, jsonify, send_from_directory, request
import os
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
from generate_video import create_promotional_video

load_dotenv() # Load Keys from .env

app = Flask(__name__, static_folder='.', static_url_path='')

def get_ai_client():
    """Returns an AI client preferring OpenRouter if available"""
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=or_key,
        )
    return None

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json or {}
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"status": "error", "message": "No message provided"}), 400

        client = get_ai_client()
        if client:
            # Use OpenRouter
            response = client.chat.completions.create(
                model="nvidia/nemotron-nano-12b-v2-vl:free",
                messages=[
                    {"role": "system", "content": "You are the 'Shivaratri Video Assistant'. Help the user create a cinematic video. Keep responses spiritual, helpful, and concise (max 3 sentences)."},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=300
            )
            return jsonify({"status": "success", "reply": response.choices[0].message.content})
        
        # Fallback to local Gemini if OpenRouter fails/missing
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"status": "error", "message": "API Key missing"}), 500
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"User says: '{user_message}'. Act as Shivaratri Video Assistant. Max 3 sentences."
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(max_output_tokens=300))
        return jsonify({"status": "success", "reply": response.text})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get data from request
        data = request.json or {}
        theme = data.get('theme', 'divine_blue')
        opening_text = data.get('openingText', 'The Night of Cosmic Awakening...')
        closing_text = data.get('closingText', 'Happy Maha Shivaratri')
        output_format = data.get('format', 'mp4')
        resolution = data.get('resolution', '1080p')
        prompt = data.get('prompt', '')

        # Vercel compatibility: Use /tmp for writing
        output_filename = f"maha_shivaratri_concept.{output_format}"
        if os.environ.get('VERCEL'):
            output_path = os.path.join('/tmp', output_filename)
        else:
            output_path = output_filename

        # Run the generation script with parameters
        create_promotional_video(
            theme=theme, 
            opening_text=opening_text, 
            closing_text=closing_text, 
            output_format=output_format, 
            resolution=resolution,
            prompt=prompt,
            output_path=output_path
        )
        
        if os.path.exists(output_path):
            return jsonify({"status": "success", "file": output_filename})
        else:
            return jsonify({"status": "error", "message": "File not generated"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    # Check current dir first, then /tmp for Vercel
    if os.path.exists(filename):
        return send_from_directory('.', filename, as_attachment=True)
    
    tmp_path = os.path.join('/tmp', filename)
    if os.path.exists(tmp_path):
        return send_from_directory('/tmp', filename, as_attachment=True)
        
    return jsonify({"status": "error", "message": "File not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
