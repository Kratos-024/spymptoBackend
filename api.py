from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv('GOOGLE_API_KEY')

if API_KEY:
    try:
        from model import process_symptom, initialize_llm
        initialize_llm(API_KEY)
        print("Gemini model initialized.")
    except ImportError:
        print(" Could not import 'model.py'. Is it in the same folder?")
    except Exception as e:
        print(f"Error initializing model: {e}")
else:
    print(" GOOGLE_API_KEY not found in .env file.")

@app.route('/classify-symptom', methods=['POST'])
def classify_symptom_api():
    try:
        data = request.get_json()

        symptom = data.get('symptom', '').strip() if data else ''
        if not symptom:
            return jsonify({"error": "Please provide a symptom."}), 400

        result = process_symptom(symptom)
        result["hospital"] = "XYZ Hospital"
        result["status"] = "success"
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500
@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "service": "Medical Symptom Classifier",
        "hospital": "XYZ Hospital"
    })
if __name__ == '__main__':
    print(" Starting Medical Symptom Classifier")
    app.run(debug=True)
