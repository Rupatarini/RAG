import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from .config import UPLOAD_FOLDER, LLM_MODEL_NAME
from .rag.vector_store import add_documents, query_index

app = Flask(__name__)

# -------------------------------
# CORS (Vercel frontend)
# -------------------------------
CORS(
    app,
    resources={r"/*": {"origins": ["https://rag-ruddy-six.vercel.app"]}}
)

# -------------------------------
# Logging
# -------------------------------
if os.getenv("FLASK_ENV") == "production":
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)

# -------------------------------
# Routes
# -------------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model": LLM_MODEL_NAME
    })

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        chunks = add_documents(filepath)
        os.remove(filepath)
        return jsonify({"chunks_indexed": chunks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question") if data else None

        if not question:
            return jsonify({"error": "No question provided"}), 400

        answer, sources = query_index(question)

        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        app.logger.exception("ASK endpoint failed")
        return jsonify({
            "error": "Internal error while answering the question"
        }), 500

# -------------------------------
# Local run (Gunicorn ignores)
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
