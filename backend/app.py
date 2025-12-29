import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# -------------------------------
# Relative imports (OPTION 2 SAFE)
# -------------------------------
from .config import UPLOAD_FOLDER, LLM_MODEL_NAME
from .rag.vector_store import add_documents, query_index, rewrite_answer

# -------------------------------
# Flask App Initialization
# -------------------------------
app = Flask(__name__)

# -------------------------------
# CORS CONFIG (Vercel Frontend)
# -------------------------------
CORS(
    app,
    resources={
        r"/*": {
            "origins": ["https://rag-ruddy-six.vercel.app"]
        }
    },
    supports_credentials=True
)

# -------------------------------
# Logging (Production Safe)
# -------------------------------
if os.getenv("FLASK_ENV") == "production":
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)

# -------------------------------
# Global Error Handler (500)
# -------------------------------
@app.errorhandler(500)
def internal_error(error):
    app.logger.error(
        "Unhandled Exception: %s",
        error,
        exc_info=True
    )
    return jsonify({
        "error": "An internal server error occurred. Please try again later."
    }), 500

# -------------------------------
# Upload Configuration
# -------------------------------
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"pdf", "txt"}

def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# -------------------------------
# ROUTES
# -------------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "RAG backend running",
        "model": LLM_MODEL_NAME
    })

# -------- Upload & Index --------
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file_saved = False

    try:
        file.save(filepath)
        file_saved = True

        chunks_indexed = add_documents(filepath)

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            "message": "File processed successfully",
            "chunks_indexed": chunks_indexed
        })

    except Exception as e:
        app.logger.error(
            "Indexing failed for file %s: %s",
            filename,
            str(e),
            exc_info=True
        )

        if file_saved and os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            "error": "Indexing failed due to an internal server issue."
        }), 500

# -------- Ask Question --------
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json(silent=True)
    question = data.get("question") if data else None

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer, sources = query_index(question)
        return jsonify({
            "answer": answer,
            "sources": sources
        })

    except Exception as e:
        app.logger.error(
            'Query failed for question "%s": %s',
            question,
            str(e),
            exc_info=True
        )
        return jsonify({
            "error": "Failed to retrieve answer due to an internal server issue."
        }), 500

# -------- Rewrite Answer --------
@app.route("/rewrite", methods=["POST"])
def rewrite_answer_endpoint():
    data = request.get_json(silent=True)
    original_answer = data.get("answer") if data else None
    style_request = data.get("style") if data else None

    if not original_answer or not style_request:
        return jsonify({
            "error": "Missing original answer or style request"
        }), 400

    try:
        new_answer = rewrite_answer(original_answer, style_request)
        return jsonify({
            "original_answer": original_answer,
            "style_request": style_request,
            "new_answer": new_answer
        })

    except Exception as e:
        app.logger.error(
            'Rewrite failed for style "%s": %s',
            style_request,
            str(e),
            exc_info=True
        )
        return jsonify({
            "error": "Failed to rewrite answer due to an internal server issue."
        }), 500

# -------------------------------
# Local Dev (Gunicorn ignores this)
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
