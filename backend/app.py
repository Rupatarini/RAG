import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback 
import logging # <--- NEW: Standard Python logging module

# FINAL FIX: Correct relative imports for sibling modules
from .config import UPLOAD_FOLDER, LLM_MODEL_NAME
# Assuming rewrite_answer is also imported from vector_store now for completeness
from .rag.vector_store import add_documents, query_index, rewrite_answer 

app = Flask(__name__)
CORS(app) 

# --- PRODUCTION CONFIGURATION ---

# 1. Set logger format
if os.getenv('FLASK_ENV') == 'production':
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)
    
# 2. Global 500 Error Handler (Catches exceptions that escape the route handlers)
@app.errorhandler(500)
def internal_error(error):
    # Log the full traceback to the application logs (not the client)
    app.logger.error('Server Error: %s', (error), exc_info=True)
    # Return a generic, safe message to the user
    return jsonify({
        'error': 'An unexpected server error occurred. Please try again later.'
    }), 500

# --------------------------------

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        file_saved = False

        try:
            file.save(filepath)
            file_saved = True

            chunks_indexed = add_documents(filepath)
            os.remove(filepath)
            
            return jsonify({
                'message': 'File processed successfully', 
                'chunks_indexed': chunks_indexed
            })
            
        except Exception as e:
            # --- PRODUCTION FIX: Log the error, do NOT expose traceback ---
            app.logger.error('Indexing failed for file %s: %s', filename, str(e), exc_info=True)
            
            if file_saved and os.path.exists(filepath):
                os.remove(filepath)
                
            # Return a generic, safe 500 message
            return jsonify({'error': 'Indexing failed due to an internal server issue.'}), 500

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        answer, sources = query_index(question)
        
        return jsonify({
            'answer': answer,
            'sources': sources
        })
    except Exception as e:
        # --- PRODUCTION FIX: Log the error, do NOT expose traceback ---
        app.logger.error('Query failed for question "%s": %s', question, str(e), exc_info=True)
        
        # Return a generic, safe 500 message
        return jsonify({'error': 'Failed to retrieve answer due to an internal server issue.'}), 500

# Assuming /rewrite is included in your production requirements
@app.route('/rewrite', methods=['POST'])
def rewrite_answer_endpoint():
    data = request.get_json()
    original_answer = data.get('answer')
    style_request = data.get('style')
    
    if not original_answer or not style_request:
        return jsonify({'error': 'Missing original answer or style request'}), 400

    try:
        new_answer = rewrite_answer(original_answer, style_request)
        
        return jsonify({
            'original_answer': original_answer,
            'style_request': style_request,
            'new_answer': new_answer
        })
        
    except Exception as e:
        app.logger.error('Rewrite failed for style "%s": %s', style_request, str(e), exc_info=True)
        return jsonify({'error': 'Failed to rewrite answer due to an internal server issue.'}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)


