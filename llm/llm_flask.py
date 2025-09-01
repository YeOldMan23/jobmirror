# app.py
from flask import Flask, request, jsonify
from llama_cpp import Llama
import argparse
import os

from download_llama import install_model

# Get env
MODEL_ID = os.getenv("MODEL_ID", "TheBloke/Mistral-7B-Instruct-v0.2-GGUF")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
CACHE_DIR = os.path.join(os.getcwd(), "model_cache")

print("---Loading Model---")
llm_model = install_model(CACHE_DIR,
                        MODEL_ID,
                        MODEL_NAME)
    

# Initialize the Flask app
print("---Running Application---")
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate_text(llm : Llama):
    data = request.json
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Run the model inference
    output = llm(prompt, max_tokens=10000, stop=["Qwen2.5-Coder"], echo=True)
    generated_text = output["choices"][0]["text"]

    return jsonify({"text": generated_text})
