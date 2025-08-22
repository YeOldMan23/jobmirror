# app.py
from flask import Flask, request, jsonify
from llama_cpp import Llama

# Initialize the Flask app
app = Flask(__name__)

# Load the model globally to avoid reloading on each request
# Make sure 'model.gguf' is in the same directory as this script
llm = Llama(model_path="./model.gguf")

@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.json
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Run the model inference
    output = llm(prompt, max_tokens=10000, stop=["Qwen2.5-Coder"], echo=True)
    generated_text = output["choices"][0]["text"]

    return jsonify({"text": generated_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)