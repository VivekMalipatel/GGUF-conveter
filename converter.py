import os
import threading
import logging
import torch
from safetensors.torch import load_file, save_file
from flask import Flask, request, jsonify, send_file
from huggingface_hub import snapshot_download
import subprocess
from werkzeug.utils import secure_filename

# Flask App Setup
app = Flask(__name__)

# 📌 Configurations
BASE_DIR = "./models"
DOWNLOADS_DIR = "./downloads"
LOG_FILE = "app.log"
LLAMA_CONVERTER_SCRIPT = "./llama.cpp/convert_hf_to_gguf.py"
LLAMA_QUANTIZE_SCRIPT = "./llama.cpp/llama-quantize"

# Ensure directories exist
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dictionary to track processing tasks
processing_tasks = {}

# GGUF Quantization Enum
GGUF_QUANTIZATION_TYPES = [
    "F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q2_K", "Q3_K", "Q3_K_S", "Q3_K_M", "Q3_K_L"
]

# Function to download the model from Hugging Face
def download_model(repo_id, local_dir):
    try:
        logging.info(f"Downloading model {repo_id}...")
        snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False, revision="main")
        logging.info(f"Download complete: {local_dir}")
    except Exception as e:
        logging.error(f"Error downloading model: {str(e)}")
        raise

# Function to convert SafeTensors to PyTorch bin (if needed)
def convert_safetensors_to_pytorch(model_dir):
    try:
        for filename in os.listdir(model_dir):
            if filename.endswith(".safetensors"):
                safetensor_path = os.path.join(model_dir, filename)
                pytorch_path = safetensor_path.replace(".safetensors", ".bin")

                logging.info(f"Converting {safetensor_path} to {pytorch_path}...")

                # Load SafeTensors
                tensors = load_file(safetensor_path)

                # Convert and save as PyTorch bin
                torch.save(tensors, pytorch_path)

                logging.info(f"Converted SafeTensors to PyTorch: {pytorch_path}")
        
    except Exception as e:
        logging.error(f"Error converting SafeTensors: {str(e)}")
        raise

# Function to convert the model to GGUF
def convert_to_gguf(local_dir, output_file, quantization_type):
    try:
        logging.info(f"Converting {local_dir} to GGUF...")

        # Ensure the llama.cpp converter exists
        if not os.path.exists(LLAMA_CONVERTER_SCRIPT):
            raise FileNotFoundError(f"Converter script not found: {LLAMA_CONVERTER_SCRIPT}")

        # Run conversion command
        command = [
            "python3", LLAMA_CONVERTER_SCRIPT,
            local_dir, "--outfile", output_file, "--outtype", quantization_type
        ]
        subprocess.run(command, check=True)
        
        logging.info(f"GGUF model saved at {output_file}")
    except Exception as e:
        logging.error(f"Error converting to GGUF: {str(e)}")
        raise

# Function to quantize GGUF model
def quantize_gguf_model(input_file, output_file, quantization_type):
    try:
        logging.info(f"Quantizing {input_file} to {quantization_type} format...")

        # Ensure the llama-quantize script exists
        if not os.path.exists(LLAMA_QUANTIZE_SCRIPT):
            raise FileNotFoundError(f"Quantization script not found: {LLAMA_QUANTIZE_SCRIPT}")

        # Run quantization command
        command = [LLAMA_QUANTIZE_SCRIPT, input_file, output_file, quantization_type]
        subprocess.run(command, check=True)

        logging.info(f"Quantized model saved at {output_file}")
    except Exception as e:
        logging.error(f"Error quantizing model: {str(e)}")
        raise

# Function to process the model conversion in the background
def process_request(repo_id, quantization, task_id):
    try:
        processing_tasks[task_id] = "Processing"

        local_dir = os.path.join(BASE_DIR, task_id)
        gguf_output = os.path.join(DOWNLOADS_DIR, f"{task_id}.gguf")
        quantized_output = os.path.join(DOWNLOADS_DIR, f"{task_id}_{quantization}.gguf")

        # Step 1: Download Model
        download_model(repo_id, local_dir)

        # Step 2: Convert SafeTensors (if present)
        convert_safetensors_to_pytorch(local_dir)

        # Step 3: Convert to GGUF
        convert_to_gguf(local_dir, gguf_output, "F16")  # Default to F16 before quantization

        # Step 4: Quantize GGUF Model
        if quantization in GGUF_QUANTIZATION_TYPES:
            quantize_gguf_model(gguf_output, quantized_output, quantization)
        else:
            logging.warning(f"Unsupported quantization type: {quantization}. Skipping quantization.")

        # Generate download link
        final_output = quantized_output if os.path.exists(quantized_output) else gguf_output
        download_link = f"http://localhost:5000/download/{task_id}"

        processing_tasks[task_id] = download_link
        logging.info(f"Processing completed for {repo_id}")

    except Exception as e:
        processing_tasks[task_id] = "Failed"
        logging.error(f"Error processing {repo_id}: {str(e)}")

# API Endpoint: Start Processing
@app.route("/convert", methods=["POST"])
def convert():
    data = request.json
    repo_id = data.get("repo_id")
    quantization = data.get("quantization")

    if not repo_id or not quantization:
        return jsonify({"error": "Missing required parameters"}), 400

    task_id = secure_filename(repo_id) + "_" + quantization

    # Run processing in a separate thread
    thread = threading.Thread(target=process_request, args=(repo_id, quantization, task_id))
    thread.start()

    return jsonify({"message": "Processing started", "task_id": task_id})

# API Endpoint: Get Status
@app.route("/status/<task_id>", methods=["GET"])
def get_status(task_id):
    status = processing_tasks.get(task_id, "Not Found")
    return jsonify({"task_id": task_id, "status": status})

# API Endpoint: Download GGUF File
@app.route("/download/<task_id>", methods=["GET"])
def download(task_id):
    file_path = os.path.join(DOWNLOADS_DIR, f"{task_id}.gguf")
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, threaded=True)