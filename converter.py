import os
import threading
import logging
import torch
from safetensors.torch import load_file, save_file
from flask import Flask, request, jsonify, send_file
from huggingface_hub import snapshot_download, HfApi
import subprocess
from werkzeug.utils import secure_filename

# Flask App Setup
app = Flask(__name__)

# 📌 Configurations
BASE_DIR = "./models"
DOWNLOADS_DIR = "./downloads"
LOG_FILE = "app.log"
LLAMA_CONVERTER_SCRIPT = "./llama.cpp/convert_hf_to_gguf.py"
PORT = 5050  # Port number variable

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
def download_model(repo_id, local_dir, token=None):
    try:
        logging.info(f"Downloading model {repo_id}...")
        os.makedirs(local_dir, exist_ok=True)  # Ensure the local directory exists
        snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False, revision="main", token=token)
        
        # Save the last modified date
        api = HfApi(token=token)
        model_info = api.model_info(repo_id)
        last_modified_remote = model_info.lastModified.isoformat()  # Convert datetime to string
        with open(os.path.join(local_dir, ".last_modified"), "w") as f:
            f.write(last_modified_remote)

        logging.info(f"Download complete: {local_dir}")
    except Exception as e:
        logging.error(f"Error downloading model {repo_id}: {str(e)}")
        raise

# Function to convert SafeTensors to PyTorch bin (if needed)
def convert_safetensors_to_pytorch(model_dir):
    try:
        for filename in os.listdir(model_dir):
            if filename.endswith(".safetensors"):
                safetensor_path = os.path.join(model_dir, filename)
                pytorch_path = safetensor_path.replace(".safetensors", ".bin")

                if os.path.exists(pytorch_path):
                    logging.info(f"Skipping conversion, {pytorch_path} already exists.")
                    continue

                logging.info(f"Converting {safetensor_path} to {pytorch_path}...")

                # Load SafeTensors
                tensors = load_file(safetensor_path)

                # Convert and save as PyTorch bin
                torch.save(tensors, pytorch_path)

                logging.info(f"Converted SafeTensors to PyTorch: {pytorch_path}")
        
    except Exception as e:
        logging.error(f"Error converting SafeTensors in {model_dir}: {str(e)}")
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
            local_dir, "--outfile", output_file, "--outtype", quantization_type.lower()  # Use lower case for outtype
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        result.check_returncode()
        
        logging.info(f"GGUF model saved at {output_file}")
    except FileNotFoundError as e:
        logging.error(f"File not found error during GGUF conversion: {str(e)}")
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess error during GGUF conversion: {str(e)}")
        logging.error(f"Subprocess stdout: {e.stdout}")
        logging.error(f"Subprocess stderr: {e.stderr}")
        if "Model MultiModalityCausalLM is not supported" in e.stderr:
            logging.error("The model architecture is not supported for GGUF conversion.")
        raise
    except Exception as e:
        logging.error(f"Error converting to GGUF: {str(e)}")
        raise

# Function to check if the model is already downloaded
def is_model_downloaded(repo_id, local_dir):
    try:
        # Check if the directory exists and is not empty
        return os.path.exists(local_dir) and os.listdir(local_dir)
    except Exception as e:
        logging.error(f"Error checking if model {repo_id} is downloaded: {str(e)}")
        return False

# Function to check if the model has changed on Hugging Face
def has_model_changed(repo_id, local_dir):
    try:
        api = HfApi()
        model_info = api.model_info(repo_id)
        last_modified_remote = model_info.lastModified
        local_model_path = os.path.join(local_dir, ".last_modified")

        if os.path.exists(local_model_path):
            with open(local_model_path, "r") as f:
                last_modified_local = f.read().strip()
            return last_modified_local != last_modified_remote

        return True
    except Exception as e:
        logging.error(f"Error checking if model {repo_id} has changed: {str(e)}")
        return True

# Function to check if the GGUF file is already converted
def is_gguf_converted(model_name, quantization):
    try:
        gguf_path = os.path.join(DOWNLOADS_DIR, f"{model_name}_{quantization}.gguf")
        return os.path.exists(gguf_path)
    except Exception as e:
        logging.error(f"Error checking if GGUF file {model_name}_{quantization}.gguf is converted: {str(e)}")
        return False

# Function to process the model conversion in the background
def process_request(repo_id, quantization, task_id, token=None):
    try:
        processing_tasks[task_id] = "Processing"

        local_dir = os.path.join(BASE_DIR, task_id)
        os.makedirs(local_dir, exist_ok=True)  # Ensure the local directory exists
        model_name = secure_filename(repo_id).replace("/", "_")
        gguf_output = os.path.join(DOWNLOADS_DIR, f"{model_name}_{quantization}.gguf")

        # Step 1: Download Model (if not already downloaded or if changed)
        if not is_model_downloaded(repo_id, local_dir) or has_model_changed(repo_id, local_dir):
            download_model(repo_id, local_dir, token)

        # Step 2: Convert SafeTensors (if present)
        convert_safetensors_to_pytorch(local_dir)

        # Step 3: Convert to GGUF
        convert_to_gguf(local_dir, gguf_output, quantization)

        # Generate download link
        download_link = f"http://localhost:{PORT}/download/{task_id}"

        processing_tasks[task_id] = download_link
        logging.info(f"Processing completed for {repo_id}")

    except Exception as e:
        processing_tasks[task_id] = "Failed"
        logging.error(f"Error processing request for {repo_id}: {str(e)}")

# API Endpoint: Start Processing
@app.route("/convert", methods=["POST"])
def convert():
    data = request.json
    repo_id = data.get("repo_id")
    quantization = data.get("quantization")
    token = data.get("token")  # Get the token from the request

    if not repo_id or not quantization:
        return jsonify({"error": "Missing required parameters"}), 400

    task_id = secure_filename(repo_id) + "_" + quantization

    # Run processing in a separate thread
    thread = threading.Thread(target=process_request, args=(repo_id, quantization, task_id, token))
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
    model_name = task_id.rsplit("_", 1)[0]
    quantization = task_id.rsplit("_", 1)[1]
    file_path = os.path.join(DOWNLOADS_DIR, f"{model_name}_{quantization}.gguf")
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, threaded=True, host="0.0.0.0", port=PORT)  # Bind to 0.0.0.0 to allow access from other devices