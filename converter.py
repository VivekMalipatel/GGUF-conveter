import os
import threading
import logging
import torch
from safetensors.torch import load_file
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from huggingface_hub import snapshot_download, HfApi
import subprocess
from werkzeug.utils import secure_filename
import socket
from fastapi.staticfiles import StaticFiles

class Config:
    BASE_DIR = "./models"
    DOWNLOADS_DIR = "./downloads"
    LOG_FILE = "app.log"
    LLAMA_CONVERTER_SCRIPT = "./llama.cpp/convert_hf_to_gguf.py"
    PORT = 5050
    GGUF_QUANTIZATION_TYPES = [
        "F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q2_K", "Q3_K", "Q3_K_S", "Q3_K_M", "Q3_K_L"
    ]

class Logger:
    @staticmethod
    def setup():
        os.makedirs(Config.BASE_DIR, exist_ok=True)
        os.makedirs(Config.DOWNLOADS_DIR, exist_ok=True)
        logging.basicConfig(filename=Config.LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger("uvicorn").handlers.clear()
        logging.getLogger("uvicorn.access").handlers.clear()
        file_handler = logging.FileHandler(Config.LOG_FILE)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger("uvicorn").addHandler(file_handler)
        logging.getLogger("uvicorn.access").addHandler(file_handler)

class ModelManager:
    def __init__(self, repo_id, local_dir, token=None):
        self.repo_id = repo_id
        self.local_dir = local_dir
        self.token = token

    def download_model(self):
        try:
            logging.info(f"Downloading model {self.repo_id}...")
            os.makedirs(self.local_dir, exist_ok=True)
            snapshot_download(repo_id=self.repo_id, local_dir=self.local_dir, local_dir_use_symlinks=False, revision="main", token=self.token)
            api = HfApi(token=self.token)
            model_info = api.model_info(self.repo_id)
            last_modified_remote = model_info.lastModified.isoformat()
            with open(os.path.join(self.local_dir, ".last_modified"), "w") as f:
                f.write(last_modified_remote)
            logging.info(f"Download complete: {self.local_dir}")
        except Exception as e:
            logging.error(f"Error downloading model {self.repo_id}: {str(e)}")
            raise

    def convert_safetensors_to_pytorch(self):
        try:
            for filename in os.listdir(self.local_dir):
                if filename.endswith(".safetensors"):
                    safetensor_path = os.path.join(self.local_dir, filename)
                    pytorch_path = safetensor_path.replace(".safetensors", ".bin")
                    if os.path.exists(pytorch_path):
                        logging.info(f"Skipping conversion, {pytorch_path} already exists.")
                        continue
                    logging.info(f"Converting {safetensor_path} to {pytorch_path}...")
                    tensors = load_file(safetensor_path)
                    torch.save(tensors, pytorch_path)
                    logging.info(f"Converted SafeTensors to PyTorch: {pytorch_path}")
        except Exception as e:
            logging.error(f"Error converting SafeTensors in self.local_dir: {str(e)}")
            raise

    def convert_to_gguf(self, output_file, quantization_type):
        try:
            logging.info(f"Converting {self.local_dir} to GGUF...")
            if not os.path.exists(Config.LLAMA_CONVERTER_SCRIPT):
                raise FileNotFoundError(f"Converter script not found: {Config.LLAMA_CONVERTER_SCRIPT}")
            command = [
                "python3", Config.LLAMA_CONVERTER_SCRIPT,
                self.local_dir, "--outfile", output_file, "--outtype", quantization_type.lower()
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

    def is_model_downloaded(self):
        try:
            return os.path.exists(self.local_dir) and os.listdir(self.local_dir)
        except Exception as e:
            logging.error(f"Error checking if model {self.repo_id} is downloaded: {str(e)}")
            return False

    def has_model_changed(self):
        try:
            api = HfApi()
            model_info = api.model_info(self.repo_id)
            last_modified_remote = model_info.lastModified
            local_model_path = os.path.join(self.local_dir, ".last_modified")
            if os.path.exists(local_model_path):
                with open(local_model_path, "r") as f:
                    last_modified_local = f.read().strip()
                return last_modified_local != last_modified_remote
            return True
        except Exception as e:
            logging.error(f"Error checking if model {self.repo_id} has changed: {str(e)}")
            return True

class GGUFConverter:
    def __init__(self):
        self.processing_tasks = {}

    def get_ip_address(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.254.254.254', 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip

    def process_request(self, repo_id, quantization, task_id, token=None):
        try:
            self.processing_tasks[task_id] = "Processing"
            local_dir = os.path.join(Config.BASE_DIR, task_id)
            model_name = secure_filename(repo_id).replace("/", "_")
            gguf_output = os.path.join(Config.DOWNLOADS_DIR, f"{model_name}_{quantization}.gguf")
            if os.path.exists(gguf_output):
                logging.info(f"GGUF file for {repo_id} with quantization {quantization} already present, skipping conversion.")
                ip_address = self.get_ip_address()
                self.processing_tasks[task_id] = f"http://{ip_address}:{Config.PORT}/downloads/{model_name}_{quantization}.gguf"
                return
            os.makedirs(local_dir, exist_ok=True)
            model_manager = ModelManager(repo_id, local_dir, token)
            if not model_manager.is_model_downloaded() or model_manager.has_model_changed():
                model_manager.download_model()
            model_manager.convert_safetensors_to_pytorch()
            model_manager.convert_to_gguf(gguf_output, quantization)
            if os.path.exists(gguf_output):
                ip_address = self.get_ip_address()
                self.processing_tasks[task_id] = f"http://{ip_address}:{Config.PORT}/downloads/{model_name}_{quantization}.gguf"
            else:
                self.processing_tasks[task_id] = "Failed"
            logging.info(f"Processing completed for {repo_id}")
        except Exception as e:
            self.processing_tasks[task_id] = "Failed"
            logging.error(f"Error processing request for {repo_id}: {str(e)}")

app = FastAPI()
converter = GGUFConverter()
Logger.setup()

app.mount("/downloads", StaticFiles(directory=Config.DOWNLOADS_DIR), name="downloads")

@app.post("/convert")
async def convert(request: Request):
    data = await request.json()
    repo_id = data.get("repo_id")
    quantization = data.get("quantization")
    token = data.get("token")
    if not repo_id or not quantization:
        return JSONResponse({"error": "Missing required parameters"}, status_code=400)
    task_id = secure_filename(repo_id) + "_" + quantization
    thread = threading.Thread(target=converter.process_request, args=(repo_id, quantization, task_id, token))
    thread.start()
    return JSONResponse({"message": "Processing started", "task_id": task_id})

@app.get("/status/{task_id}")
def get_status(task_id: str):
    status = converter.processing_tasks.get(task_id, "Not Found")
    return JSONResponse({"task_id": task_id, "status": status})

@app.get("/download/{task_id}")
def download(task_id: str):
    model_name = task_id.rsplit("_", 1)[0]
    quantization = task_id.rsplit("_", 1)[1]
    return JSONResponse({
        "http_link": f"http://{converter.get_ip_address()}:{Config.PORT}/downloads/{model_name}_{quantization}.gguf"
    })