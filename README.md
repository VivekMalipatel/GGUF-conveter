# GGUF-converter

Convert any Safe Tensors or PyTorch model to a GGUF format.

## Overview

This project provides a FastAPI-based web service that allows users to convert models from Safe Tensors or PyTorch format to GGUF format. The service supports various quantization types and can handle multiple concurrent requests. The main functionalities include downloading models from Hugging Face, converting Safe Tensors to PyTorch, and converting models to GGUF format.

## Prerequisites

- Python 3.8 or higher
- `pip` package manager

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/VivekMalipatel/GGUF-converter.git
    cd GGUF-converter/GGUF-conveter
    ```

2. Create and activate a Python virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

4. Clone the `llama.cpp` repository into the project folder:

    ```sh
    git clone https://github.com/ggerganov/llama.cpp.git
    ```

## Configuration

Ensure the following directories exist:

- `./models`: Directory where models will be downloaded.
- `./downloads`: Directory where converted GGUF files will be saved.

## Running the Application

1. Start the FastAPI server:

    ```sh
    uvicorn converter:app --host 0.0.0.0 --port 5050 --reload
    ```

2. The server will be running at `http://127.0.0.1:5050`.

## API Endpoints

### Convert Model

- **Endpoint**: `/convert`
- **Method**: `POST`
- **Description**: Convert a model to GGUF format.
- **Request Body**:
    ```json
    {
        "repo_id": "string",          // The repository ID of the model on Hugging Face
        "quantization": "string",     // The quantization type (e.g., "F32", "F16", "Q8_0")
        "token": "string"             // (Optional) Hugging Face API token for private models
    }
    ```
- **Response**:
    ```json
    {
        "message": "Processing started",
        "task_id": "string"
    }
    ```

### Check Status

- **Endpoint**: `/status/{task_id}`
- **Method**: `GET`
- **Description**: Check the status of a conversion task.
- **Response**:
    ```json
    {
        "task_id": "string",
        "status": "string"  // "Processing", "Failed", or the download URL
    }
    ```

### Download GGUF File

- **Endpoint**: `/download/{task_id}`
- **Method**: `GET`
- **Description**: Get the download link for the converted GGUF file.
- **Response**:
    ```json
    {
        "http_link": "string"  // URL to download the GGUF file
    }
    ```

## How It Works

1. **Model Download**: When a request is made to convert a model, the service first checks if the model is already downloaded. If not, it downloads the model from Hugging Face using the provided `repo_id` and `token`.

2. **Safe Tensors to PyTorch Conversion**: If the model contains Safe Tensors, they are converted to PyTorch format.

3. **GGUF Conversion**: The model is then converted to GGUF format using the `llama.cpp` script. The output file is saved in the `./downloads` directory.

4. **Task Management**: Each conversion request is handled in a separate thread, and the status of each task is tracked using a unique `task_id`.

5. **Logging**: Detailed logs are maintained in the `app.log` file, which includes information about each step of the process.

## Limitations

1. **Concurrency**: While the service can handle multiple concurrent requests, excessive concurrent requests might lead to resource contention, such as disk I/O and network bandwidth.

2. **Model Support**: Not all model architectures are supported for GGUF conversion. For example, models with the architecture `MultiModalityCausalLM` are not supported.

3. **Error Handling**: Errors during model download, conversion, or subprocess execution are logged, but the service might not provide detailed error messages to the client.

4. **Quantization Types**: The quantization types supported are limited to those specified in the `Config.GGUF_QUANTIZATION_TYPES` list.

5. **Development Server**: The provided instructions use `uvicorn` with `--reload` for development purposes. For production deployment, a production WSGI server should be used.

## Logging

Logs are saved to `app.log` in the current directory. The log file contains detailed information about the processing of each request.

## Notes

- This application is intended for development and testing purposes. For production deployment, use a production WSGI server.
- Ensure you have access to the models on Hugging Face if they are private or gated.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
