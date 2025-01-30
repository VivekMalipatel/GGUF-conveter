import requests
import time

# Start processing
try:
    response = requests.post(
        "http://127.0.0.1:5050/convert",
        json={
            "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
            "quantization": "Q8_0",
            "token": "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"  # Add your token here
        }
    )
    response.raise_for_status()  # Raise an error for bad status codes
    response_data = response.json()
    print(response_data)
except requests.exceptions.RequestException as e:
    print(f"Error starting conversion: {e}")
    print(f"Response content: {response.content}")
    exit(1)

task_id = response_data.get("task_id")

# Check status
while True:
    try:
        status_response = requests.get(f"http://127.0.0.1:5050/status/{task_id}")
        status_response.raise_for_status()  # Raise an error for bad status codes
        status_data = status_response.json()
        status = status_data.get("status")
        print(f"Status: {status}")
        if status != "Processing":
            break
    except requests.exceptions.RequestException as e:
        print(f"Error checking status: {e}")
        print(f"Response content: {status_response.content}")
        exit(1)
    time.sleep(5)

# Download GGUF file
if status != "Failed":
    try:
        download_response = requests.get(f"http://127.0.0.1:5050/download/{task_id}")
        download_response.raise_for_status()  # Raise an error for bad status codes
        if download_response.status_code == 200:
            with open(f"{task_id}.gguf", "wb") as f:
                f.write(download_response.content)
            print(f"Downloaded {task_id}.gguf")
        else:
            print("Error downloading file:", download_response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        print(f"Response content: {download_response.content}")
else:
    print("Conversion failed.")