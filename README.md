# SAM Face Aging API - Banana Serverless

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-red?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-HTTP_Server-000000?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A serverless API for **AI-powered face aging** built on the [Banana.dev](https://banana.dev) platform. Given a face photo and a target age, the service generates a realistic aged (or de-aged) version of the face using the [SAM (Style-based Age Manipulation)](https://github.com/yuval-alaluf/SAM) deep learning model.

## Features

- **Face Aging & De-aging** - Transform any face photo to a specified target age
- **Automatic Face Alignment** - Uses dlib's 68-point facial landmark detector for precise face alignment
- **Rotation Correction** - Automatically detects and corrects image rotation to find faces in any orientation
- **Base64 I/O** - Accepts and returns images as base64-encoded strings for easy API integration
- **GPU Accelerated** - Runs inference on CUDA-enabled GPUs for fast processing
- **Serverless Ready** - Designed for scalable deployment on Banana.dev infrastructure

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Model | [SAM (pSp Network)](https://github.com/yuval-alaluf/SAM) - Pretrained on FFHQ |
| Deep Learning | PyTorch, torchvision |
| Face Detection | dlib, OpenCV |
| HTTP Server | Flask |
| Containerization | Docker (CUDA base image) |
| Deployment | Banana.dev Serverless |

## Project Structure

```
banana_serverless/
├── app.py              # Entry point - init() loads model, inference() handles requests
├── server.py           # Flask HTTP server with healthcheck and inference endpoints
├── functions.py        # Core ML logic - model loading, face alignment, age prediction
├── Rotator.py          # Image rotation correction using dlib face detection
├── download.py         # Model weight downloader (used during container build)
├── send_log.py         # Remote logging utility
├── Dockerfile          # Container definition with CUDA support
├── build.sh            # System dependency installation script
├── requirements.txt    # Python dependencies
├── test.py             # Basic HTTP endpoint test
└── data1.json          # Sample test data
```

## Quickstart

**[Follow the quickstart guide in Banana's documentation to use this repo](https://docs.banana.dev/banana-docs/quickstart).**

*(Choose "GitHub Repository" deployment method)*

### Prerequisites

- Docker with NVIDIA GPU support
- CUDA 11+ compatible GPU
- Pretrained SAM model weights (`sam_ffhq_aging.pt`)
- dlib face landmark predictor (`shape_predictor_68_face_landmarks.dat`)

### Local Development

1. **Clone the repository and the SAM model:**

   ```bash
   git clone https://github.com/yakubilik/banana_serverless.git
   cd banana_serverless
   git clone https://github.com/yuval-alaluf/SAM
   ```

2. **Install system dependencies:**

   ```bash
   bash build.sh
   ```

3. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Place pretrained model weights** in `./pretrained_models/`:
   - `sam_ffhq_aging.pt` (SAM age manipulation model)
   - `shape_predictor_68_face_landmarks.dat` (dlib face landmarks)

5. **Run the server:**

   ```bash
   python server.py
   ```

   The server will start on `http://localhost:8000`.

### Docker Deployment

```bash
docker build -t sam-face-aging .
docker run --gpus all -p 8000:8000 sam-face-aging
```

## API Usage

### Endpoint

`POST /`

### Request Body

```json
{
  "modelInputs": {
    "image": "<base64-encoded-image-string>",
    "target_age": 60
  }
}
```

### Response

```json
{
  "oldImage": "<base64-encoded-result-image>"
}
```

### Example (Python)

```python
import requests
import base64

# Read and encode the input image
with open("photo.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Send the aging request
response = requests.post("http://localhost:8000/", json={
    "modelInputs": {
        "image": image_b64,
        "target_age": 70
    }
})

# Decode and save the result
result_b64 = response.json()["oldImage"]
with open("aged_photo.jpg", "wb") as f:
    f.write(base64.b64decode(result_b64))
```

### Healthcheck

```
GET /healtcheck
```

Returns GPU availability status:

```json
{
  "state": "healthy",
  "gpu": true
}
```

## How It Works

1. **Model Loading** - On startup, the SAM pSp network is loaded from pretrained weights onto the GPU
2. **Face Alignment** - Input images are aligned using dlib's 68-point facial landmark predictor
3. **Rotation Handling** - If face detection fails, the image is rotated in 90-degree increments until a face is found
4. **Age Transformation** - The aligned face is passed through the SAM network with the target age parameter
5. **Output** - The resulting aged face image is encoded as base64 and returned

## Helpful Links

- [Banana Serverless Framework](https://docs.banana.dev/banana-docs/core-concepts/inference-server/serverless-framework) - Understand the framework and each file's role
- [Deploy Anything on Banana](https://docs.banana.dev/banana-docs/resources/how-to-serve-anything-on-banana) - Generalize this framework for other models
- [SAM Paper (Alaluf et al.)](https://arxiv.org/abs/2004.02546) - Original research behind the age manipulation model

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
