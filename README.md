# Dynamic Enterprise Document AI

A commercially-ready, schema-agnostic Enterprise Document AI extraction platform powered by the official **Qwen2.5-VL-7B-Instruct** model running directly via Hugging Face Transformers.

---

## Key Highlights

- **Direct Open-Source Model Inference**: Runs `Qwen/Qwen2.5-VL-7B-Instruct` directly via Hugging Face Transformers with zero third-party token fees or credit constraints.
- **Commercial-Friendly Licensing**: Built with open-source libraries and the Apache-2.0 licensed Qwen2.5-VL model.
- **Provider Abstraction Layer**: Pluggable `BaseVLMProvider` interface (`LocalQwenVLProvider`, `OpenAICompatibleVLMProvider`) enabling seamless model substitution.
- **Hardware-Aware Execution**: Automatic hardware detection for CUDA GPU acceleration (`bfloat16`/`float16`) with automatic CPU fallback.
- **Multi-Page PDF & Image Intelligence**: High-speed rasterization, bounded pixel scaling, continuation table stitching, and dynamic field deduplication across multi-page documents.
- **Robust JSON Normalization & Repair**: Multi-stage JSON repair engine correcting missing commas, unescaped characters, and truncated outputs without crashing.
- **Streamlit & FastAPI Architecture**: Full-featured interactive dashboard with dynamic form editing, CSV/JSON export, and high-performance REST API endpoints.

---

## System Architecture

```
User / Client
      │
      ├── Streamlit UI (streamlit_app.py / frontend/app.py)
      │         │ (HTTP / Multipart)
      ▼         ▼
FastAPI API Endpoints (/api/v1/extract/upload, /api/v1/extract)
      │
Document Extraction Service (backend/extraction/service.py)
      ├── PyMuPDF Rasterizer & Image Optimizer (bounded pixels)
      ├── Dynamic Prompt Builder (compact extraction schema)
      │
Vision-Language Provider Abstraction (backend/extraction/qwen_provider.py)
      ├── LocalQwenVLProvider (Default: Qwen2.5-VL-7B on Hugging Face Transformers)
      │       ├── Hardware Detection: CUDA GPU (bf16/fp16) or CPU Fallback (fp32)
      │       └── Cached Thread-Safe Model Singleton
      └── OpenAICompatibleVLMProvider (Optional: Self-hosted vLLM / Ollama endpoint)
      │
Dynamic JSON Parser & Consolidation Engine (backend/extraction/parser.py)
      ├── Fence cleaner & JSON syntax repair
      ├── Dynamic entity normalization
      └── Multi-page field deduplication & table continuation merger
```

---

## Environment Variables

Configure application settings in `.env` or through container environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MODEL_BACKEND` | `local` | Provider backend: `local` (Hugging Face Transformers) or `hosted_api` (self-hosted endpoint) |
| `QWEN_MODEL_ID` | `Qwen/Qwen2.5-VL-7B-Instruct` | Hugging Face model identifier or local weights path |
| `QWEN_DEVICE` | `auto` | Target device: `auto` (detects CUDA > MPS > CPU), `cuda`, or `cpu` |
| `QWEN_TORCH_DTYPE` | `auto` | Precision: `auto` (`bfloat16`/`float16` on GPU, `float32` on CPU), `bfloat16`, `float16` |
| `QWEN_MAX_NEW_TOKENS`| `1024` | Configurable token ceiling for extracted structured output |
| `QWEN_TEMPERATURE` | `0.0` | Sampling temperature (`0.0` for deterministic extraction) |
| `HF_HOME` | `None` | Optional custom Hugging Face model weights cache directory |
| `BACKEND_API_URL` | `http://localhost:8000` | URL connecting Streamlit frontend to FastAPI backend |
| `PORT` | `8000` | Server listening port |

---

## Local Development & Installation

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Optional for GPU acceleration: NVIDIA GPU with CUDA 12.x drivers installed.

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/HarshitaSmriti/dynamic-document-ai.git
cd dynamic_extractor

# Create and activate virtual environment
python -m venv .venv

# Windows:
.\.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

*For CUDA-enabled PyTorch (GPU execution), install the CUDA build of PyTorch:*
```bash
# Example for CUDA 12.1 / 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Run Backend API Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Run Streamlit UI Dashboard

```bash
streamlit run streamlit_app.py
```

### 5. Run Test Suite

```bash
pytest
```

---

## Hardware Requirements & Deployment Guide

### GPU Recommendations (for Local Ingestion)
- **Model**: `Qwen/Qwen2.5-VL-7B-Instruct` (~14.5 GB weights in FP16/BF16).
- **Minimum VRAM**: 16 GB VRAM (e.g., NVIDIA RTX 4080, RTX 4090, A4000, T4 with 16GB, or AWS `g5.xlarge` / `g4dn.xlarge`).
- **Recommended VRAM**: 24 GB VRAM (NVIDIA RTX 3090, RTX 4090, A10G, A5000).

### CPU Execution (Fallback)
- **Minimum RAM**: 16 GB - 32 GB System RAM.
- **Expected Speed**: CPU inference runs at ~0.5 to 2 tokens/sec depending on CPU cores and AVX512 support. Recommended for testing and lower-throughput workloads.

### Render / Cloud Deployment Notes
- **Render Standard/Starter CPU tiers** (512MB - 2GB RAM) cannot load a 7B parameter vision model into memory.
- **Recommended Cloud Topology**:
  1. **Option A (All-in-One GPU Node)**: Deploy backend and frontend on a GPU instance (AWS EC2 g5.xlarge, RunPod, Lambda Labs, GCP).
  2. **Option B (Hybrid Deployment)**: Host the lightweight FastAPI backend and Streamlit UI on Render, and set `MODEL_BACKEND=hosted_api` pointing `QWEN_API_BASE=https://<your-gpu-instance>/v1` to a self-hosted GPU inference engine (vLLM / Ollama).

---

## Licensing & Attribution

- Core Application Code: Proprietary / Enterprise License.
- Model Weights: **Qwen2.5-VL-7B-Instruct** is licensed under **Apache 2.0** by Alibaba Cloud.
- See [`NOTICE`](./NOTICE) and [`THIRD_PARTY_LICENSES.md`](./THIRD_PARTY_LICENSES.md) for full attributions and license details.
