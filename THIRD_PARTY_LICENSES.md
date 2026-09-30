# Third-Party Licenses and Attributions

This project utilizes open-source libraries and machine learning models. Below is a summary of their licenses and attributions:

| Component / Library | License | Usage / Purpose | Attribution / Source |
| :--- | :--- | :--- | :--- |
| **Qwen/Qwen2.5-VL-7B-Instruct** | **Apache 2.0** | Core Vision-Language Model for multimodal document extraction | Qwen Team, Alibaba Cloud |
| **Transformers** | **Apache 2.0** | Model loading, tokenization, chat templates, and inference pipeline | Hugging Face Inc. |
| **Accelerate** | **Apache 2.0** | Hardware-aware device mapping and multi-GPU/CPU inference orchestration | Hugging Face Inc. |
| **PyTorch** | **BSD 3-Clause** | Tensor computing and deep learning framework | Meta Platforms, Inc. & PyTorch Contributors |
| **Pillow** | **HPND (Permissive)** | Image opening, scaling, and color mode formatting | Alex Clark & Contributors |
| **PyMuPDF** | **AGPL-3.0 / Commercial** | High-performance PDF rendering to raster image buffers | Artifex Software, Inc. |
| **FastAPI** | **MIT** | REST API routing and async request handling | Sebastián Ramírez |
| **Pydantic** | **MIT** | Data validation and schema definition | Samuel Colvin |
| **Uvicorn** | **BSD 3-Clause** | ASGI server implementation | Encode OSS |
| **Streamlit** | **Apache 2.0** | Interactive web dashboard and dynamic form UI | Snowflake Inc. / Streamlit |
| **Pandas** | **BSD 3-Clause** | Table export and data manipulation | PyData Team |

---

### Commercial Readiness Notice
- All dependencies used in the core inference path are standard open-source components with commercial-friendly licenses.
- No paid proprietary AI APIs (such as OpenRouter, OpenAI paid tokens, or proprietary vendor lock-in) are required for operation.
- The model weights for `Qwen2.5-VL-7B-Instruct` are released under the Apache 2.0 license, permitting commercial usage, modification, and redistribution with proper attribution as maintained herein.
