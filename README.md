# Dynamic Document AI Extractor

A modern, schema-agnostic Enterprise Document AI extraction platform powered by Vision-Language Models (Qwen-VL) and dynamic form rendering.

## Key Features

- **Schema-Agnostic Dynamic Extraction**: Extract arbitrary structured data from any document type (invoices, receipts, contracts, forms, medical records, ID cards) without hardcoded schemas.
- **VLM Provider Abstraction**: Modular Vision-Language Model interface (`QwenProvider`) decoupled from business logic.
- **Robust JSON Parsing & Validation**: Automated markdown stripping, repair, and dynamic Pydantic schema validation.
- **Dynamic Forms Engine**: Streamlit-based UI that renders dynamic form inputs based on user-defined extraction schemas or inferred structures.
- **Modular FastAPI Backend**: High-performance RESTful API endpoints for asynchronous document processing and extraction.

---

## Project Structure

```
dynamic_extractor/
├── backend/
│   ├── main.py                  # FastAPI application entrypoint
│   ├── config.py                # Environment & application settings
│   ├── api/                     # API routers and endpoints
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── extraction/              # Core extraction domain logic
│   │   ├── __init__.py
│   │   ├── prompt.py            # Dynamic prompt builder
│   │   ├── service.py           # Document extraction orchestrator
│   │   ├── qwen_provider.py     # Qwen VLM provider abstraction
│   │   └── parser.py            # JSON parser & repair engine
│   ├── schemas/                 # Pydantic request/response/extraction schemas
│   │   ├── __init__.py
│   │   └── extraction.py
│   └── __init__.py
├── frontend/
│   ├── app.py                   # Streamlit UI dashboard
│   ├── components/              # Reusable UI widgets
│   │   ├── __init__.py
│   │   └── dynamic_form.py      # Dynamic schema form renderer
│   └── __init__.py
├── tests/                       # Unit and integration test suite
│   ├── __init__.py
│   ├── test_api.py
│   └── test_parser.py
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore configuration
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## Getting Started (Setup & Run)

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` and adjust settings as required:
```bash
cp .env.example .env
```

### 3. Running Backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Running Frontend
```bash
streamlit run frontend/app.py
```

### 5. Running Tests
```bash
pytest
```
