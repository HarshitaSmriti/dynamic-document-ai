"""Streamlit Application Entrypoint for Dynamic Document AI Extractor."""

import base64
import json
import os
import requests
import streamlit as st
from PIL import Image

from frontend.components.dynamic_form import render_dynamic_form

# Application Configuration
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Dynamic Document AI Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📄 Dynamic Enterprise Document AI")
st.caption("Schema-Agnostic Multimodal Document Extraction Platform")

# Sidebar: Configuration & Schema Builder
with st.sidebar:
    st.header("⚙️ Extraction Settings")

    backend_endpoint = st.text_input("Backend API URL", value=BACKEND_URL)

    extraction_mode = st.radio(
        "Extraction Mode",
        options=["Freeform Discovery", "Custom Schema Definition"],
        help="Choose whether to automatically discover fields or specify a dynamic schema.",
    )

    custom_schema = None
    custom_instructions = None

    if extraction_mode == "Custom Schema Definition":
        st.subheader("Dynamic Schema Builder")
        schema_name = st.text_input("Document Type Name", value="Contract / Invoice / Form")
        schema_desc = st.text_area("Document Context / Instructions", value="Extract all relevant key fields.")

        raw_fields_json = st.text_area(
            "Target Fields (JSON Schema)",
            value='''[
  {"name": "document_id", "type": "string", "description": "Identifier or reference number"},
  {"name": "date", "type": "date", "description": "Primary date of document"},
  {"name": "parties_involved", "type": "list", "description": "Names of entities/parties"},
  {"name": "total_amount", "type": "number", "description": "Final monetary amount if present"}
]''',
            height=200,
        )

        try:
            fields_data = json.loads(raw_fields_json)
            custom_schema = {
                "schema_name": schema_name,
                "description": schema_desc,
                "fields": fields_data,
            }
        except Exception:
            st.warning("⚠️ Invalid JSON in Schema Definition")
    else:
        custom_instructions = st.text_area(
            "Extraction Guidance (Optional)",
            placeholder="e.g. Focus on extracting table items, signatures, and summary totals.",
        )

# Main Area: Document Upload & Extraction
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a document image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Document", use_column_width=True)

        extract_btn = st.button("🚀 Run Dynamic Extraction", type="primary", use_container_width=True)
    else:
        extract_btn = False

with col2:
    st.subheader("2. Extracted Data & Dynamic Form")

    if extract_btn and uploaded_file is not None:
        with st.spinner("Processing document with Vision-Language Model..."):
            try:
                # Prepare payload
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()
                b64_encoded = base64.b64encode(file_bytes).decode("utf-8")

                payload = {
                    "document_base64": b64_encoded,
                    "schema_definition": custom_schema,
                    "raw_prompt_override": custom_instructions,
                }

                response = requests.post(
                    f"{backend_endpoint}/api/v1/extract",
                    json=payload,
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    st.session_state["extraction_result"] = result
                else:
                    st.error(f"Backend API error ({response.status_code}): {response.text}")

            except Exception as e:
                st.error(f"Connection error to backend: {str(e)}")

    if "extraction_result" in st.session_state:
        res = st.session_state["extraction_result"]
        data = res.get("extracted_data", {})

        tabs = st.tabs(["📝 Editable Form", "🔍 Structured JSON", "ℹ️ Model Details"])

        with tabs[0]:
            st.markdown("##### Verified / Editable Data")
            edited_data = render_dynamic_form(data)
            st.download_button(
                "💾 Download Verified JSON",
                data=json.dumps(edited_data, indent=2),
                file_name="extracted_document.json",
                mime="application/json",
            )

        with tabs[1]:
            st.json(data)

        with tabs[2]:
            st.write(f"**Success:** {res.get('success')}")
            if res.get("error_message"):
                st.warning(f"**Error / Warning:** {res.get('error_message')}")
            with st.expander("Raw Model Output"):
                st.code(res.get("raw_model_output", "N/A"), language="json")
    else:
        st.info("Upload a document and click 'Run Dynamic Extraction' to inspect extracted fields.")
