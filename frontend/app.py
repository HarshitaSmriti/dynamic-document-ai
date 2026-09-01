"""Enterprise Dynamic Document AI - Streamlit Dashboard.

Public-ready Streamlit frontend calling the backend extraction API via STREAMLIT_API_URL.
No API keys are embedded or exposed in this frontend.
"""

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import requests
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Page Setup MUST BE FIRST STREAMLIT CALL
st.set_page_config(
    page_title="Dynamic Document AI Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_dynamic_form(data: Dict[str, Any], key_prefix: str = "field") -> Dict[str, Any]:
    """Recursively render editable Streamlit input components for dynamic JSON key-values."""
    edited_data = {}

    if not data or not isinstance(data, dict):
        st.info("No structured data to display in form.")
        return edited_data

    for key, value in data.items():
        if key.startswith("_"):
            continue

        unique_key = f"{key_prefix}_{key}"
        label = key.replace("_", " ").title()

        if isinstance(value, dict):
            with st.expander(f"📁 {label}", expanded=True):
                edited_data[key] = render_dynamic_form(value, key_prefix=unique_key)

        elif isinstance(value, list):
            with st.expander(f"📋 {label} ({len(value)} items)", expanded=True):
                edited_list = []
                for idx, item in enumerate(value):
                    if isinstance(item, dict):
                        st.markdown(f"**Item {idx + 1}**")
                        edited_item = render_dynamic_form(item, key_prefix=f"{unique_key}_{idx}")
                        edited_list.append(edited_item)
                        st.divider()
                    else:
                        item_val = st.text_input(
                            f"Item {idx + 1}",
                            value=str(item) if item is not None else "",
                            key=f"{unique_key}_{idx}",
                        )
                        edited_list.append(item_val)
                edited_data[key] = edited_list

        elif isinstance(value, bool):
            edited_data[key] = st.checkbox(label, value=value, key=unique_key)

        elif isinstance(value, (int, float)):
            if isinstance(value, int):
                edited_data[key] = st.number_input(label, value=int(value), step=1, key=unique_key)
            else:
                edited_data[key] = st.number_input(label, value=float(value), step=0.01, format="%.2f", key=unique_key)

        else:
            str_val = str(value) if value is not None else ""
            if len(str_val) > 80:
                edited_data[key] = st.text_area(label, value=str_val, key=unique_key)
            else:
                edited_data[key] = st.text_input(label, value=str_val, key=unique_key)

    return edited_data


# Environment Configuration (Render / Local API URL)
DEFAULT_API_URL = os.getenv("STREAMLIT_API_URL", os.getenv("BACKEND_API_URL", "http://localhost:8000"))

# Header Section
st.title("📄 Dynamic Enterprise Document AI")
st.caption("Schema-Agnostic Multimodal Extraction with Qwen2.5-VL • Multi-Page PDF & Image Intelligence")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ API & Extraction Settings")

    backend_endpoint = st.text_input(
        "Backend API Base URL",
        value=DEFAULT_API_URL,
        help="URL of the deployed Render Backend API (or http://localhost:8000)",
    ).rstrip("/")

    # Health Check Probe
    if st.button("🔄 Check Backend Health", use_container_width=True):
        try:
            res = requests.get(f"{backend_endpoint}/health", timeout=5)
            if res.status_code == 200:
                h_data = res.json()
                st.success(f"✅ Online ({h_data.get('provider', {}).get('model_name', 'Qwen-VL')})")
            else:
                st.error(f"⚠️ Returned HTTP {res.status_code}")
        except Exception as err:
            st.error(f"❌ Connection Failed: {str(err)}")

    st.divider()
    st.subheader("Extraction Mode")
    extraction_mode = st.radio(
        "Select Mode",
        options=["✨ Universal Dynamic Discovery", "📐 Custom Schema Guidance"],
        help="Dynamic Discovery extracts all document entities without forcing a fixed schema.",
    )

    custom_schema = None
    custom_instructions = None

    if extraction_mode == "📐 Custom Schema Guidance":
        schema_name = st.text_input("Document Name / Classification", value="Custom Document")
        raw_fields_json = st.text_area(
            "Target Schema Definition (JSON)",
            value='''[
  {"name": "document_number", "type": "string", "description": "Primary ID"},
  {"name": "date", "type": "date", "description": "Issue date"},
  {"name": "parties", "type": "list", "description": "Parties involved"},
  {"name": "total_amount", "type": "number", "description": "Total value"}
]''',
            height=160,
        )
        try:
            fields_data = json.loads(raw_fields_json)
            custom_schema = {"schema_name": schema_name, "fields": fields_data}
        except Exception:
            st.warning("⚠️ Invalid JSON in Schema Definition")
    else:
        custom_instructions = st.text_area(
            "Custom Extraction Guidance (Optional)",
            placeholder="e.g. Focus on material details and payment conditions.",
            help="Optional natural language guidance for the Vision-Language Model.",
        )

# Main Two-Column Layout
col_upload, col_result = st.columns([1, 1.3], gap="large")

with col_upload:
    st.subheader("1. Document Ingestion")
    uploaded_file = st.file_uploader(
        "Upload PDF or Image (Multi-page supported)",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        help="Upload multi-page invoices, receipts, agreements, purchase orders, or shipping bills.",
    )

    if uploaded_file is not None:
        is_pdf = uploaded_file.name.lower().endswith(".pdf") or uploaded_file.type == "application/pdf"
        file_size_kb = len(uploaded_file.getvalue()) / 1024

        st.info(f"📁 **File:** `{uploaded_file.name}` ({file_size_kb:.1f} KB) • **Format:** {'Multi-Page PDF' if is_pdf else 'Image'}")

        if not is_pdf:
            try:
                img = Image.open(uploaded_file)
                st.image(img, caption="Document Preview", use_container_width=True)
            except Exception:
                st.warning("Could not render image preview.")
        else:
            st.markdown("📑 *PDF document loaded and ready for multi-page extraction.*")

        extract_btn = st.button("🚀 Run Dynamic Extraction", type="primary", use_container_width=True)
    else:
        extract_btn = False

with col_result:
    st.subheader("2. Structured Extraction Results")

    if extract_btn and uploaded_file is not None:
        with st.spinner("Processing document with Vision-Language Model..."):
            t_start = time.perf_counter()
            try:
                uploaded_file.seek(0)
                file_bytes = uploaded_file.read()

                files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type or "application/octet-stream")}
                data_payload = {}
                if custom_instructions and custom_instructions.strip():
                    data_payload["custom_instructions"] = custom_instructions.strip()
                if custom_schema:
                    data_payload["schema_json"] = json.dumps(custom_schema)

                endpoint = f"{backend_endpoint}/api/v1/extract/upload"
                response = requests.post(
                    endpoint,
                    files=files,
                    data=data_payload,
                    timeout=300,
                )
                latency = round(time.perf_counter() - t_start, 2)

                if response.status_code == 200:
                    res_json = response.json()
                    st.session_state["extraction_response"] = res_json
                    st.session_state["latency"] = latency
                    st.success(f"✅ Extraction completed in {latency}s!")
                else:
                    st.error(f"❌ Backend API Error (HTTP {response.status_code}): {response.text}")
                    st.session_state.pop("extraction_response", None)

            except requests.exceptions.ConnectionError:
                st.error(f"❌ Could not connect to Backend API at `{backend_endpoint}`. Verify the server is running.")
            except Exception as e:
                st.error(f"❌ Extraction error: {str(e)}")

    if "extraction_response" in st.session_state:
        resp = st.session_state["extraction_response"]
        data = resp.get("data", {})
        doc_type = data.get("document_type", "Unknown Document")
        fields = data.get("document", {})
        tables = data.get("tables", [])
        lists = data.get("lists", [])
        warnings = data.get("warnings", [])

        # Document Type & Metrics Bar
        st.markdown(
            f"""
            <div style="background: rgba(37, 99, 235, 0.1); border: 1px solid rgba(37, 99, 235, 0.3); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
                <span style="font-size: 13px; color: #93c5fd; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">CLASSIFICATION:</span>
                <span style="font-size: 16px; color: #ffffff; font-weight: 700; margin-left: 8px;">{doc_type.upper().replace('_', ' ')}</span>
                <span style="font-size: 12px; color: #9ca3af; float: right; margin-top: 2px;">⚡ {st.session_state.get('latency', 'N/A')}s</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tab_fields, tab_tables, tab_lists, tab_json, tab_warn = st.tabs([
            f"📝 Dynamic Fields ({len(fields)})",
            f"📊 Tables ({len(tables)})",
            f"📋 Lists ({len(lists)})",
            "🔍 Raw JSON",
            f"⚠️ Warnings ({len(warnings)})",
        ])

        # Tab 1: Dynamic Fields
        with tab_fields:
            if fields:
                st.markdown("##### Discovered Key-Value Pairs")
                edited_fields = render_dynamic_form(fields, key_prefix="doc_form")
                st.download_button(
                    "💾 Download Fields JSON",
                    data=json.dumps(edited_fields, indent=2, ensure_ascii=False),
                    file_name=f"{uploaded_file.name}_fields.json",
                    mime="application/json",
                    use_container_width=True,
                )
            else:
                st.info("No key-value fields discovered in document.")

        # Tab 2: Extracted Tables
        with tab_tables:
            if tables:
                for idx, tbl in enumerate(tables, 1):
                    t_name = tbl.get("table_name", f"Table {idx}")
                    headers = tbl.get("headers", [])
                    rows = tbl.get("rows", [])
                    pages = tbl.get("pages", [1])

                    st.markdown(f"**{t_name}** *(Spanning pages: {', '.join(map(str, pages))})*")
                    if headers and rows:
                        df = pd.DataFrame(rows, columns=headers[:len(rows[0])] if len(rows) > 0 else headers)
                        st.dataframe(df, use_container_width=True)

                        # CSV Download
                        csv_data = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            f"📥 Export '{t_name}' to CSV",
                            data=csv_data,
                            file_name=f"{t_name.lower().replace(' ', '_')}.csv",
                            mime="text/csv",
                            key=f"csv_btn_{idx}",
                        )
                    else:
                        st.write(tbl)
                    st.divider()
            else:
                st.info("No tabular data detected.")

        # Tab 3: Extracted Lists
        with tab_lists:
            if lists:
                for idx, lst in enumerate(lists, 1):
                    l_name = lst.get("list_name", f"List {idx}")
                    items = lst.get("items", [])
                    st.markdown(f"##### {l_name}")
                    for item in items:
                        st.markdown(f"- {item}")
                    st.divider()
            else:
                st.info("No lists or enumerations detected.")

        # Tab 4: Raw JSON
        with tab_json:
            st.json(data)
            st.download_button(
                "💾 Download Full Extracted JSON",
                data=json.dumps(data, indent=2, ensure_ascii=False),
                file_name=f"{uploaded_file.name}_extracted.json",
                mime="application/json",
                use_container_width=True,
            )

        # Tab 5: Warnings
        with tab_warn:
            if warnings:
                for w in warnings:
                    st.warning(w)
            else:
                st.success("✅ No visual or JSON anomalies detected.")

    elif not extract_btn:
        st.info("👈 Upload a document and click **'Run Dynamic Extraction'** to begin.")
