"""Universal Schema-Agnostic Dynamic Extraction Prompt Builder.

Instructs Vision-Language Models (e.g. Qwen2.5-VL) to dynamically discover
and extract structured information from any document type with zero hallucination.
"""

import json
from typing import Any, List, Optional
from backend.schemas.extraction import DynamicFieldDefinition, ExtractionSchema


class DynamicPromptBuilder:
    """Universal prompt builder for compact, high-precision document intelligence."""

    SYSTEM_PROMPT = "You are an enterprise document extraction engine. Return ONLY valid JSON matching the exact schema."

    @classmethod
    def build_extraction_prompt(
        cls,
        schema: Optional[ExtractionSchema] = None,
        custom_instructions: Optional[str] = None,
        page_num: Optional[int] = None,
        total_pages: Optional[int] = None,
    ) -> str:
        """Build an ultra-compact extraction prompt to conserve token budget."""
        parts = []

        if page_num is not None and total_pages is not None:
            parts.append(f"Page {page_num}/{total_pages}:")

        parts.append(
            "Extract all visible data into this exact JSON schema:\n"
            "```json\n"
            "{\n"
            '  "document_type": "invoice|receipt|report|contract|other",\n'
            '  "document": {"field_name": "extracted_value"},\n'
            '  "tables": [{"table_name": "name", "headers": ["col1"], "rows": [["val1"]]}],\n'
            '  "lists": [{"list_name": "name", "items": ["item1"]}],\n'
            '  "warnings": []\n'
            "}\n"
            "```\n"
            "Rules: Extract exact numbers, dates, parties, line items into tables. Return ONLY JSON."
        )

        if schema and schema.fields:
            parts.append("Target fields: " + ", ".join(f.name for f in schema.fields))

        if custom_instructions and custom_instructions.strip():
            parts.append(f"Guidance: {custom_instructions.strip()[:100]}")

        return "\n".join(parts)
