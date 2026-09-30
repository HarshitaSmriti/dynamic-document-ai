"""Universal Schema-Agnostic Dynamic Extraction Prompt Builder.

Instructs Vision-Language Models (e.g. Qwen2.5-VL) to dynamically discover
and extract structured information from any document type with zero hallucination.
"""

import json
from typing import Any, List, Optional
from backend.schemas.extraction import DynamicFieldDefinition, ExtractionSchema


class DynamicPromptBuilder:
    """Universal prompt builder for compact, high-precision document intelligence."""

    SYSTEM_PROMPT = (
        "You are an expert enterprise document extraction engine. "
        "Analyze the supplied document image with exact textual and visual fidelity. "
        "Return ONLY valid JSON. Never hallucinate or invent missing information. "
        "Do not provide commentary, markdown explanations, or preamble outside the JSON."
    )

    @classmethod
    def build_extraction_prompt(
        cls,
        schema: Optional[ExtractionSchema] = None,
        custom_instructions: Optional[str] = None,
        page_num: Optional[int] = None,
        total_pages: Optional[int] = None,
    ) -> str:
        """Build a concise, production-grade extraction prompt."""
        parts = []

        if page_num is not None and total_pages is not None:
            parts.append(f"DOCUMENT PAGE {page_num} OF {total_pages}:")
        else:
            parts.append("DOCUMENT EXTRACTION TASK:")

        parts.append(
            "Analyze the document image and extract all visible data into this exact JSON structure:\n"
            "```json\n"
            "{\n"
            '  "document_type": "<inferred classification: e.g. tax_invoice, purchase_order, receipt, contract, bill_of_lading, id_card, report, form, etc.>",\n'
            '  "document": {\n'
            '    "<field_name>": "<exact_extracted_value_or_null>"\n'
            "  },\n"
            '  "tables": [\n'
            '    {\n'
            '      "table_name": "<table_name>",\n'
            '      "headers": ["Col1", "Col2"],\n'
            '      "rows": [["Val1", "Val2"]]\n'
            '    }\n'
            "  ],\n"
            '  "lists": [\n'
            '    {\n'
            '      "list_name": "<list_name>",\n'
            '      "items": ["Item1", "Item2"]\n'
            '    }\n'
            "  ],\n"
            '  "warnings": []\n'
            "}\n"
            "```"
        )

        if schema and schema.fields:
            parts.append(f"\nTARGET SCHEMA CONTEXT ({schema.schema_name or 'Custom Schema'}):")
            if schema.description:
                parts.append(f"Guidance: {schema.description}")
            parts.append("Extract target fields into 'document' matching this schema:")
            schema_template = cls._format_fields_template(schema.fields)
            parts.append(f"```json\n{schema_template}\n```")
        else:
            parts.append(
                "\nEXTRACTION RULES:\n"
                "1. Extract all visible entities, key-value pairs, identifiers, monetary figures, dates, and parties.\n"
                "2. Detect all structured tables into 'tables' (headers + rows arrays).\n"
                "3. Detect enumerated clauses, terms, or bullet points into 'lists'.\n"
                "4. Preserve exact textual numbers, punctuation, codes, and currencies.\n"
                "5. Never invent or hallucinate data; use null for unavailable fields.\n"
                "6. If text is blurry, occluded, or ambiguous, log a note in 'warnings'."
            )

        if custom_instructions and custom_instructions.strip():
            parts.append(f"\nUSER EXTRACTION GUIDANCE:\n{custom_instructions.strip()}")

        parts.append(
            "\nOUTPUT FORMAT:\n"
            "- Output ONLY valid JSON inside ```json ... ```.\n"
            "- Double quote all keys and string values.\n"
            "- Do not add explanations or conversational filler."
        )

        return "\n".join(parts)

    @classmethod
    def _format_fields_template(cls, fields: List[DynamicFieldDefinition], indent: int = 2) -> str:
        """Format field definitions into a JSON schema template."""
        def _field_to_spec(field: DynamicFieldDefinition) -> Any:
            type_str = field.type.value if hasattr(field.type, "value") else str(field.type)
            desc = f" ({field.description})" if field.description else ""
            if field.nested_fields:
                if type_str == "list":
                    return [_field_to_spec(f) for f in field.nested_fields]
                return {f.name: _field_to_spec(f) for f in field.nested_fields}
            return f"<{type_str}>{desc}"

        template_dict = {f.name: _field_to_spec(f) for f in fields}
        return json.dumps(template_dict, indent=indent)
