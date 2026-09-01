"""Universal Schema-Agnostic Dynamic Extraction Prompt Builder.

Instructs Vision-Language Models (e.g. Qwen2.5-VL) to dynamically discover
and extract structured information from any document type without hardcoded schemas.
"""

import json
from typing import Any, List, Optional
from backend.schemas.extraction import DynamicFieldDefinition, ExtractionSchema


class DynamicPromptBuilder:
    """Universal prompt builder for dynamic multimodal document intelligence."""

    SYSTEM_PROMPT = (
        "You are an expert enterprise document intelligence AI system. "
        "Your task is to analyze document images with extreme visual and textual precision. "
        "You must discover and extract all visible structured information dynamically. "
        "Do NOT assume or force any predefined schema or invoice format unless explicitly guided. "
        "Strictly output a single, complete, valid JSON object enclosed in a ```json codeblock."
    )

    @classmethod
    def build_extraction_prompt(
        cls,
        schema: Optional[ExtractionSchema] = None,
        custom_instructions: Optional[str] = None,
        page_num: Optional[int] = None,
        total_pages: Optional[int] = None,
    ) -> str:
        """Build the dynamic extraction prompt instructing the VLM to discover all fields dynamically."""
        prompt_parts = []

        # Multi-page context header if applicable
        if page_num is not None and total_pages is not None:
            prompt_parts.append(
                f"### DOCUMENT ANALYSIS TASK (Page {page_num} of {total_pages})\n"
                "Carefully inspect everything visible on this page."
            )
        else:
            prompt_parts.append(
                "### DOCUMENT ANALYSIS TASK\n"
                "Carefully inspect everything visible in the provided document image(s)."
            )

        prompt_parts.append(
            "\nYou MUST output a single valid JSON object following this exact outer envelope:\n"
            "```json\n"
            "{\n"
            '  "document_type": "<auto-detected classification: e.g. receipt, purchase_order, bill_of_lading, contract, form, identity_card, certificate, report, tax_invoice, etc.>",\n'
            '  "document": {\n'
            '    "<exact_field_name_from_document>": "<extracted_value_or_nested_object>"\n'
            "  },\n"
            '  "tables": [\n'
            '    {\n'
            '      "table_name": "<descriptive_table_or_section_name>",\n'
            '      "headers": ["Column 1", "Column 2"],\n'
            '      "rows": [["Row1 Col1", "Row1 Col2"]]\n'
            '    }\n'
            "  ],\n"
            '  "lists": [\n'
            '    {\n'
            '      "list_name": "<descriptive_list_or_clause_name>",\n'
            '      "items": ["Item 1", "Item 2"]\n'
            '    }\n'
            "  ],\n"
            '  "warnings": []\n'
            "}\n"
            "```"
        )

        if schema and schema.fields:
            prompt_parts.append(
                f"\n### TARGET CONTEXT: {schema.schema_name or 'Custom Schema'}"
            )
            if schema.description:
                prompt_parts.append(f"Context / Instructions: {schema.description}")

            prompt_parts.append("\nExtract the target fields into the `document` object following this template:")
            schema_template = cls._format_fields_template(schema.fields)
            prompt_parts.append(f"```json\n{schema_template}\n```")
        else:
            prompt_parts.append(
                "\n### DYNAMIC DISCOVERY RULES:\n"
                "1. Document Type: Identify and classify the document accurately based on visual and textual cues.\n"
                "2. Dynamic Document Fields (`document`): Discover all key-value pairs, identifiers, reference codes, parties/organizations, dates, terms, and summary metrics. Preserve original field labels found in the document.\n"
                "3. Tabular Data (`tables`): Extract all grids, line items, matrix comparisons, and schedules with exact column headers and row values.\n"
                "4. Enumerated Items (`lists`): Extract bullet points, numbered requirements, deliverable clauses, and remarks.\n"
                "5. Data Fidelity: Preserve numbers, codes, and text exactly as written. Do NOT hallucinate, assume, or invent values.\n"
                "6. Extraction Warnings (`warnings`): If any text is torn, blurred, cropped, handwritten/illegible, or ambiguous, log it in the warnings array."
            )

        if custom_instructions:
            prompt_parts.append(f"\n### SPECIAL USER INSTRUCTIONS:\n{custom_instructions}")

        prompt_parts.append(
            "\n### STRICT FORMATTING REQUIREMENTS:\n"
            "- Return ONLY valid JSON inside ```json ... ```.\n"
            "- Ensure all JSON keys and strings use standard double quotes.\n"
            "- Separate all key-value pairs and array items with commas."
        )

        return "\n".join(prompt_parts)

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
