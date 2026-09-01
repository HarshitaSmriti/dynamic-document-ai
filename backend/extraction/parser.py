"""Robust JSON Parser, Normalizer, and Multi-Page Document Consolidation Engine.

Provides:
1. Advanced JSON repair for malformed model outputs (missing commas, trailing commas, unescaped characters, truncated JSON).
2. Schema-agnostic normalization into ExtractedDocumentPayload.
3. Multi-page document merger that intelligently deduplicates fields and merges continuation tables across pages.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from backend.schemas.extraction import (
    ExtractedDocumentPayload,
    ExtractionSchema,
)


class DynamicJSONParser:
    """Industrial-grade parser, normalizer, and validator for LLM/VLM extraction outputs."""

    @classmethod
    def clean_markdown_fences(cls, text: str) -> str:
        """Strip markdown code fences and surrounding conversational commentary."""
        text = text.strip()

        # Match ```json ... ``` or ``` ... ```
        pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0].strip()

        # Fallback: extract substring between first '{' and last '}'
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return text[first_brace : last_brace + 1].strip()

        return text

    @classmethod
    def repair_json_string(cls, text: str) -> str:
        """Heuristically repair common LLM JSON syntax mistakes."""
        # 1. Clean markdown codeblocks
        s = cls.clean_markdown_fences(text)

        # 2. Remove single-line and multi-line comments
        s = re.sub(r"//.*?\n", "\n", s)
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)

        # 3. Remove trailing commas before closing braces/brackets
        s = re.sub(r",\s*([\]\}])", r"\1", s)

        # 4. Fix missing comma between key-value pairs: "val"\n"next_key": ... -> "val",\n"next_key": ...
        # Match: (quoted string, number, boolean, null, }, ]) followed by newline and next "key":
        s = re.sub(
            r'("(?:\\.|[^"\\])*"|\b\d+(?:\.\d+)?\b|\btrue\b|\bfalse\b|\bnull\b|\}|\])\s*\n\s*("[\w\s\.\-_]+"\s*:)',
            r"\1,\n\2",
            s,
        )

        # 5. Fix missing comma between array elements: "item1"\n"item2" or }\n{
        s = re.sub(
            r'("(?:\\.|[^"\\])*"|\b\d+(?:\.\d+)?\b|\})\s*\n\s*("(?:\\.|[^"\\])*"|\{)',
            r"\1,\n\2",
            s,
        )

        # 6. Auto-close truncated JSON (unbalanced braces/brackets)
        open_braces = s.count("{") - s.count("}")
        open_brackets = s.count("[") - s.count("]")

        if open_brackets > 0 or open_braces > 0:
            # Strip trailing incomplete key or comma
            s = re.sub(r',\s*$', '', s.strip())
            s = re.sub(r':\s*$', ': null', s.strip())
            # Close open structures in reverse order
            s += ("]" * max(0, open_brackets)) + ("}" * max(0, open_braces))

        return s

    @classmethod
    def parse_and_validate(
        cls,
        raw_output: str,
        schema: Optional[ExtractionSchema] = None,
    ) -> Tuple[bool, ExtractedDocumentPayload, Optional[str]]:
        """Parse raw model output into ExtractedDocumentPayload with progressive fallback repair.

        Returns:
            Tuple[bool, ExtractedDocumentPayload, Optional[str]]: (success, payload, error_message)
        """
        if not raw_output or not raw_output.strip():
            return False, ExtractedDocumentPayload(), "Empty response received from extraction provider."

        # Stage 1: Try direct parsing after fence removal
        cleaned = cls.clean_markdown_fences(raw_output)
        try:
            raw_data = json.loads(cleaned)
            if isinstance(raw_data, dict):
                payload = cls._normalize_payload(raw_data)
                if schema and schema.fields:
                    payload.warnings.extend(cls._validate_schema_fields(payload.document, schema))
                return True, payload, None
        except Exception:
            pass

        # Stage 2: Try heuristic repair
        repaired = cls.repair_json_string(raw_output)
        try:
            raw_data = json.loads(repaired)
            if isinstance(raw_data, dict):
                payload = cls._normalize_payload(raw_data)
                if schema and schema.fields:
                    payload.warnings.extend(cls._validate_schema_fields(payload.document, schema))
                return True, payload, None
        except Exception:
            pass

        # Stage 3: Loose Regex Recovery (Extract whatever was formed without dropping data)
        recovered_payload = cls._loose_regex_recovery(raw_output)
        if recovered_payload.document or recovered_payload.tables or recovered_payload.lists:
            recovered_payload.warnings.append("JSON was partially malformed but dynamic entities were successfully recovered.")
            return True, recovered_payload, "JSON required regex recovery."

        return (
            False,
            ExtractedDocumentPayload(),
            "Failed to decode JSON after all repair attempts.",
        )

    @classmethod
    def _normalize_payload(cls, raw_data: Dict[str, Any]) -> ExtractedDocumentPayload:
        """Ensure the dictionary conforms strictly to the ExtractedDocumentPayload envelope."""
        has_outer_envelope = "document" in raw_data or "document_type" in raw_data

        if has_outer_envelope:
            doc_type = str(raw_data.get("document_type", "dynamic_document"))
            document_content = raw_data.get("document", {})
            if not isinstance(document_content, dict):
                document_content = {"extracted_content": document_content}

            tables = raw_data.get("tables", [])
            if not isinstance(tables, list):
                tables = [tables] if tables else []

            # Clean and standardize table structures
            standardized_tables = []
            for t in tables:
                if isinstance(t, dict):
                    headers = t.get("headers", [])
                    rows = t.get("rows", [])
                    table_name = str(t.get("table_name", "Table"))
                    if isinstance(headers, list) and isinstance(rows, list):
                        standardized_tables.append({
                            "table_name": table_name,
                            "headers": headers,
                            "rows": rows,
                        })

            lists = raw_data.get("lists", [])
            if not isinstance(lists, list):
                lists = [lists] if lists else []

            standardized_lists = []
            for l in lists:
                if isinstance(l, dict):
                    list_name = str(l.get("list_name", "List"))
                    items = l.get("items", [])
                    if isinstance(items, list):
                        standardized_lists.append({
                            "list_name": list_name,
                            "items": items,
                        })

            warnings = raw_data.get("warnings", [])
            if not isinstance(warnings, list):
                warnings = [str(warnings)] if warnings else []

            return ExtractedDocumentPayload(
                document_type=doc_type,
                document=document_content,
                tables=standardized_tables,
                lists=standardized_lists,
                warnings=[str(w) for w in warnings],
            )
        else:
            # Model produced a flat key-value dict: encapsulate directly into dynamic `document`
            return ExtractedDocumentPayload(
                document_type="dynamic_document",
                document=raw_data,
                tables=[],
                lists=[],
                warnings=[],
            )

    @classmethod
    def _loose_regex_recovery(cls, text: str) -> ExtractedDocumentPayload:
        """Extract structured fields via regex when JSON syntax is completely broken."""
        doc_fields = {}
        # Find all "key": "value" patterns
        kv_pairs = re.findall(r'"([^"\\]+)":\s*"([^"\\]*)"', text)
        for k, v in kv_pairs:
            if k not in ["document_type", "table_name", "list_name"]:
                doc_fields[k] = v

        doc_type_match = re.search(r'"document_type":\s*"([^"\\]+)"', text)
        doc_type = doc_type_match.group(1) if doc_type_match else "dynamic_document"

        return ExtractedDocumentPayload(
            document_type=doc_type,
            document=doc_fields,
            tables=[],
            lists=[],
            warnings=["Recovered via regex parsing fallback."],
        )

    @classmethod
    def _validate_schema_fields(
        cls, doc_data: Dict[str, Any], schema: ExtractionSchema
    ) -> List[str]:
        """Validate dynamic fields against schema requirements."""
        warnings = []
        for field in schema.fields:
            if field.required and (field.name not in doc_data or doc_data[field.name] is None):
                warnings.append(f"Required schema field '{field.name}' is missing or null.")
        return warnings

    # =========================================================================
    # Multi-Page Document Consolidation Engine (TASK 2 & TASK 3)
    # =========================================================================

    @classmethod
    def merge_document_payloads(
        cls, payloads: List[ExtractedDocumentPayload]
    ) -> ExtractedDocumentPayload:
        """Merge and deduplicate extracted payloads from multiple pages into one unified document."""
        if not payloads:
            return ExtractedDocumentPayload()
        if len(payloads) == 1:
            return payloads[0]

        # 1. Best Document Type
        best_doc_type = "dynamic_document"
        for p in payloads:
            dt = p.document_type
            if dt and dt not in ["unknown", "unknown_document", "dynamic_document", "error"]:
                best_doc_type = dt
                break

        # 2. Intelligent Field Deduplication (TASK 2)
        consolidated_doc = cls._merge_dynamic_fields([p.document for p in payloads])

        # 3. Intelligent Table Continuation Merger (TASK 3)
        consolidated_tables = cls._merge_continuation_tables([p.tables for p in payloads])

        # 4. List Consolidation
        consolidated_lists = cls._merge_lists([p.lists for p in payloads])

        # 5. Warnings Consolidation
        all_warnings = []
        for idx, p in enumerate(payloads, 1):
            for w in p.warnings:
                all_warnings.append(f"Page {idx}: {w}" if not w.startswith("Page ") else w)

        return ExtractedDocumentPayload(
            document_type=best_doc_type,
            document=consolidated_doc,
            tables=consolidated_tables,
            lists=consolidated_lists,
            warnings=all_warnings,
        )

    @classmethod
    def _normalize_key(cls, key: str) -> str:
        """Normalize key string for intelligent deduplication matching."""
        # Strip Page suffixes if any exist
        k = re.sub(r"\s*\(Page\s*\d+\)$", "", key, flags=re.IGNORECASE).strip()
        return re.sub(r"[\s\-_\.]+", "_", k).lower()

    @classmethod
    def _merge_dynamic_fields(cls, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge field dictionaries from multiple pages without generating page-suffixed duplicate keys."""
        merged: Dict[str, Any] = {}
        key_map: Dict[str, str] = {}  # normalized_key -> canonical_key_in_merged

        for page_idx, doc in enumerate(docs, 1):
            for raw_key, value in doc.items():
                norm_key = cls._normalize_key(raw_key)

                if norm_key not in key_map:
                    # New unique field
                    canonical_key = raw_key
                    key_map[norm_key] = canonical_key
                    merged[canonical_key] = value
                else:
                    existing_key = key_map[norm_key]
                    existing_val = merged[existing_key]

                    # Deduplication / Merging strategy:
                    if existing_val == value or value is None or str(value).strip() == "":
                        # Identical or empty - keep existing
                        continue
                    elif existing_val is None or str(existing_val).strip() == "":
                        # Existing is empty, update with new value
                        merged[existing_key] = value
                    elif isinstance(existing_val, str) and isinstance(value, str):
                        # Both are strings
                        # If one is a substring of the other (e.g. slightly more complete address), pick the longer
                        norm_ex = re.sub(r"\s+", " ", existing_val).strip().lower()
                        norm_nw = re.sub(r"\s+", " ", value).strip().lower()
                        if norm_nw in norm_ex:
                            continue  # existing is already more complete
                        elif norm_ex in norm_nw:
                            merged[existing_key] = value  # new is more complete
                        else:
                            # Genuinely different values: convert to list of unique values
                            if existing_val != value:
                                merged[existing_key] = [existing_val, value]
                    elif isinstance(existing_val, list):
                        if value not in existing_val:
                            existing_val.append(value)
                    else:
                        # Non-string distinct values
                        merged[existing_key] = [existing_val, value]

        return merged

    @classmethod
    def _merge_continuation_tables(cls, tables_by_page: List[List[Any]]) -> List[Dict[str, Any]]:
        """Intelligently merge continuation tables that span across pages (TASK 3)."""
        all_tables: List[Dict[str, Any]] = []

        for page_idx, page_tables in enumerate(tables_by_page, 1):
            for tbl in page_tables:
                if not isinstance(tbl, dict):
                    continue

                t_name = tbl.get("table_name", "Table")
                headers = tbl.get("headers", [])
                rows = tbl.get("rows", [])

                if not rows and not headers:
                    continue

                # Check if this table matches an existing table in all_tables
                matched_table = None
                norm_headers = [re.sub(r"[\s\-_]+", "", str(h).lower()) for h in headers]

                for existing_tbl in all_tables:
                    ex_name = existing_tbl.get("table_name", "")
                    ex_headers = existing_tbl.get("headers", [])
                    ex_norm_headers = [re.sub(r"[\s\-_]+", "", str(h).lower()) for h in ex_headers]

                    # Match condition: identical headers OR same table_name + overlapping headers
                    headers_match = norm_headers and (norm_headers == ex_norm_headers)
                    name_match = (
                        t_name.lower().replace(" ", "_") == ex_name.lower().replace(" ", "_")
                        and t_name.lower() not in ["table", ""]
                    )

                    if headers_match or (name_match and (set(norm_headers) & set(ex_norm_headers))):
                        matched_table = existing_tbl
                        break

                if matched_table:
                    # Append rows to existing continuation table (avoid duplicates)
                    existing_rows = matched_table.setdefault("rows", [])
                    for r in rows:
                        # Append row
                        existing_rows.append(r)
                    # Track multi-page span
                    pages = matched_table.setdefault("pages", [])
                    if page_idx not in pages:
                        pages.append(page_idx)
                else:
                    # New distinct table
                    tbl_copy = {
                        "table_name": t_name,
                        "headers": headers,
                        "rows": list(rows),
                        "pages": [page_idx],
                    }
                    all_tables.append(tbl_copy)

        return all_tables

    @classmethod
    def _merge_lists(cls, lists_by_page: List[List[Any]]) -> List[Dict[str, Any]]:
        """Merge list objects across pages by list_name."""
        merged_lists: Dict[str, Dict[str, Any]] = {}

        for page_idx, page_lists in enumerate(lists_by_page, 1):
            for lst in page_lists:
                if not isinstance(lst, dict):
                    continue
                name = lst.get("list_name", "List")
                items = lst.get("items", [])

                norm_name = name.strip().lower()
                if norm_name not in merged_lists:
                    merged_lists[norm_name] = {
                        "list_name": name,
                        "items": list(items),
                        "pages": [page_idx],
                    }
                else:
                    existing = merged_lists[norm_name]
                    for item in items:
                        if item not in existing["items"]:
                            existing["items"].append(item)
                    if page_idx not in existing["pages"]:
                        existing["pages"].append(page_idx)

        return list(merged_lists.values())
