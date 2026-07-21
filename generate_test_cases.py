"""Generate Q&A test cases from an invoice PDF.

Usage:
    python generate_test_cases.py track1-4.pdf

Outputs:
    invoice_docs.txt          — OCR text for RAG ingestion
    tests/invoice_test_cases.json — 22-field Q&A test cases
"""

import json
import os
import re
import sys

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from langchain_ollama import ChatOllama

LLM_MODEL = "qwen2.5:1.5b"

FIELD_QUESTIONS = {
    "invoice_type": "What is the type of this invoice?",
    "invoice_number": "What is the invoice number?",
    "invoice_date": "What is the invoice date?",
    "goods_description": "What goods are described in the invoice?",
    "quantity_of_goods": "What is the quantity of goods?",
    "units_of_quantity": "What are the units of quantity?",
    "total_amount": "What is the total invoice amount?",
    "currency": "What currency is the invoice in?",
    "payment_terms": "What are the payment terms?",
    "beneficiary_name": "Who is the beneficiary?",
    "beneficiary_country": "What country is the beneficiary in?",
    "beneficiary_signature_present": "Is the beneficiary's signature present on the document?",
    "applicant_name": "Who is the applicant or buyer?",
    "applicant_signature_present": "Is the applicant's signature present on the document?",
    "remitter_name": "Who is the remitter?",
    "remitter_country": "What country is the remitter in?",
    "beneficiary_bank_name": "What is the beneficiary bank name?",
    "beneficiary_bank_account_number": "What is the beneficiary bank account number?",
    "beneficiary_bank_country": "What country is the beneficiary bank in?",
    "beneficiary_swift_code": "What is the beneficiary bank swift code?",
    "intermediary_bank_name": "What is the intermediary bank name?",
    "intermediary_swift": "What is the intermediary bank swift code?",
}

REGEX_FIELDS = {
    "invoice_number": [
        r"(?:Invoice\s*)?No\.?:\s*([\w\-/]+)",
        r"Invoice\s+Number[:\s]+([\w\-/]+)",
    ],
    "invoice_date": [
        r"Date:\s*(\d{1,2}(?:st|nd|rd|th)?[\s.,]+\w+[\s.,]+\d{4})",
        r"Date:\s*([\d]+\w*[\s.,]+\w+[\s.,]+\d{4})",
    ],
    "invoice_type": [
        r"(Commercial\s+Invoice)",
        r"(Pro\s*Forma\s+Invoice)",
        r"(Proforma\s+Invoice)",
    ],
    "total_amount": [
        r"TOTAL\s+[\d,]+\s+([\d,]+\.\d{2})",
        r"TOTAL\s+([\d,]+\.\d{2})",
    ],
    "currency": [
        r"\b(USD)\b",
        r"\b(EUR)\b",
        r"\b(GBP)\b",
        r"UNITED\s+STATES\s+DOLLAR",
    ],
    "quantity_of_goods": [
        r"TOTAL\s+([\d,]+)\s+[\d,]+\.\d{2}",
        r"TOTAL\s+([\d,]+)",
    ],
    "units_of_quantity": [
        r"\b(\d+)\s*(KGS|KG|LBS|BAGS|PIECES|PCS|MT|TONS)\b",
    ],
    "beneficiary_swift_code": [
        r"Swift\s*Code[:\s]+(\w+)",
        r"SWIFT[:\s]+(\w+)",
    ],
    "beneficiary_bank_account_number": [
        r"Account\s*Number[:\s]+(\d+)",
    ],
    "payment_terms": [
        r"Payment[:\s]*[:\s]*(.+?)(?:\n|$)",
        r"Terms[:\s]*(?:of\s+)?Payment[:\s]*(.+?)(?:\n|$)",
    ],
}

LLM_FIELDS = [
    "goods_description",
    "beneficiary_name",
    "beneficiary_country",
    "beneficiary_signature_present",
    "applicant_name",
    "applicant_signature_present",
    "remitter_name",
    "remitter_country",
    "beneficiary_bank_name",
    "beneficiary_bank_country",
    "intermediary_bank_name",
    "intermediary_swift",
]


def extract_text(pdf_path: str) -> str:
    print(f"  OCR: running docTR on {pdf_path}...")
    model = ocr_predictor(pretrained=True)
    doc = DocumentFile.from_pdf(pdf_path)
    result = model(doc)
    text = result.render()
    print(f"  OCR: extracted {len(text)} characters")
    return text


def extract_regex(text: str) -> dict:
    fields = {}
    for field, patterns in REGEX_FIELDS.items():
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if field == "currency" and value == "UNITED STATES DOLLAR":
                    value = "USD"
                fields[field] = value
                break
        if field not in fields:
            fields[field] = "NOT_PRESENT"
    return fields


def extract_llm(text: str, llm, fields_to_extract: list) -> dict:
    fields = {}
    for field in fields_to_extract:
        question = FIELD_QUESTIONS[field]
        prompt = (
            "You are an invoice data extraction assistant. "
            "Given the invoice text below, extract the value for the field described. "
            "If the field is not present in the document, reply ONLY with NOT_PRESENT. "
            "Do not add explanations.\n\n"
            f"Invoice text:\n{text[:3000]}\n\n"
            f"Field: {question}\n"
            "Value:"
        )
        raw = llm.invoke(prompt)
        result = raw.content if hasattr(raw, "content") else str(raw)
        result = result.strip().strip('"').strip("'")
        if "not present" in result.lower() or "not found" in result.lower():
            fields[field] = "NOT_PRESENT"
        else:
            fields[field] = result
        print(f"  {field}: {fields[field][:80]}")
    return fields


def validate_fields(fields: dict, regex_fields: dict) -> dict:
    validated = dict(fields)

    swift = regex_fields.get("beneficiary_swift_code", "")
    if swift and not swift.startswith("NOT"):
        if validated.get("intermediary_swift", "") == swift:
            validated["intermediary_swift"] = "NOT_PRESENT"
        if re.match(r"^[A-Z0-9]{6,12}$", validated.get("intermediary_bank_name", "")):
            validated["intermediary_bank_name"] = "NOT_PRESENT"

    desc = validated.get("goods_description", "")
    desc_lower = desc.lower()
    if "91 inches" in desc_lower or "121 inches" in desc_lower:
        desc = re.sub(r"91\s+INCHES", "9 INCHES", desc, flags=re.IGNORECASE)
        desc = re.sub(r"121\s+INCHES", "12 INCHES", desc, flags=re.IGNORECASE)
        validated["goods_description"] = desc

    return validated


def build_test_cases(all_fields: dict) -> list:
    cases = []
    for field, question in FIELD_QUESTIONS.items():
        expected = all_fields.get(field, "NOT_PRESENT")
        cases.append({"question": question, "expected": expected, "field": field})
    return cases


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_test_cases.py <path_to_invoice.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        sys.exit(1)

    project_root = os.path.dirname(os.path.abspath(__file__))
    output_text = os.path.join(project_root, "invoice_docs.txt")
    output_json = os.path.join(project_root, "tests", "invoice_test_cases.json")

    print("=" * 60)
    print("  Invoice Field Extraction Pipeline")
    print("=" * 60)
    print(f"  Input:  {pdf_path}")
    print(f"  Output: {output_json}")
    print("=" * 60)

    print("\n[1/4] OCR extraction...")
    text = extract_text(pdf_path)
    with open(output_text, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved OCR text to {output_text}")

    print("\n[2/4] Regex field extraction...")
    regex_fields = extract_regex(text)
    for k, v in regex_fields.items():
        print(f"  {k}: {v[:80]}")

    print("\n[3/4] LLM field extraction...")
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    llm_fields = extract_llm(text, llm, LLM_FIELDS)

    all_fields = {**regex_fields, **llm_fields}
    all_fields = validate_fields(all_fields, regex_fields)

    print("\n[4/4] Generating test cases...")
    test_cases = build_test_cases(all_fields)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2)

    present = sum(1 for v in all_fields.values() if v != "NOT_PRESENT")
    missing = sum(1 for v in all_fields.values() if v == "NOT_PRESENT")
    print(f"\n  Generated {len(test_cases)} test cases")
    print(f"  Fields found: {present}")
    print(f"  Fields missing: {missing}")
    print(f"  Saved to {output_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
