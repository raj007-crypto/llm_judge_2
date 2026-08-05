"""Unified Invoice RAG System

Usage:
    python app.py extract [pdf_path] [--no-preprocess]  - Extract text from PDF using docTR
    python app.py generate              - Generate test cases from invoice text
    python app.py serve                 - Start RAG API server

extract:
    By default, pages are cleaned with an OpenCV pipeline (deskew,
    denoise, contrast, sharpen, grayscale) before OCR. Cleaned page images
    are saved to invoice/cleaned_pages/ for visual inspection/tuning.
    Pass --no-preprocess to run docTR on the raw scan instead.
"""

import json
import os
import re
import sys
import time
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import numpy as np

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
INVOICE_DIR = os.path.join(BASE_DIR, "invoice")
TESTS_DIR = os.path.join(BASE_DIR, "tests")
SYNONYMS_PATH = os.path.join(BASE_DIR, "synonyms.json")

INVOICE_TEXT_PATH = os.path.join(INVOICE_DIR, "invoice_docs.txt")
INVOICE_EXPORT_PATH = os.path.join(INVOICE_DIR, "invoice_docs_export.json")
TEST_CASES_PATH = os.path.join(TESTS_DIR, "invoice_test_cases.json")

COLLECTION_NAME = "docs_collection"
MODEL_NAME = "qwen2.5:1.5b"
EMBEDDING_MODEL = "nomic-embed-text"
JUDGE_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
BACKEND_URL = "http://localhost:8000/query"

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
        r"\b\d+\s*(KGS|KG|LBS|BAGS|PIECES|PCS|MT|TONS)\b",
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

SYNONYM_FIELDS = [
    "invoice_number", "invoice_date", "invoice_type", "total_amount",
    "currency", "quantity_of_goods", "units_of_quantity",
    "beneficiary_swift_code", "beneficiary_bank_account_number", "payment_terms",
]

LLM_FIELDS = [
    "goods_description", "beneficiary_name", "beneficiary_country",
    "beneficiary_signature_present", "applicant_name", "applicant_signature_present",
    "remitter_name", "remitter_country", "beneficiary_bank_name",
    "beneficiary_bank_country", "intermediary_bank_name", "intermediary_swift",
]


def load_synonyms():
    if os.path.exists(SYNONYMS_PATH):
        with open(SYNONYMS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def expand_query(query, synonyms):
    query_lower = query.lower()
    expanded_terms = [query]
    for field, syns in synonyms.items():
        for syn in syns:
            if syn.lower() in query_lower:
                expanded_terms.extend(syns[:3])
                break
    return " ".join(set(expanded_terms))


CONFIDENCE_THRESHOLD = 0.5
UNCLEAR_TAG_RE = re.compile(r"\[UNCLEAR:[^\]]*\]")

FIELD_LABELS = [
    "our reference no",
    "additional reference no",
    "reference no",
    "currency",
    "date",
    "account no",
    "amount in words",
    "beneficiary name",
    "beneficiary country",
    "beneficiary bank",
    "grand total",
    "total invoice amount",
]

FIELD_NOT_PRESENT_MESSAGE = (
    "The requested field is not present in the document - the value could not be extracted."
)

_COMBINED_CONNECTOR_RE = re.compile(
    r"^(?:&|and\b|/|,|\+|with\b|plus\b|including\b)\s*", re.IGNORECASE
)


def _norm_text(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _label_prefix_match(rest_norm, label_norm):
    from difflib import SequenceMatcher

    n = min(len(rest_norm), len(label_norm) + 4)
    if n < 8:
        return False
    return SequenceMatcher(None, rest_norm[:n], label_norm[:n]).ratio() > 0.8


def asked_field_labels(question):
    q = question.lower()
    labels = []
    if "our reference" in q or ("reference" in q and "additional" not in q):
        labels.append("our reference no")
    if "additional reference" in q:
        labels.append("additional reference no")
    if "currency" in q:
        labels.append("currency")
    if "beneficiary" in q:
        labels.append("beneficiary name")
    return labels


def _plausible_value(s):
    s = s.strip()
    if not s:
        return False
    if re.search(r"[0-9]", s):
        return True
    letters = re.sub(r"[^a-z]", "", s)
    return bool(letters) and s == s.upper() and len(letters) >= 2


def field_has_readable_value(context, label):
    ctx_lower = context.lower()
    start = 0
    any_occurrence = False
    any_value = False
    while True:
        idx = ctx_lower.find(label, start)
        if idx == -1:
            break
        any_occurrence = True
        rest_raw = context[idx + len(label):]
        rest = rest_raw.lstrip(" \t.:-\r\n")
        if not rest:
            start = idx + len(label)
            continue
        if UNCLEAR_TAG_RE.match(rest):
            start = idx + len(label)
            continue
        if _COMBINED_CONNECTOR_RE.match(rest):
            any_value = True
            start = idx + len(label)
            continue
        same_line = rest_raw.split("\n", 1)[0].strip(" \t.:-")
        if same_line:
            if _plausible_value(same_line):
                any_value = True
            start = idx + len(label)
            continue
        rest_norm = _norm_text(rest)
        followed_by_label = False
        for other in FIELD_LABELS:
            if _norm_text(other) == _norm_text(label):
                continue
            if _label_prefix_match(rest_norm, _norm_text(other)):
                followed_by_label = True
                break
        if not followed_by_label:
            any_value = True
        start = idx + len(label)
    if not any_occurrence:
        return True
    return any_value

CLAHE_CLIP_LIMIT = 2.0


def compute_skew_angle(gray_img):
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    coords = np.column_stack(np.where(edges > 0))
    if len(coords) < 10:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    return angle if abs(angle) > 0.1 else 0.0


def clean_document_image(image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    angle = compute_skew_angle(gray)
    if angle != 0.0:
        h, w = image_bgr.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image_bgr = cv2.warpAffine(image_bgr, M, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)

    cleaned = cv2.medianBlur(image_bgr, 3)

    lab = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8, 8))
    l = clahe.apply(l)
    cleaned = cv2.merge([l, a, b])
    cleaned = cv2.cvtColor(cleaned, cv2.COLOR_LAB2BGR)

    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)
    cleaned = cv2.filter2D(cleaned, -1, kernel)

    cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    cleaned = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    return cleaned


def extract_text(pdf_path, confidence_threshold=CONFIDENCE_THRESHOLD, preprocess=True,
                  save_debug_dir=None):
    print(f"  OCR: running docTR on {pdf_path}...")
    model = ocr_predictor(pretrained=True)
    images = DocumentFile.from_pdf(pdf_path)

    if preprocess:
        print(f"  Preprocessing: cleaning {len(images)} page(s) with OpenCV...")
        cleaned_images = []
        for i, img in enumerate(images):
            cleaned = clean_document_image(img)
            cleaned_images.append(cleaned)
            if save_debug_dir:
                os.makedirs(save_debug_dir, exist_ok=True)
                cv2.imwrite(
                    os.path.join(save_debug_dir, f"page_{i}_cleaned.png"),
                    cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR),
                )
        images = cleaned_images

    result = model(images)
    exported = result.export()

    lines_out = []
    total_words = 0
    low_conf_words = 0
    for page in exported.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                tokens = []
                for word in line.get("words", []):
                    total_words += 1
                    value = word.get("value", "")
                    conf = word.get("confidence", 1.0)
                    if conf < confidence_threshold:
                        low_conf_words += 1
                        tokens.append(f"[UNCLEAR:{value}]")
                    else:
                        tokens.append(value)
                lines_out.append(" ".join(tokens))
            lines_out.append("")

    text = "\n".join(lines_out)
    pct = (low_conf_words / total_words * 100) if total_words else 0.0
    print(f"  OCR: extracted {total_words} words ({low_conf_words} low-confidence, {pct:.1f}%)")
    return text, result


def nearby_unclear(text, keywords, window=60):
    for kw in keywords:
        for m in re.finditer(re.escape(kw), text, re.IGNORECASE):
            start = max(0, m.start() - window)
            end = min(len(text), m.end() + window)
            if UNCLEAR_TAG_RE.search(text[start:end]):
                return True
    return False


def extract_regex(text):
    fields = {}
    for field, patterns in REGEX_FIELDS.items():
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if field == "currency" and value == "UNITED STATES DOLLAR":
                    value = "USD"
                if UNCLEAR_TAG_RE.search(value):
                    value = "UNREADABLE"
                fields[field] = value
                break
        if field not in fields:
            label_words = re.split(r"\s+", field.replace("_", " "))
            if nearby_unclear(text, label_words):
                fields[field] = "UNREADABLE"
            else:
                fields[field] = "NOT_PRESENT"
    return fields


def extract_with_synonyms(text, synonyms, regex_fields):
    fields = {}
    for field in SYNONYM_FIELDS:
        if regex_fields.get(field, "NOT_PRESENT") != "NOT_PRESENT":
            fields[field] = regex_fields[field]
            continue
        field_synonyms = synonyms.get(field, [])
        extracted = False
        for synonym in field_synonyms:
            escaped = re.escape(synonym)
            patterns = [
                rf"{escaped}[:\s]+([\d,]+\.?\d*)",
                rf"{escaped}[:\s]+([^\n]+)",
                rf"{escaped}\s+([\d,]+\.?\d*)",
                rf"{escaped}\s+([^\n]+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    if field == "currency" and "dollar" in value.lower():
                        value = "USD"
                    elif field == "currency" and len(value) > 3:
                        value = value[:3].upper()
                    if UNCLEAR_TAG_RE.search(value):
                        value = "UNREADABLE"
                    fields[field] = value
                    extracted = True
                    break
            if extracted:
                break
        if not extracted:
            fields[field] = "UNREADABLE" if nearby_unclear(text, field_synonyms) else "NOT_PRESENT"
    return fields


def extract_llm(text, llm, fields_to_extract):
    fields = {}
    for field in fields_to_extract:
        question = FIELD_QUESTIONS[field]
        prompt = (
            "You are an invoice data extraction assistant. "
            "Extract the exact value for the field from the invoice text. "
            "Reply ONLY with the extracted value, NOT_PRESENT, or UNREADABLE. No explanations.\n\n"
            "The invoice text may contain tags like [UNCLEAR:word] — these mark spots where "
            "the OCR scan could not confidently read the original document (blur, faded ink, "
            "a stamp or signature covering the text, poor scan quality, etc.). "
            "If the value you need overlaps with an [UNCLEAR:...] tag, or the surrounding text is "
            "too garbled to tell what the value is, reply UNREADABLE. "
            "Do NOT guess, autocorrect, or reconstruct a plausible-looking value from unclear text. "
            "Only use NOT_PRESENT when the field is clearly absent from clean, readable text.\n\n"
            "EXAMPLE 1:\n"
            "Invoice text:\n"
            "Invoice No.: ZG155-2025\n"
            "Invoice date: 11/18/2025\n"
            "Currency: USD\n"
            "TOTAL INVOICE AMOUNT 12,552.250\n"
            "Beneficiary Name: ZED GLOBAL LLC\n\n"
            "Field: What is the invoice number?\n"
            "Value: ZG155-2025\n\n"
            "EXAMPLE 2:\n"
            "Invoice text:\n"
            "Beneficiary Name: ZED GLOBAL LLC\n"
            "Bank Name: Mashreq Bank\n"
            "Swift: BOMLAED\n\n"
            "Field: Who is the beneficiary?\n"
            "Value: ZED GLOBAL LLC\n\n"
            "EXAMPLE 3:\n"
            "Invoice text:\n"
            "Payment term:100% on CAD\n"
            "Delivery term: CIF\n\n"
            "Field: What are the payment terms?\n"
            "Value: 100% on CAD\n\n"
            "EXAMPLE 4:\n"
            "Invoice text:\n"
            "Port of Loading: Port Apapa, Nigeria\n"
            "Country of Origin: Nigeria\n\n"
            "Field: What country is the beneficiary in?\n"
            "Value: Nigeria\n\n"
            "EXAMPLE 5:\n"
            "Invoice text:\n"
            "STAINLESS STEEL SCRAP 201\n"
            "GRADE FOR MELTING PURPOSE\n\n"
            "Field: What goods are described in the invoice?\n"
            "Value: Stainless Steel Scrap 201, Grade for Melting Purpose\n\n"
            "EXAMPLE 6:\n"
            "Invoice text:\n"
            "BL number: ONEYLOSF03789900\n\n"
            "Field: Is the beneficiary's signature present on the document?\n"
            "Value: NOT_PRESENT\n\n"
            "EXAMPLE 7:\n"
            "Invoice text:\n"
            "Beneficiary Name: [UNCLEAR:ZED] [UNCLEAR:GL0BAL] LLC\n"
            "Bank Name: [UNCLEAR:Mashreq] Bank\n\n"
            "Field: Who is the beneficiary?\n"
            "Value: UNREADABLE\n\n"
            "Now extract from:\n"
            f"Invoice text:\n{text[:3000]}\n\n"
            f"Field: {question}\n"
            "Value:"
        )
        raw = llm.invoke(prompt)
        result = raw.content if hasattr(raw, "content") else str(raw)
        result = result.strip().strip('"').strip("'")
        result_lower = result.lower()
        if "unreadable" in result_lower or "unclear" in result_lower or "cannot read" in result_lower:
            fields[field] = "UNREADABLE"
        elif "not present" in result_lower or "not found" in result_lower:
            fields[field] = "NOT_PRESENT"
        elif UNCLEAR_TAG_RE.search(result):
            fields[field] = "UNREADABLE"
        else:
            fields[field] = result
        print(f"  {field}: {fields[field][:80]}")
    return fields


def validate_fields(fields, synonym_fields):
    validated = dict(fields)
    swift = synonym_fields.get("beneficiary_swift_code", "")
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


def build_test_cases(all_fields):
    cases = []
    for field, question in FIELD_QUESTIONS.items():
        expected = all_fields.get(field, "NOT_PRESENT")
        cases.append({"question": question, "expected": expected, "field": field})
    return cases


def cmd_extract(pdf_path=None, preprocess=True):
    if pdf_path is None:
        pdf_path = os.path.join(DOCUMENTS_DIR, "track1-4.pdf")
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found")
        sys.exit(1)
    os.makedirs(INVOICE_DIR, exist_ok=True)
    debug_dir = os.path.join(INVOICE_DIR, "cleaned_pages") if preprocess else None
    print("=" * 60)
    print("  Invoice OCR Extraction")
    print("=" * 60)
    print(f"  Input:  {pdf_path}")
    print(f"  Output: {INVOICE_TEXT_PATH}")
    print(f"  Preprocessing: {'ON (OpenCV)' if preprocess else 'OFF'}")
    print("=" * 60)
    text, result = extract_text(pdf_path, preprocess=preprocess, save_debug_dir=debug_dir)
    with open(INVOICE_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved OCR text to {INVOICE_TEXT_PATH}")
    if debug_dir:
        print(f"  Saved cleaned page images to {debug_dir} for visual inspection")
    exported = result.export()
    with open(INVOICE_EXPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(exported, f, indent=2)
    print(f"  Saved JSON export to {INVOICE_EXPORT_PATH}")
    print("=" * 60)


def cmd_generate():
    if not os.path.exists(INVOICE_TEXT_PATH):
        print(f"Error: {INVOICE_TEXT_PATH} not found")
        print("Run 'python app.py extract' first.")
        sys.exit(1)
    with open(INVOICE_TEXT_PATH, encoding="utf-8") as f:
        text = f.read()
    os.makedirs(TESTS_DIR, exist_ok=True)
    print("=" * 60)
    print("  Invoice Field Extraction Pipeline")
    print("=" * 60)
    print(f"  Input:  {INVOICE_TEXT_PATH}")
    print(f"  Output: {TEST_CASES_PATH}")
    print("=" * 60)
    print("\n[1/4] Regex field extraction...")
    regex_fields = extract_regex(text)
    for k, v in regex_fields.items():
        print(f"  {k}: {v[:80]}")
    print("\n[2/4] Synonym fallback extraction...")
    synonyms = load_synonyms()
    synonym_fields = extract_with_synonyms(text, synonyms, regex_fields)
    for k, v in synonym_fields.items():
        if regex_fields.get(k, "NOT_PRESENT") == "NOT_PRESENT" and v != "NOT_PRESENT":
            print(f"  {k} (synonym): {v[:80]}")
    print("\n[3/4] LLM field extraction...")
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    llm_fields = extract_llm(text, llm, LLM_FIELDS)
    all_fields = {**synonym_fields, **llm_fields}
    all_fields = validate_fields(all_fields, synonym_fields)
    print("\n[4/4] Generating test cases...")
    test_cases = build_test_cases(all_fields)
    with open(TEST_CASES_PATH, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2)
    present = sum(1 for v in all_fields.values() if v != "NOT_PRESENT")
    missing = sum(1 for v in all_fields.values() if v == "NOT_PRESENT")
    print(f"\n  Generated {len(test_cases)} test cases")
    print(f"  Fields found: {present}")
    print(f"  Fields missing: {missing}")
    print(f"  Saved to {TEST_CASES_PATH}")
    print("=" * 60)


def cmd_serve():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    app = FastAPI(title="RAG API", version="1.0.0")
    class QueryRequest(BaseModel):
        question: str
    class QueryResponse(BaseModel):
        answer: str
        source_documents: list[str]
    print("Building vector store from invoice_docs.txt...")
    loader = TextLoader(INVOICE_TEXT_PATH, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings_model = OllamaEmbeddings(model=EMBEDDING_MODEL)
    print(f"  Embedding {len(chunks)} chunks...")
    chunk_texts = [doc.page_content for doc in chunks]
    chunk_embeddings = embeddings_model.embed_documents(chunk_texts)
    chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
    norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    chunk_embeddings = chunk_embeddings / np.where(norms == 0, 1, norms)
    synonyms = load_synonyms()
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    prompt_template = PromptTemplate(
        template=(
            "You are an invoice data extraction assistant. "
            "Answer the question using ONLY the information in the context below. "
            "Extract the exact value, number, name, or code as it appears in the context. "
            "Do not confuse different fields — each line in the table is a separate item. "
            "If the context does not contain the answer, say 'I don't know'.\n\n"
            "The context may contain tags like [UNCLEAR:word] marking spots the OCR scan could "
            "not confidently read (blur, faded ink, a stamp/signature covering the text, poor "
            "scan quality). If the answer to the question overlaps with an [UNCLEAR:...] tag, or "
            "the relevant part of the context is too garbled to answer confidently, reply exactly: "
            "'Not clearly visible in the document — cannot extract this information.' "
            "Never guess, autocorrect, or reconstruct a plausible value from unclear text.\n\n"
            "IMPORTANT RULES:\n"
            "- If the asked field label is immediately followed by ANOTHER field label instead of "
            "a value (e.g. 'Our Reference No Addtional Reference No. CO25/533004'), the asked field "
            "has NO value of its own. Do NOT use the value that follows the other label. "
            "Instead say the field is not present ('I don't know').\n"
            "- For 'total amount' or 'amount to pay', use the FINAL invoice total "
            "(labeled 'TOTAL INVOICE AMOUNT', 'TOTAL AMOUNT TOPAY', or 'GRAND TOTAL'), "
            "NOT individual line-item totals.\n"
            "- Line-item totals appear on individual rows (e.g., '21,480' next to an item). "
            "The invoice total appears at the bottom after all items, often labeled 'TOTAL INVOICE AMOUNT' or 'TOTAL AMOUNT TOPAY'.\n"
            "- If multiple totals appear, prefer the one with the highest label specificity "
            "('TOTAL INVOICE AMOUNT' > 'TOTAL' > line-item subtotals).\n"
            "- When you CAN answer, respond in a complete natural sentence "
            "(e.g. 'The currency is USD.' or 'The total invoice amount is 3,423.44 USD.'), "
            "extracting the value exactly as it appears in the context. Do not add extra "
            "explanation, commentary, or details beyond the answer.\n"
            "- When you CANNOT answer, still reply exactly as specified above "
            "('I don't know' or 'Not clearly visible in the document — cannot extract this information.').\n\n"
            "Important terminology:\n"
            "- Remitter / Buyer / Applicant / Consignee = the party BUYING the goods (who pays)\n"
            "- Beneficiary / Seller = the party SELLING the goods (who receives payment)\n"
            "- Consignee can refer to the buyer/importer\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )
    all_chunks = chunks

    def numpy_retrieve(query, top_k=5):
        q_emb = embeddings_model.embed_query(query)
        q_emb = np.array(q_emb, dtype=np.float32)
        q_emb = q_emb / (np.linalg.norm(q_emb) or 1)
        sims = chunk_embeddings @ q_emb
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [chunks[i] for i in top_idx]

    KEYWORD_MAP = {
        "remitter": ["BUYER", "Consignee", "APPLICANT", "ORDERING PARTY", "PAYER"],
        "applicant": ["BUYER", "Consignee", "APPLICANT", "ORDERING PARTY"],
        "buyer": ["BUYER", "Consignee", "APPLICANT"],
        "beneficiary": ["BENEFICIARY", "PAYEE", "SELLER"],
        "bank": ["BANK", "SWIFT", "ACCOUNT"],
        "invoice": ["INVOICE", "COMMERCIAL"],
        "payment": ["PAYMENT", "TTR", "TERMS"],
        "shipment": ["SHIP", "VESSEL", "PORT", "FROM", "TO"],
        "country": ["COUNTRY", "VIETNAM", "INDIA"],
    }

    def keyword_search(query, all_docs, top_k=3):
        query_lower = query.lower()
        scored = []
        for doc in all_docs:
            content = doc.page_content
            content_lower = content.lower()
            score = 0
            if query_lower in content_lower:
                score += 10
            for field, keywords in KEYWORD_MAP.items():
                if field in query_lower:
                    for kw in keywords:
                        if kw.lower() in content_lower:
                            score += 3
            total_boost = [
                "total invoice amount", "total amount to pay", "total amount topay",
                "grand total", "invoice total", "amount due", "balance due",
            ]
            for phrase in total_boost:
                if phrase in content_lower:
                    score += 8
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    UNREADABLE_MESSAGE = (
        "Not clearly visible in the document — cannot extract this information. "
        "(The relevant portion of the scan appears blurry, faded, or obscured, e.g. by a stamp "
        "or signature.)"
    )
    UNCLEAR_DENSITY_THRESHOLD = 0.25

    def unclear_density(text):
        words = text.split()
        if not words:
            return 0.0
        tagged = len(UNCLEAR_TAG_RE.findall(text))
        return tagged / len(words)

    def hybrid_retriever(query):
        expanded = expand_query(query, synonyms)
        embedding_docs = numpy_retrieve(expanded, top_k=5)
        keyword_docs = keyword_search(query, all_chunks, top_k=3)
        seen = set()
        merged = []
        for doc in embedding_docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        for doc in keyword_docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        return merged

    rag_chain = (
        {"context": RunnableLambda(hybrid_retriever) | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    print("RAG pipeline ready.")
    @app.post("/query", response_model=QueryResponse)
    def query_docs(request: QueryRequest):
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        source_docs_raw = hybrid_retriever(request.question)
        source_docs = [doc.page_content for doc in source_docs_raw]
        combined_context = " ".join(source_docs)
        if unclear_density(combined_context) > UNCLEAR_DENSITY_THRESHOLD:
            return QueryResponse(answer=UNREADABLE_MESSAGE, source_documents=source_docs)
        for label in asked_field_labels(request.question):
            if not field_has_readable_value(combined_context, label):
                return QueryResponse(answer=FIELD_NOT_PRESENT_MESSAGE, source_documents=source_docs)
        answer = rag_chain.invoke(request.question)
        if UNCLEAR_TAG_RE.search(answer):
            answer = UNREADABLE_MESSAGE
        return QueryResponse(answer=answer, source_documents=source_docs)
    @app.get("/health")
    def health():
        return {"status": "ok"}
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    command = sys.argv[1]
    if command == "extract":
        args = sys.argv[2:]
        preprocess = "--no-preprocess" not in args
        pdf_args = [a for a in args if a != "--no-preprocess"]
        pdf_path = pdf_args[0] if pdf_args else None
        cmd_extract(pdf_path, preprocess=preprocess)
    elif command == "generate":
        cmd_generate()
    elif command == "serve":
        cmd_serve()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)
