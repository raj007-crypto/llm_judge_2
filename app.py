"""Unified Invoice RAG System

Usage:
    python app.py extract [pdf_path] [--no-preprocess]  - Extract text from PDF using docTR
    python app.py generate              - Generate test cases from invoice text
    python app.py serve                 - Start RAG API server
    python app.py test                  - Run full evaluation suite

extract:
    By default, pages are cleaned with an Albumentations pipeline (deskew,
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
import albumentations as A

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

ADDITIONAL_QUESTIONS = [
    "Which date is the time delivery on?",
    "What is the bill number?",
    "What are the bank charges?",
    "Where is the shipment from?",
    "Where is the shipment going?",
    "What is the vessel name?",
    "What is the consignee address?",
    "What is the HS code?",
    "What is the container number?",
    "What are the shipping marks?",
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

CLAHE_CLIP_LIMIT = 2.0
SHARPEN_ALPHA = (0.2, 0.4)


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
        deskew = A.Compose([
            A.Rotate(limit=(angle, angle), border_mode=cv2.BORDER_REPLICATE, p=1.0),
        ])
        image_bgr = deskew(image=image_bgr)["image"]

    pipeline = A.Compose([
        A.MedianBlur(blur_limit=3, p=1.0),
        A.CLAHE(clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=(8, 8), p=1.0),
        A.Sharpen(alpha=SHARPEN_ALPHA, lightness=(0.9, 1.1), p=1.0),
        A.ToGray(p=1.0),
    ])
    cleaned = pipeline(image=image_bgr)["image"]
    if cleaned.ndim == 2:
        cleaned = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    else:
        cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    return cleaned


def extract_text(pdf_path, confidence_threshold=CONFIDENCE_THRESHOLD, preprocess=True,
                  save_debug_dir=None):
    print(f"  OCR: running docTR on {pdf_path}...")
    model = ocr_predictor(pretrained=True)
    images = DocumentFile.from_pdf(pdf_path)

    if preprocess:
        print(f"  Preprocessing: cleaning {len(images)} page(s) with Albumentations...")
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


def normalize_for_judge(s):
    s = s.lower().strip()
    s = re.sub(r"[$,\u20b9\u00a3\u20ac]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(".")
    return s


def judge_answer(question, answer, expected, context=""):
    JUDGE_PROMPT_TEMPLATE = """Compare these two values and decide if they match.

Expected: "{expected}"
Actual: "{actual}"

Rules:
- If the values are identical, answer: 5
- If one value is contained within the other, answer: 5
- If the values represent the same thing, answer: 5
- If the values are different numbers, answer: 1
- If the values are completely different, answer: 1

Answer only a number (1-5):"""

    JUDGE_PROMPT_NOT_PRESENT = """You are a strict judge. The expected answer is NOT_PRESENT, meaning this information is NOT in the document.

Actual: "{actual}"

Rules:
- If the actual output says it doesn't know, cannot find, or the information is not available -> answer: 5
- If the actual output provides ANY specific name, number, or value -> it is hallucinating -> answer: 1

Your answer (only a number):"""

    JUDGE_PROMPT_UNREADABLE = """You are a strict judge. The expected answer is UNREADABLE, meaning this information IS in the document but the scan/stamp/handwriting is too unclear for OCR to read reliably.

Actual: "{actual}"

Rules:
- If the actual output says the information is not clearly visible, unreadable, unclear, or cannot be extracted due to document/scan quality -> answer: 5
- If the actual output says it doesn't know / not present / not found (without mentioning it's an image-quality issue) -> answer: 3
- If the actual output provides a specific, confident-sounding name, number, or value -> it is hallucinating from garbled text -> answer: 1

Your answer (only a number):"""

    JUDGE_PROMPT_CONTEXT = """You are a strict judge evaluating whether an answer is correct based on the document context.

Question: "{question}"
Context: "{context}"
Answer: "{actual}"

Rules:
- Check if the answer is supported by the context
- Check if the answer is complete and relevant to the question
- If the answer is fully supported by context and answers the question -> answer: 5
- If the answer is partially supported or incomplete -> answer: 3
- If the answer is not supported by context or irrelevant -> answer: 1
- If the answer says "I don't know" but context has the answer -> answer: 1
- If the answer says "I don't know" and context doesn't have the answer -> answer: 5

Your answer (only a number):"""

    if expected.upper() == "NOT_PRESENT":
        prompt = JUDGE_PROMPT_NOT_PRESENT.format(actual=answer)
    elif expected.upper() == "UNREADABLE":
        prompt = JUDGE_PROMPT_UNREADABLE.format(actual=answer)
    elif expected:
        norm_expected = normalize_for_judge(expected)
        norm_answer = normalize_for_judge(answer)
        prompt = JUDGE_PROMPT_TEMPLATE.format(expected=norm_expected, actual=norm_answer)
    else:
        prompt = JUDGE_PROMPT_CONTEXT.format(question=question, context=context[:1000], actual=answer)
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": JUDGE_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 100},
        }, timeout=120)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        score = None
        m = re.search(r"\b([1-5])\b", raw)
        if m:
            score = int(m.group(1))
        return score, raw
    except Exception as e:
        return None, str(e)


def judge_pass(score, expected):
    if score is None:
        return False
    if expected.upper() in ("NOT_PRESENT", "UNREADABLE"):
        return score >= 4
    return score >= 3


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"[$,]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def check_answer_correctness(expected, answer):
    if not expected:
        return False
    if expected.upper() == "NOT_PRESENT":
        ans = answer.lower()
        return any(phrase in ans for phrase in [
            "don't know", "not present", "not mentioned",
            "not found", "not available", "no information",
            "cannot find", "no data", "does not mention", "not specified",
        ])
    if expected.upper() == "UNREADABLE":
        ans = answer.lower()
        return any(phrase in ans for phrase in [
            "not clearly visible", "unreadable", "unclear", "cannot extract",
            "cannot be reliably extracted", "poor scan", "blurry", "obscured",
        ])
    exp = normalize(expected)
    ans = normalize(answer)
    if exp in ans:
        return True
    exp_stripped = re.sub(r"^(messrs\.?:?\s*|m/s\.?\s*|the\s+|mr\.?\s*|mrs\.?\s*|ms\.?\s*|dr\.?\s*)", "", exp)
    if exp_stripped and exp_stripped in ans:
        return True
    ans_stripped = re.sub(r"^(messrs\.?:?\s*|m/s\.?\s*|the\s+|mr\.?\s*|mrs\.?\s*|ms\.?\s*|dr\.?\s*)", "", ans)
    if ans_stripped and ans_stripped in exp:
        return True
    if ans_stripped and ans_stripped in exp_stripped:
        return True
    try:
        return abs(float(exp) - float(ans)) < 0.01
    except ValueError:
        return False


def check_answer_in_context(answer, contexts):
    keys = []
    val = re.sub(r"(?i)^the\s+.*?(?:is|was|are|were)[:\s]+", "", answer).strip()
    val = val.rstrip(".")
    if val:
        keys = [normalize(val)]
    else:
        tokens = normalize(answer).split()
        keys = [t for t in tokens if len(t) > 2]
    combined_ctx = " ".join(normalize(c) for c in contexts)
    for k in keys:
        if k in combined_ctx:
            return True
    return False


def ask_backend(question):
    resp = requests.post(BACKEND_URL, json={"question": question}, timeout=60)
    resp.raise_for_status()
    return resp.json()


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
    print(f"  Preprocessing: {'ON (Albumentations)' if preprocess else 'OFF'}")
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
            "- For 'total amount' or 'amount to pay', use the FINAL invoice total "
            "(labeled 'TOTAL INVOICE AMOUNT', 'TOTAL AMOUNT TOPAY', or 'GRAND TOTAL'), "
            "NOT individual line-item totals.\n"
            "- Line-item totals appear on individual rows (e.g., '21,480' next to an item). "
            "The invoice total appears at the bottom after all items, often labeled 'TOTAL INVOICE AMOUNT' or 'TOTAL AMOUNT TOPAY'.\n"
            "- If multiple totals appear, prefer the one with the highest label specificity "
            "('TOTAL INVOICE AMOUNT' > 'TOTAL' > line-item subtotals).\n\n"
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
        answer = rag_chain.invoke(request.question)
        if UNCLEAR_TAG_RE.search(answer):
            answer = UNREADABLE_MESSAGE
        return QueryResponse(answer=answer, source_documents=source_docs)
    @app.get("/health")
    def health():
        return {"status": "ok"}
    uvicorn.run(app, host="0.0.0.0", port=8000)


def cmd_test():
    try:
        with open(TEST_CASES_PATH) as f:
            test_cases = json.load(f)
            if not isinstance(test_cases, list):
                test_cases = []
    except (FileNotFoundError, json.JSONDecodeError):
        test_cases = []
    for q in ADDITIONAL_QUESTIONS:
        test_cases.append({"question": q, "expected": "", "field": "additional"})
    print("=" * 70)
    print("  Invoice RAG + LLM Judge — Full Evaluation Suite")
    print("=" * 70)
    print(f"  Backend  : {BACKEND_URL}")
    print(f"  Judge    : {JUDGE_MODEL}")
    print(f"  Tests    : {len(test_cases)} invoice Q&A pairs")
    print("  Pass/Fail based on: LLM judge evaluation (score >= 3 = PASS)")
    print("=" * 70)
    results = []
    start = time.time()
    for i, tc in enumerate(test_cases, 1):
        question = tc.get("question", "")
        expected = tc.get("expected", "")
        if not question:
            continue
        print(f"\n[{i}/{len(test_cases)}] {question}")
        try:
            result = ask_backend(question)
            answer = result.get("answer", "")
            contexts = result.get("source_documents", [])
        except Exception as e:
            print(f"  Backend error: {e}")
            results.append({"pass": False, "question": question, "answer": "ERROR"})
            continue
        correct_answer = check_answer_correctness(expected, answer) if expected else False
        in_context = check_answer_in_context(answer, contexts)
        combined_context = "\n".join(contexts) if contexts else ""
        if expected:
            score, reason = judge_answer(question, answer, expected)
        else:
            score, reason = judge_answer(question, answer, "", combined_context)
        overall_pass = judge_pass(score, expected if expected else "CONTEXT_BASED")
        status = "PASS" if overall_pass else "FAIL"
        print(f"  Answer   : {answer}")
        if expected:
            print(f"  Expected : {expected}")
        score_str = f"{score}/5" if score is not None else "N/A"
        print(f"  Judge    : {score_str}")
        if expected:
            print(f"  Correct  : {'YES' if correct_answer else 'NO'}")
        print(f"  Verdict  : {status}")
        results.append({
            "pass": overall_pass, "correct_answer": correct_answer,
            "in_context": in_context, "question": question,
            "answer": answer, "expected": expected, "score": score, "reason": reason,
        })
        time.sleep(0.5)
    elapsed = time.time() - start
    passed = sum(1 for r in results if r["pass"])
    failed = sum(1 for r in results if not r["pass"])
    correct_answers = sum(1 for r in results if r.get("correct_answer"))
    print("\n")
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total tests        : {len(results)}")
    print(f"  Correct answers    : {correct_answers}/{len(results)}")
    print(f"  Overall PASS       : {passed}/{len(results)}")
    print(f"  Overall FAIL       : {failed}/{len(results)}")
    print(f"  Time elapsed       : {elapsed:.1f}s")
    print("=" * 70)
    if failed > 0:
        print("\n  FAILED CASES:")
        print("-" * 70)
        for r in results:
            if not r["pass"]:
                print(f"  Q: {r['question']}")
                print(f"     A: {r.get('answer', 'N/A')}")
                if r.get("expected"):
                    print(f"     Expected: {r.get('expected', 'N/A')}")
                print(f"     Judge: {r.get('score', '?')}/5")
                reason_short = (r.get("reason", "N/A") or "")[:150]
                print(f"     Reason: {reason_short}...")
                print()
    print("=" * 70)
    verdict = "ALL PASS" if failed == 0 else f"{failed} FAILED"
    print(f"  RESULT: {verdict}")
    print("=" * 70)


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
    elif command == "test":
        cmd_test()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)
