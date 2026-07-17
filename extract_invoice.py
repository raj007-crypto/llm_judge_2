"""One-time script to extract invoice text from track1-4.pdf using docTR."""

import json

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)
doc = DocumentFile.from_pdf("track1-4.pdf")
result = model(doc)

# Structured text for RAG ingestion
text = result.render()
with open("invoice_docs.txt", "w", encoding="utf-8") as f:
    f.write(text)

# JSON export with word-level bounding boxes (for future provenance tracking)
exported = result.export()
with open("invoice_docs_export.json", "w", encoding="utf-8") as f:
    json.dump(exported, f, indent=2)

print(f"Extracted {len(text)} characters to invoice_docs.txt")
print("Exported JSON to invoice_docs_export.json")
