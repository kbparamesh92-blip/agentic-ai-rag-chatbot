from pypdf import PdfReader
from pathlib import Path


def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append({
                "text": text,
                "page_number": page_number,
                "source": Path(pdf_path).name
            })

    return documents


if __name__ == "__main__":
    pdf_path = "Ebook-Agentic-AI.pdf"

    documents = load_pdf(pdf_path)

    print(f"Loaded {len(documents)} pages")
