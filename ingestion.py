from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


PDF_PATH = "Ebook-Agentic-AI.pdf"


def load_and_chunk_pdf():
    reader = PdfReader(PDF_PATH)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append({
                "text": text,
                "page_number": page_number
            })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = []

    for document in documents:
        page_chunks = splitter.split_text(document["text"])

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page_number": document["page_number"]
            })

    return chunks


def create_vector_store():
    chunks = load_and_chunk_pdf()

    texts = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "source": PDF_PATH,
            "page_number": chunk["page_number"]
        }
        for chunk in chunks
    ]

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    vectorstore.save_local("faiss_index")

    return vectorstore


if __name__ == "__main__":
    create_vector_store()
    print("PDF chunks successfully stored in FAISS.") 
