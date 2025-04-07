from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader

# Load PDF and extract text
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() + "\n"
    return raw_text

# Chunk text
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

# Generate FAISS index
def create_faiss_index(texts, index_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local(index_path)
    print(f"FAISS index saved to: {index_path}")

if __name__ == "__main__":
    pdf_path = "SEBI_-Securities_Market_Booklet.pdf"
    index_path = "sebi_faiss_index"

    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text)
    create_faiss_index(chunks, index_path)
