from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

DATA_PATH = "data"

documents = []

# Load PDFs
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())

if len(documents) == 0:
    print("No documents found in data folder!")
    exit()

print("Number of pages loaded:", len(documents))

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks created:", len(chunks))

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector DB if not exists
if not os.path.exists("embeddings/index.faiss"):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local("embeddings")
    print("Vector DB created!")
else:
    print("Embeddings already exist.")