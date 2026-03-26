from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
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

# Create/load vector DB
if not os.path.exists("embeddings"):
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local("embeddings")
    print("Vector DB created!")
else:
    print("Loading existing embeddings...")

vectorstore = FAISS.load_local(
    "embeddings",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# LLM
qa_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-small"
)

print("\nAI Study Assistant Ready!")

# QA Loop
while True:
    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question: {query}

Answer briefly:
"""

    result = qa_pipeline(prompt, max_new_tokens=100)

    print("\nAnswer:")
    print(result[0]["generated_text"])