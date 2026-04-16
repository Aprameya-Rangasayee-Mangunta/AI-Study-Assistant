from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
import os

# Check if embeddings exist
if not os.path.exists("embeddings/index.faiss"):
    print("Embeddings not found! Run preprocess.py first.")
    exit()

# Load vector DB
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "embeddings",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# LLM
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large"
)

print("\nAI Study Assistant Ready!")

# QA Loop
while True:
    try:
        query = input("\nAsk a question (or type 'exit'): ")
    except EOFError:
        print("\nNo input received. Exiting.")
        break
    if query.lower() == "exit":
        print("Thank You For using your personalised AI Study Assistant")
        break
    # Save queries
    try:
        with open("inputs.txt", "r") as f:
            existing_queries = f.read().splitlines()
    except FileNotFoundError:
        existing_queries = []

    if query not in existing_queries:
        with open("inputs.txt", "a") as f:
            f.write(query + "\n")

    # Retrieve docs
    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    # Limit context size (IMPORTANT)
    context = context[:1500]

    # Optional: cleaner output
    print("\n[Context retrieved successfully]")

    # Prompt
    prompt = f"""
You are a helpful study assistant.

Provide a clear and concise explanation of the question. Use the provided context if relevant, otherwise answer based on general knowledge.

Context:
{context}

Question: {query}

Answer:
"""
    # Generate answer
    result = qa_pipeline(
        prompt,
        max_new_tokens=150,
        do_sample=False
    )
    print("\nAnswer:")
    print(result[0]["generated_text"])