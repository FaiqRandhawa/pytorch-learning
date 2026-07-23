import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

docs = [
    "PyTorch is a deep learning framework used for building neural networks.",
    "RAG stands for Retrieval Augmented Generation. It combines search with LLMs.",
    "Transformers use self-attention to process sequences in parallel unlike RNNs.",
    "Fine tuning is the process of taking a pretrained model and training it on custom data.",
    "ChromaDB is a vector database that stores embeddings for semantic search.",
    "FastAPI is a modern Python web framework for building APIs quickly.",
    "Backpropagation computes gradients by applying chain rule through computation graph.",
    "CNNs use filters that slide across images to detect spatial patterns like edges and curves.",
    "HuggingFace provides thousands of pretrained models for NLP, vision and audio tasks.",
]

# Create embeddings for the documents using SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(docs)


# ── 3. store in vector db ─────────────────────────────────────
client     = chromadb.Client()
collection = client.create_collection("knowledge")


collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(docs))]
)


##now check querying the vector database with a sample query

question = "What is backpropagation?"
q_embedding = embedder.encode([question]).tolist()

results = collection.query(
    query_embeddings=q_embedding,
    n_results=2
)

print("question:", question)
print("\nrelevant docs found:")
for doc in results['documents'][0]:
    print("-", doc)

##answer with LLM, what this does is to use the retrieved documents as context for the LLM to answer the question. We will use a text2text generation model (like T5) to generate an answer based on the context provided by the retrieved documents.


context = "\n".join(results['documents'][0])
generator = pipeline("text2text-generation", model="google/flan-t5-small")

prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
answer = generator(prompt, max_length=100)[0]['generated_text']
print("\nLLM answer:", answer)