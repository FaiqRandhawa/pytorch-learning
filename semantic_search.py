# semantic_search.py
# Run with: python semantic_search.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

print("--> Loading embedding model...")
# Using a tiny, fast model specifically designed for sentence embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# ─── 1. OUR MINI DATABASE ────────────────────────────────────────
documents = [
    "A full-stack developer writes both frontend and backend code.",
    "The stock market saw a massive drop in tech shares today.",
    "Artificial intelligence relies heavily on matrix multiplication and calculus.",
    "Making the perfect espresso requires 9 bars of pressure.",
    "PyTorch tensors can be easily moved to GPUs for hardware acceleration."
]

# ─── 2. THE SEARCH QUERY ─────────────────────────────────────────
query = "How does deep learning do its math?"
print(f"\nSearch Query: '{query}'\n")

# Combine query and docs so we can process them all at once
texts = [query] + documents

# ─── 3. TEXT TO TENSORS (EMBEDDINGS) ─────────────────────────────
# Tokenize sentences and convert to PyTorch tensors
encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    model_output = model(**encoded)

# Mean Pooling: Average the token embeddings to get a single vector per sentence
attention_mask = encoded['attention_mask']
token_embeddings = model_output[0] 
input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

# Sum embeddings and divide by the number of active tokens
sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
sentence_embeddings = sum_embeddings / sum_mask

# Normalize the vectors so we can calculate cosine similarity easily
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# ─── 4. MATRIX MATH FOR SEARCH (COSINE SIMILARITY) ───────────────
# Separate the query vector from the document vectors
query_vector = sentence_embeddings[0]
doc_vectors = sentence_embeddings[1:]

# Perform a dot product between the query and all documents simultaneously
# This outputs a similarity score from -1 to 1 for each document
similarities = torch.matmul(doc_vectors, query_vector)

# ─── 5. DISPLAY RESULTS ──────────────────────────────────────────
# Sort the results by highest similarity score
sorted_indices = torch.argsort(similarities, descending=True)

print("--- Top Search Results ---")
for idx in sorted_indices:
    doc_index = idx.item()
    score = similarities[idx].item()
    print(f"[{score:.4f}] {documents[doc_index]}")