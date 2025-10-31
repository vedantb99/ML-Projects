import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================================================================
# Part 0: Setup
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =============================================================================
# Part 1: Retrieval System (NumPy-based)
# =============================================================================

# 1. Our knowledge base
documents = [
    "The planet Mars has two moons, Phobos and Deimos.",
    "Jupiter is the largest planet in our solar system.",
    "The Earth revolves around the Sun in approximately 365.25 days.",
    "Saturn is known for its prominent ring system.",
    "Venus is the second planet from the Sun and is the hottest.",
    "The Sun is a star at the center of the Solar System."
]

# 2. Load the encoder model
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

# 3. Create and normalize vector embeddings for all documents
doc_embeddings = encoder.encode(documents, convert_to_numpy=True)
norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
normalized_doc_embeddings = doc_embeddings / norms

def retrieve_with_numpy(query: str, doc_embeddings: np.ndarray, k: int = 2):
    query_embedding = encoder.encode([query], convert_to_numpy=True)
    query_norm = np.linalg.norm(query_embedding)
    normalized_query_embedding = query_embedding / query_norm
    similarity_scores = np.dot(normalized_query_embedding, doc_embeddings.T)
    top_k_indices = np.argsort(similarity_scores[0])[::-1][:k]
    return [(documents[i], similarity_scores[0][i]) for i in top_k_indices]

# =============================================================================
# Part 2: Generation System (LLM-based)
# =============================================================================

# 1. Load the tokenizer and generator model
MODEL_NAME = "LiquidAI/LFM2-700M"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
generator_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
generator_model.to(DEVICE)
generator_model.eval()

def generate_real_answer(query: str, context_docs: list, model, tokenizer):
    context = "\n".join([doc for doc, score in context_docs])
    prompt = (
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{query}\n\n"
        "ANSWER:\n"
    )
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    answer = decoded_text[len(prompt):].strip()
    return answer, prompt

# =============================================================================
# Part 3: Running the Full RAG Pipeline
# =============================================================================

user_query = "which planet is the biggest?"

# Step 1: RETRIEVAL
retrieved_docs = retrieve_with_numpy(user_query, normalized_doc_embeddings, k=2)

# Step 2: AUGMENTED GENERATION
final_answer, full_prompt = generate_real_answer(user_query, retrieved_docs, generator_model, tokenizer)

# --- Display the results ---
print("\n--- Retrieved Documents (Top 2) ---")
for doc, score in retrieved_docs:
    print(f"Score: {score:.4f} | Document: {doc}")

print("\n\n--- Prompt Sent to LLM ---")
print(full_prompt)

print("\n--- Final Answer from LLM ---")
print(final_answer)