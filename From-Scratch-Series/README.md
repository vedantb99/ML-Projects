# Simple RAG with NumPy & Hugging Face

This project is a clear, from-scratch implementation of a **Retrieval-Augmented Generation (RAG)** pipeline. It demonstrates the core mechanics of RAG using **NumPy** for the vector similarity search and a small, efficient language model from the **Hugging Face Hub** (`LiquidAI/LFM2-700M`) for text generation.

The goal is to provide an easy-to-understand example of how RAG works without relying on complex, specialized vector database libraries like FAISS or Pinecone.

-----

## \#\# What is RAG? 🤔

Retrieval-Augmented Generation is a technique that enhances the capabilities of Large Language Models (LLMs) by connecting them to an external knowledge base. Instead of relying solely on the information it was trained on, the model can "look up" relevant facts before answering a question.

The process involves two main stages:

1.  **Retrieval**: Given a user query, the system searches through a collection of documents (the knowledge base) to find the most relevant snippets of information. This is done using vector similarity.
2.  **Augmented Generation**: The retrieved information is combined with the original query and fed as a detailed prompt to the LLM. The model then generates an answer based on the provided context, making its response more accurate and factually grounded.

[Image of a Retrieval-Augmented Generation diagram]

-----

## \#\# How It Works: Technical Breakdown 🛠️

This implementation is broken down into two primary Python functions: `retrieve_with_numpy` and `generate_real_answer`.

### **Part 1: The Retrieval System (NumPy)**

This part is responsible for finding the most relevant documents.

1.  **Vector Encoding**: All documents in our knowledge base are first converted into numerical vectors (embeddings) using the `sentence-transformers/all-MiniLM-L6-v2` model.
2.  **Normalization**: The document vectors are normalized to unit length (L2 norm). This clever trick allows us to use the **dot product** as a direct measure of **cosine similarity**. A higher dot product value means the vectors are more aligned and thus more semantically similar.
3.  **Similarity Search**:
      * The user's query is also encoded and normalized.
      * We then compute the dot product between the query vector and the entire matrix of document vectors. This single operation efficiently calculates the similarity score for every document.
      * `np.argsort` is used to find the indices of the documents with the highest scores, giving us our top results.

### **Part 2: The Generation System (Hugging Face)**

This part uses the retrieved context to formulate an answer.

1.  **Model Loading**: The `LiquidAI/LFM2-700M` model and its tokenizer are loaded from the Hugging Face Hub. This is a relatively small model, making it easy to run on consumer hardware.

2.  **Prompt Engineering**: A clear, structured prompt is created. It combines the retrieved documents (as context) with the original user query. This is the crucial "augmented" step.

    ```
    CONTEXT:
    [Retrieved document 1]
    [Retrieved document 2]

    QUESTION:
    [Original user query]

    ANSWER:
    ```

3.  **Text Generation**: The complete prompt is tokenized and fed to the `generator_model`. The model's `generate()` function produces the final text answer based on the context it was given.

-----

## \#\# Dependencies & Installation

You'll need a few Python libraries to run this project. You can install them all using pip:

```bash
pip install torch numpy sentence-transformers transformers
```

> **Note**: If you have a CUDA-enabled GPU, make sure you have the correct version of PyTorch installed to take advantage of it\!

-----

## \#\# How to Run the Code

1.  Save the complete code as a Python file (e.g., `run_rag.py`).

2.  Run the script from your terminal:

    ```bash
    python run_rag.py
    ```

You should see output that looks like this, showing the retrieved documents, the prompt sent to the model, and the final generated answer.

```
Using device: cuda

--- Retrieved Documents (Top 2) ---
Score: 0.7451 | Document: Jupiter is the largest planet in our solar system.
Score: 0.5029 | Document: Venus is the second planet from the Sun and is the hottest.

--- Prompt Sent to LLM ---
CONTEXT:
Jupiter is the largest planet in our solar system.
Venus is the second planet from the Sun and is the hottest.

QUESTION:
which planet is the biggest?

ANSWER:

--- Final Answer from LLM ---
Jupiter is the largest planet in our solar system.
```

-----

## \#\# Customization

This project is a great starting point. Here are a few ways you can easily customize it:

  * **Expand the Knowledge Base**: Simply add more strings to the `documents` list in the script.
  * **Change the Models**: Swap out the `SentenceTransformer` or the `AutoModelForCausalLM` with any other model from the Hugging Face Hub.
  * **Adjust `k`**: Change the number of retrieved documents by modifying the `k` parameter in the `retrieve_with_numpy` function call.