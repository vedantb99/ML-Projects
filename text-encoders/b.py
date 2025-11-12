import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
import warnings
import torch

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("--- SOTA Model Experiment ---")

# 1. Load the SOTA Qwen3-Embedding-8B model
model_name = "Qwen/Qwen3-Embedding-8B"
print(f"Loading SOTA model: {model_name}...")
print("This will take time, especially on first run...")

sota_model = SentenceTransformer(
    model_name, 
    trust_remote_code=True,
    model_kwargs={'load_in_4bit': True}  # Load in 4-bit for 24GB VRAM
)
sota_model.max_seq_length = 512  # Truncate to prevent OOM on long docs

print("SOTA model loaded.")

def run_clustering_experiment():
    print("\nLoading TwentyNewsgroups dataset...")
    newsgroups_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    sentences = newsgroups_data.data
    true_labels = newsgroups_data.target
    num_clusters = len(np.unique(true_labels))

    print(f"Dataset loaded. {len(sentences)} documents in {num_clusters} categories.")

    print("\nEncoding documents with SOTA model...")
    print("This is the most time-consuming step. Please be patient.")
    
    sentence_embeddings = sota_model.encode(
        sentences, 
        show_progress_bar=True,
        batch_size=32 
    )
    print("Encoding complete. Running KMeans clustering...")

    # 3. Run KMeans clustering
    kmeans = KMeans(
        n_clusters=num_clusters,
        n_init=10,
        random_state=42
    )
    predicted_labels = kmeans.fit_predict(sentence_embeddings)

    print("Clustering complete. Calculating score...")
    
    score = v_measure_score(true_labels, predicted_labels)

    print("\n--- Experiment Complete ---")
    
    # Our baseline scores
    w2v_score = 18.21
    llm2vec_score = 30.26
    
    # The new SOTA score from our experiment
    sota_exp_score = score * 100
    print(f"  Average Word2Vec Score (V-Measure): {w2v_score:.2f}")
    print(f"  LLM2Vec-Mistral-7B Score (V-Measure): {llm2vec_score:.2f}")
    print(f"  Qwen3-Embedding-8B Score (V-Measure): {sota_exp_score:.2f} (Our Result)")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("\nWARNING: No GPU detected. This will be extremely slow.")
        print("Please run this on a machine with a CUDA-enabled GPU.\n")
    
    run_clustering_experiment()