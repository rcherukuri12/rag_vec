import math
from typing import List, Tuple

import numpy as np

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from turbovec import TurboQuantIndex


# -----------------------------
# Local model setup
# -----------------------------
Settings.llm = Ollama(model="gemma4:26b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")


# -----------------------------
# Helpers
# -----------------------------
def chunk_documents(
    input_files: List[str],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
):
    documents = SimpleDirectoryReader(input_files=input_files).load_data()
    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = parser.get_nodes_from_documents(documents)
    return nodes


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    embed_model = Settings.embed_model
    vectors: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_vecs = embed_model.get_text_embedding_batch(batch)
        vectors.extend(batch_vecs)

    arr = np.asarray(vectors, dtype=np.float32)
    arr = np.ascontiguousarray(arr)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape {arr.shape}")

    return arr


def build_prompt(question: str, contexts: List[str]) -> str:
    joined_context = "\n\n".join(
        f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )

    return f"""You are answering questions using only the provided context.

Context:
{joined_context}

Question:
{question}

Instructions:
- Answer using only the context above.
- If the answer is not in the context, say you do not have enough information.
- Be concise but specific.

Answer:
"""


def retrieve_top_k(
    index: TurboQuantIndex,
    query_vector: List[float],
    chunks: List[str],
    top_k: int = 4,
) -> List[Tuple[int, float, str]]:
    # Convert single query vector to a batch of 1 query: shape (1, dim)
    query_np = np.asarray(query_vector, dtype=np.float32)
    query_np = np.ascontiguousarray(query_np)

    if query_np.ndim != 1:
        raise ValueError(f"Expected 1D query vector before batching, got shape {query_np.shape}")

    query_batch = query_np.reshape(1, -1)
    query_batch = np.ascontiguousarray(query_batch)

    result = index.search(query_batch, k=top_k)

    if not (isinstance(result, tuple) and len(result) == 2):
        raise ValueError(
            f"Unexpected return from TurboQuantIndex.search(): {type(result)} / {result}"
        )

    scores, ids = result

    scores = np.asarray(scores)
    ids = np.asarray(ids)

    # Handle either batched shape (1, k) or flat shape (k,)
    if scores.ndim == 2:
        scores = scores[0]
    if ids.ndim == 2:
        ids = ids[0]

    scores = [float(x) for x in scores]
    ids = [int(x) for x in ids]

    out: List[Tuple[int, float, str]] = []
    for idx, score in zip(ids, scores):
        if 0 <= idx < len(chunks):
            out.append((idx, score, chunks[idx]))

    return out


# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading and chunking documents...")
    nodes = chunk_documents(["data.txt"], chunk_size=512, chunk_overlap=64)

    chunks = [node.get_content() for node in nodes]
    if not chunks:
        raise ValueError("No chunks were created from data.txt")

    print(f"Chunks created: {len(chunks)}")

    print("Embedding chunks with Ollama...")
    vectors_np = embed_texts(chunks)

    dim = vectors_np.shape[1]
    bit_width = 4

    print(f"Creating TurboQuant index (dim={dim}, bit_width={bit_width})...")
    tq_index = TurboQuantIndex(dim=dim, bit_width=bit_width)

    print("Adding vectors to TurboQuant...")
    tq_index.add(vectors_np)

    # Compression statistics
    num_vectors = vectors_np.shape[0]
    bytes_per_float32 = 4
    original_bytes = num_vectors * dim * bytes_per_float32
    compressed_bytes = num_vectors * dim * bit_width // 8
    compression_ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else math.inf

    print("\n--- TurboQuant Compression Statistics ---")
    print(f"Vectors indexed      : {num_vectors}")
    print(f"Dimensions           : {dim}")
    print(f"Bit width            : {bit_width}-bit")
    print(f"Original size        : {original_bytes:,} bytes ({original_bytes / 1024:.1f} KB) at float32")
    print(f"Compressed size      : {compressed_bytes:,} bytes ({compressed_bytes / 1024:.1f} KB) at {bit_width}-bit")
    print(f"Compression ratio    : {compression_ratio:.1f}x smaller")
    print("-----------------------------------------\n")

    questions = [
        "What GPU infrastructure does Raja use?",
        "What company does Raja run?",
    ]

    llm = Settings.llm

    print("--- Local RAG Pipeline: TurboQuant + Gemma4 + Ollama ---\n")

    for q in questions:
        print(f"Q: {q}")

        query_vector = Settings.embed_model.get_query_embedding(q)
        retrieved = retrieve_top_k(tq_index, query_vector, chunks, top_k=4)

        if not retrieved:
            print("A: I could not retrieve any relevant context.\n")
            continue

        contexts = [text for _, _, text in retrieved]
        prompt = build_prompt(q, contexts)
        response = llm.complete(prompt)

        print("Retrieved chunks:")
        for idx, score, text in retrieved:
            preview = text[:180].replace("\n", " ")
            print(f"  - chunk_id={idx}, score={score:.4f}, preview={preview}...")

        print(f"\nA: {response}\n")


if __name__ == "__main__":
    main()
    