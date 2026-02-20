from core.embedder import Embedder
from core.vector_store import VectorStore
from core.retriever import Retriever

COLLECTION = "research_papers"

# Initialize components
# Note: We don't re-embed today — ChromaDB already has the data from Day 4
print("=== Initializing Components ===")
embedder = Embedder()
store = VectorStore()
retriever = Retriever(embedder, store, collection_name=COLLECTION, n_results=5)

# Confirm data exists in ChromaDB
count = store.get_collection_count(COLLECTION)
print(f"Chunks in ChromaDB: {count}")

# ── TEST 1: Basic retrieval ──
print("\n=== TEST 1: Basic Retrieve ===")
query1 = "How does the attention mechanism work?"
results = retriever.retrieve(query1)
print(f"Query: '{query1}'")
print(f"Retrieved {len(results)} chunks\n")
for i, r in enumerate(results):
    print(f"Result {i+1} | Similarity: {r['similarity']}")
    print(f"Preview: {r['text'][:200]}")
    print()

# ── TEST 2: Context string for LLM ──
print("\n=== TEST 2: Context String (for LLM) ===")
query2 = "What is the encoder decoder structure?"
context = retriever.retrieve_as_context(query2)
print(f"Query: '{query2}'")
print(f"\nFormatted context:\n")
print(context[:800])  # print first 800 chars

# ── TEST 3: Relevance check ──
print("\n=== TEST 3: Relevance Check ===")

queries = [
    "What optimizer was used for training?",
    "How many attention heads are used?",
    "What is the recipe for chocolate cake?",  # irrelevant query
]

for q in queries:
    chunks, score = retriever.retrieve_with_scores(q)
    relevant = retriever.is_relevant(q)
    print(f"Query: '{q}'")
    print(f"Avg similarity: {score} | Relevant: {relevant}\n")