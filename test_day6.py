from core.embedder import Embedder
from core.vector_store import VectorStore
from core.retriever import Retriever
from core.translator import Translator

COLLECTION = "research_papers"

# Initialize the full pipeline
print("=== Initializing Pipeline ===")
embedder = Embedder()
store = VectorStore()
retriever = Retriever(embedder, store, collection_name=COLLECTION, n_results=5)
translator = Translator(retriever, model_name="qwen2.5:7b")

# ── TEST 1: Answer a question ──
print("\n=== TEST 1: Question Answering ===")
question = "What is the main contribution of this paper?"
print(f"Question: {question}\n")
result = translator.answer_question(question)
print(f"Answer:\n{result['answer']}")
print(f"\nAvg Relevance: {result['avg_relevance']}")

# ── TEST 2: Translate a section ──
print("\n=== TEST 2: Translation ===")
text_to_translate = """The Transformer model architecture relies entirely 
on attention mechanisms, dispensing with recurrence and convolutions entirely. 
This allows for significantly more parallelization during training."""

print(f"Original text:\n{text_to_translate}\n")
result2 = translator.translate(text_to_translate, target_language="Hindi")
print(f"Hindi Translation:\n{result2['translation']}")

# ── TEST 3: Simplify ──
print("\n=== TEST 3: Simplify ===")
complex_text = """Multi-head attention allows the model to jointly attend 
to information from different representation subspaces at different positions."""

print(f"Complex text:\n{complex_text}\n")
result3 = translator.simplify(complex_text)
print(f"Simplified:\n{result3['simplified']}")

print("\n=== DAY 6 COMPLETE ===")