import arxiv
import httpx
import tempfile
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ArxivClient:
    """
    Handles everything related to arXiv:
    1. Searching papers by keyword
    2. Downloading PDFs directly from arXiv
    3. Recommending similar papers based on uploaded paper
    """

    def __init__(self, embedder=None):
        """
        embedder: optional Embedder instance for recommendations
        If not provided, recommendations won't work but search will
        """
        self.client = arxiv.Client()
        self.embedder = embedder

    def search(self, query: str, max_results: int = 10) -> list[dict]:
        """
        Searches arXiv for papers matching the query.
        Returns list of paper dicts with metadata.
        """
        print(f"Searching arXiv for: '{query}'")

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        for paper in self.client.results(search):
            results.append({
                "title": paper.title,
                "authors": ", ".join(a.name for a in paper.authors[:3]) +
                           (" et al." if len(paper.authors) > 3 else ""),
                "abstract": paper.summary,
                "published": paper.published.strftime("%Y-%m-%d"),
                "arxiv_id": paper.entry_id.split("/")[-1],
                "pdf_url": paper.pdf_url,
                "categories": ", ".join(paper.categories[:2]),
                "paper_obj": paper  # keep original object for downloading
            })

        print(f"Found {len(results)} papers")
        return results

    def download_pdf(self, paper_dict: dict, save_dir: str = "./data/uploads") -> str:
        """
        Downloads a paper PDF from arXiv.
        Returns the local file path where it was saved.
        """
        os.makedirs(save_dir, exist_ok=True)

        arxiv_id = paper_dict["arxiv_id"]
        # Clean ID for filename (remove version like v1, v2)
        clean_id = arxiv_id.replace("/", "_")
        save_path = os.path.join(save_dir, f"{clean_id}.pdf")

        # Skip download if already exists
        if os.path.exists(save_path):
            print(f"PDF already exists: {save_path}")
            return save_path

        print(f"Downloading PDF: {paper_dict['title'][:50]}...")

        # Use the paper object to download
        paper_obj = paper_dict["paper_obj"]
        paper_obj.download_pdf(dirpath=save_dir, filename=f"{clean_id}.pdf")

        print(f"Downloaded to: {save_path}")
        return save_path

    def get_recommendations(
        self,
        uploaded_paper_text: str,
        query_keywords: str,
        top_n: int = 5,
        candidate_pool: int = 20
    ) -> list[dict]:
        """
        Recommends papers similar to the uploaded paper.

        How it works:
        1. Extract keywords from uploaded paper
        2. Search arXiv for candidate papers using those keywords
        3. Embed the uploaded paper + all candidate abstracts
        4. Rank candidates by cosine similarity
        5. Return top N most similar

        uploaded_paper_text: clean text from the uploaded PDF
        query_keywords: keywords to search arXiv (auto-generated or user provided)
        top_n: how many recommendations to return
        candidate_pool: how many arXiv papers to fetch as candidates
        """
        if self.embedder is None:
            raise RuntimeError("Embedder required for recommendations")

        print(f"Finding recommendations for uploaded paper...")

        # Step 1: Search arXiv for candidate papers
        candidates = self.search(query_keywords, max_results=candidate_pool)

        if not candidates:
            return []

        # Step 2: Embed the uploaded paper
        # Use first 2000 chars (intro/abstract) — most representative
        paper_snippet = uploaded_paper_text[:2000]
        paper_embedding = np.array(
            self.embedder.embed_single(paper_snippet)
        ).reshape(1, -1)

        # Step 3: Embed all candidate abstracts
        print(f"Embedding {len(candidates)} candidate abstracts...")
        abstracts = [c["abstract"] for c in candidates]
        candidate_embeddings = np.array(
            self.embedder.embed_texts(abstracts)
        )

        # Step 4: Compute cosine similarity between uploaded paper
        # and each candidate
        similarities = cosine_similarity(paper_embedding, candidate_embeddings)[0]

        # Step 5: Rank by similarity
        ranked_indices = np.argsort(similarities)[::-1]  # descending order

        recommendations = []
        for idx in ranked_indices[:top_n]:
            candidate = candidates[idx].copy()
            candidate["similarity_score"] = round(float(similarities[idx]), 4)
            candidate["similarity_pct"] = round(float(similarities[idx]) * 100, 1)
            recommendations.append(candidate)

        print(f"Top recommendation: {recommendations[0]['title'][:50]} ({recommendations[0]['similarity_pct']}%)")
        return recommendations

    def extract_keywords(self, text: str) -> str:
        """
        Extracts simple keywords from paper text for arXiv search.
        Takes first 500 chars (title + abstract area) and cleans it up.
        """
        # Take just the beginning — title and abstract
        snippet = text[:500]

        # Remove page markers we added
        snippet = snippet.replace("--- PAGE 1 ---", "")
        snippet = snippet.replace("--- PAGE 2 ---", "")

        # Take first 2 lines that aren't empty (usually title area)
        lines = [l.strip() for l in snippet.split("\n") if l.strip()]
        keywords = " ".join(lines[:3])

        # Trim to reasonable search query length
        return keywords[:150]