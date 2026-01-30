import threading
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_LOCK = threading.RLock()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = Chroma(
    collection_name="devdocs_knowledge",
    embedding_function=embeddings,
    persist_directory=None,
)


def vs_add_documents(docs: List[Document]) -> None:
    with VECTOR_LOCK:
        vector_store.add_documents(docs)


def vs_similarity_search(query: str, k: int, tech: Optional[str] = None) -> List[Document]:
    """Similarity search with higher internal candidate pool to allow manual filtering."""
    with VECTOR_LOCK:
        if tech is not None:
            try:
                return vector_store.similarity_search(query, k=k, filter={"tech": tech})
            except TypeError:
                pass
            except Exception:
                pass

        internal_k = max(k * 3, 12)
        docs = vector_store.similarity_search(query, k=internal_k)
        if tech is None:
            return docs[:k]
        filtered = [d for d in docs if d.metadata.get("tech") == tech]
        if len(filtered) < k:
            needed = k - len(filtered)
            filler = [d for d in docs if d not in filtered][:needed]
            filtered.extend(filler)
        return filtered[:k]


def rag_context_for(query: str, tech: Optional[str], k: int = 4, min_chars: int = 120) -> str:
    docs = vs_similarity_search(query, k=k, tech=tech)
    if not docs:
        return ""
    chunks = []
    for doc in docs:
        text = (doc.page_content or "").strip()
        if len(text) < min_chars:
            continue
        chunks.append(f"[Source: {doc.metadata.get('url')}]\n{text[:520]}...")
    return "\n\n".join(chunks).strip()
