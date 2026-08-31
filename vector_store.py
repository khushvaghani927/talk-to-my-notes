import os
import time
import shutil
import re
from typing import List, Tuple, Optional, Any
import chromadb
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma


class LocalFastEmbeddings(Embeddings):
    """
    100% Free, Local ONNX Embeddings (all-MiniLM-L6-v2) powered by ChromaDB.
    Runs locally on CPU with ZERO API calls, ZERO quota limits, and instant indexing.
    """

    def __init__(self):
        import chromadb.utils.embedding_functions as ef
        self._ef = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self._ef(texts)
        return [list(map(float, vec)) for vec in embeddings]

    def embed_query(self, text: str) -> List[float]:
        res = self._ef([text])
        return [float(x) for x in res[0]]


class GeminiBatchedEmbeddings(Embeddings):
    """
    Batched Google Gemini Embeddings with auto-retry on 429 and
    automatic fallback to LocalFastEmbeddings if quota is fully exhausted.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        batch_size: int = 50,
        delay_between_batches: float = 0.4
    ):
        import google.genai as genai
        self.api_key = api_key
        self.model = model.replace("models/", "")
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.client = genai.Client(api_key=self.api_key)
        self._local_fallback = None

    def _get_local_fallback(self) -> LocalFastEmbeddings:
        if self._local_fallback is None:
            self._local_fallback = LocalFastEmbeddings()
        return self._local_fallback

    def _embed_with_retry(self, contents: Any, max_retries: int = 3) -> List[List[float]]:
        retry_delay = 4.0

        for attempt in range(1, max_retries + 1):
            try:
                res = self.client.models.embed_content(
                    model=self.model,
                    contents=contents
                )
                if isinstance(contents, list):
                    return [list(e.values) for e in res.embeddings]
                else:
                    return list(res.embeddings[0].values)

            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                    print(f"[Gemini Embedding Quota Reached] Attempt {attempt}/{max_retries}. Falling back or waiting...")
                    if attempt == max_retries:
                        print("[Auto-Fallback] Using LocalFastEmbeddings to complete indexing without errors.")
                        if isinstance(contents, list):
                            return self._get_local_fallback().embed_documents(contents)
                        else:
                            return self._get_local_fallback().embed_query(contents)
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise e

        if isinstance(contents, list):
            return self._get_local_fallback().embed_documents(contents)
        else:
            return self._get_local_fallback().embed_query(contents)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._embed_with_retry(batch)
            all_embeddings.extend(embeddings)
            if i + self.batch_size < len(texts):
                time.sleep(self.delay_between_batches)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed_with_retry(text)


def get_embedding_function(
    provider: str = "local",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Any:
    """
    Returns the selected embedding function.
    Default: LocalFastEmbeddings (100% free, unlimited, zero quota limits).
    """
    provider = provider.lower()

    if "local" in provider or "fast" in provider or "huggingface" in provider:
        return LocalFastEmbeddings()

    elif provider in ["gemini", "google"]:
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            return LocalFastEmbeddings()
        
        raw_model = model_name or "gemini-embedding-001"
        cleaned_model = raw_model.replace("models/", "")
        
        if cleaned_model in ["text-embedding-004", "models/text-embedding-004"]:
            cleaned_model = "gemini-embedding-001"
        
        return GeminiBatchedEmbeddings(
            api_key=key,
            model=cleaned_model,
            batch_size=50,
            delay_between_batches=0.4
        )

    elif provider in ["openai"]:
        from langchain_openai import OpenAIEmbeddings
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API Key is required for OpenAI embeddings.")
        embedding_model = model_name or "text-embedding-3-small"
        return OpenAIEmbeddings(
            model=embedding_model,
            api_key=key
        )

    else:
        return LocalFastEmbeddings()


class VectorStoreManager:
    """
    Manages completely isolated, in-memory ChromaDB vector collections per upload session.
    Guarantees ZERO data bleed or persistence across different PDF files.
    """

    def __init__(self):
        self.chroma_client: Optional[chromadb.ClientAPI] = None
        self.vector_store: Optional[Chroma] = None
        self.collection_name: Optional[str] = None

    def create_vector_store(
        self,
        chunks: List[Document],
        embedding_function: Any
    ) -> Chroma:
        """
        Creates a brand-new, clean in-memory vector store for the current PDF documents.
        Completely discards any previous database instance.
        """
        if not chunks:
            raise ValueError("Cannot create vector store with empty document chunks.")

        # 1. Reset client and create a fresh Ephemeral in-memory Chroma client
        self.chroma_client = chromadb.EphemeralClient()
        self.collection_name = f"doc_{int(time.time() * 1000)}"

        # 2. Instantiate isolated Chroma vector store in RAM
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name=self.collection_name,
            embedding_function=embedding_function
        )

        # 3. Add chunks for the current document
        self.vector_store.add_documents(chunks)
        return self.vector_store

    def get_retriever(self, k: int = 5, search_type: str = "similarity"):
        if not self.vector_store:
            raise ValueError("Vector store has not been initialized.")
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        if not self.vector_store:
            raise ValueError("Vector store has not been initialized.")
        try:
            return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception:
            docs = self.vector_store.similarity_search(query, k=k)
            return [(d, 0.0) for d in docs]

    def clear(self):
        """
        Completely purges the current vector store and in-memory client.
        """
        self.chroma_client = None
        self.vector_store = None
        self.collection_name = None
