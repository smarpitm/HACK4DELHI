import os
import uuid
import numpy as np
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# --- Class Definitions ---

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"\nLoading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            # Check for GPU
            device = self.model.device
            print(f"Model loaded on: {device}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.encode(texts, show_progress_bar=True)

class VectorStoreManager:
    # UPDATED: Now accepts embedding_manager to fix your error
    def __init__(self, collection_name: str = "pdf_documents", embedding_manager=None, persist_directory: str = "./data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_manager = embedding_manager
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store connected. Collection: {self.collection_name}")
            print(f"Documents in DB: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")

        print(f"Adding {len(documents)} new documents...")
        
        ids = []
        metadatas = []
        documents_text = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            meta = doc.metadata.copy() if doc.metadata else {}
            meta['doc_index'] = i
            metadatas.append(meta)
            documents_text.append(doc.page_content)

        # Add in batches to prevent timeouts
        batch_size = 500
        total_docs = len(ids)
        
        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings.tolist()[i:end_idx],
                metadatas=metadatas[i:end_idx],
                documents=documents_text[i:end_idx]
            )
            print(f"Uploaded batch {i} to {end_idx}")

        print("Upload Complete.")

# --- EXECUTION BLOCK ---
# This ensures PDFs are ONLY loaded when you run 'python dataprep.py' directly.
# They will NOT reload when you run 'llmrag.py'.
if __name__ == "__main__":
    print("--- STARTING DATA INGESTION ---")
    
    # 1. Load Documents
    loader = DirectoryLoader("./data", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    raw_docs = loader.load()
    print(f"Loaded {len(raw_docs)} raw pages.")

    # 2. Split Text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)
    print(f"Created {len(chunks)} chunks.")

    # 3. Generate Embeddings
    embedding_mgr = EmbeddingManager()
    texts = [doc.page_content for doc in chunks]
    vectors = embedding_mgr.generate_embeddings(texts)

    # 4. Store in DB
    vector_store = VectorStoreManager(collection_name="pdf_documents", embedding_manager=embedding_mgr)
    vector_store.add_documents(chunks, vectors)
    
    print("--- INGESTION FINISHED ---")