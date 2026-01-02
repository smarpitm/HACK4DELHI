import os
import uuid
import numpy as np
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Step 1: Data Parsing 
print("--- Loading Documents ---")
dir_loader = DirectoryLoader(
    "./data",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=False
)

Documents = dir_loader.load()
print(f"Loaded {len(Documents)} pages/documents.")

# Step 2: Chunking 
print("\n--- Splitting Text ---")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(Documents)
print(f"Total Chunks created: {len(chunks)}")

#  Step 3: Embedding Manager 
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"\nLoading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise  # Keep raise here only if loading fails

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

#  Step 4: Vector Store Manager 
class VectorStoreManager:
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "./data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
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
            print(f"\nVector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")

        print(f"Preparing to add {len(documents)} documents to vector store...")
        
        ids = []
        metadatas = []
        documents_text = []
        
        # Prepare data lists
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare metadata (ensure it's a flat dict)
            meta = doc.metadata.copy() if doc.metadata else {}
            meta['doc_index'] = i
            meta['content_length'] = len(doc.page_content)
            metadatas.append(meta)

            # Document content
            documents_text.append(doc.page_content)

        # Add to collection (Batch Operation)
        try:
            # Note: Chroma expects embeddings as a list of lists, not numpy array
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store.")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise



# 1. Initialize Embedding Manager
embedding_manager = EmbeddingManager()

# 2. Extract text content from chunks for embedding generation
texts = [doc.page_content for doc in chunks]

# 3. Generate Embeddings
embeddings = embedding_manager.generate_embeddings(texts)

# 4. Initialize Vector Store
vector_store = VectorStoreManager()

# 5. Add to Store
vector_store.add_documents(chunks, embeddings)
#rag reterival pipeline from vector store 
