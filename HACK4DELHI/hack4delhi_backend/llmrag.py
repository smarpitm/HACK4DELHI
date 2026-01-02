import os
import logging
import time
from typing import List, Dict, Any

# Third-party imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

from dataprep import VectorStoreManager, EmbeddingManager

logging.basicConfig(level=logging.ERROR) 

if load_dotenv(find_dotenv()):
    print("✅ Loaded environment variables from .env")
elif load_dotenv("1.env"):
    print("✅ Loaded environment variables from 1.env")
else:
    print("⚠️  Warning: Could not find .env or 1.env file.")

class RAGRetrieval:
    def __init__(self, vector_store: VectorStoreManager, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        try:
            print(f"🔍 Analyzing top {top_k} document segments...", end="\r")
            
            query_embedding_np = self.embedding_manager.generate_embeddings([query])[0]
            query_embedding_list = query_embedding_np.tolist()

            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved_docs = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    doc_data = {
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "score": results['distances'][0][i],
                        "id": results['ids'][0][i]
                    }
                    retrieved_docs.append(doc_data)
            
            print(f"✅ Analysis complete. Found {len(retrieved_docs)} relevant segments.    ")
            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

class RAGGenerator:
    def __init__(self, api_key: str, model_name: str = "llama-3.1-8b-instant"):
        if not api_key:
            raise ValueError("❌ Groq API Key is missing.")
            
        self.llm = ChatGroq(
            api_key=api_key, 
            model=model_name, 
            temperature=0.1, 
            streaming=True
        )
        # We no longer build a static prompt here; we build it dynamically in generate_stream

    def format_context(self, docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return "No relevant context found."
        
        formatted_parts = []
        for i, doc in enumerate(docs):
            source = doc['metadata'].get('source', 'Unknown File')
            content = doc['content'][:2500] 
            formatted_parts.append(f"--- Segment {i+1} [Source: {source}] ---\n{content}")
            
        return "\n\n".join(formatted_parts)

    def generate_stream(self, query: str, retrieved_docs: List[Dict[str, Any]], category: str = "General"):
        context_str = self.format_context(retrieved_docs)
        
        # --- DYNAMIC PROMPT INJECTION ---
        system_instruction = (
            "You are a high-precision analyst for the 'One Nation One Election' panel.\n"
            f"FOCUS CATEGORY: {category}\n" 
            "Guidelines:\n"
            "1. Analyze the Question primarily through the lens of the FOCUS CATEGORY.\n"
            "2. USE CONTEXT: The answer might be split across multiple excerpts. Read carefully.\n"
            "3. BE FACTUAL: Only answer based on the provided Context. Do not hallucinate.\n"
            "4. FALLBACK: If the answer is NOT in the context, use general knowledge but start with: "
            "'**Based on general knowledge (not found in documents):**'\n"
            "5. CITATIONS: Mention the source file (e.g., '[Source: report.pdf]')."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", "Context:\n{formatted_context}\n\nQuestion: {query}")
        ])

        try:
            chain = prompt | self.llm | StrOutputParser()
            
            print("\n🤖 Answer: ", end="", flush=True)
            
            full_response = ""
            for chunk in chain.stream({"formatted_context": context_str, "query": query}):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            print() 
            return full_response
            
        except Exception as e:
            error_msg = str(e)
            if "413" in error_msg or "rate_limit_exceeded" in error_msg:
                 print("\n\n⚠️  RATE LIMIT REACHED.")
            else:
                print(f"Error during generation: {e}")
            return ""

# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 Initializing Precision System...")
    try:
        embedding_manager = EmbeddingManager()
        vector_store = VectorStoreManager(collection_name="pdf_documents", embedding_manager=embedding_manager)
        rag_retriever = RAGRetrieval(vector_store, embedding_manager)
        api_key = os.getenv("GROQ_API_KEY")
        rag_generator = RAGGenerator(api_key=api_key)
        
        print("✅ System Ready!\n")

        while True:
            user_query = input("\n👉 Question (or 'q' to quit): ")
            if user_query.lower() in ['q', 'exit']: break
            
            cat = input("👉 Category (e.g., Federal Structure) [Enter for General]: ") or "General"
            
            docs = rag_retriever.retrieve(user_query, top_k=10)
            rag_generator.generate_stream(user_query, docs, category=cat)
            print("-" * 50)

    except Exception as e:
        print(f"❌ Pipeline Error: {e}")