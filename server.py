import os
import sys
# Consolidate imports (removed duplicates)
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv

# --- 1. INITIALIZE FLASK APP (DO THIS ONLY ONCE) ---
app = Flask(__name__)
CORS(app)

# --- 2. LOAD ENVIRONMENT VARIABLES ---
if load_dotenv(find_dotenv()):
    print("✅ API Key loaded from .env")
elif load_dotenv("1.env"):
    print("✅ API Key loaded from 1.env")

# --- 3. SETUP RAG COMPONENTS ---
try:
    from dataprep import VectorStoreManager, EmbeddingManager
    from llmrag import RAGRetrieval, RAGGenerator, ChatGroq
    import newschecker
    import vediochecker
except ImportError as e:
    print(f"\n❌ CRITICAL ERROR: Missing modules. Detail: {e}")
    sys.exit(1)

print("🚀 Booting up RAG Server...")
api_key = os.getenv("GROQ_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ ERROR: API Key missing. Please check your .env file.")

# Initialize RAG Objects
embedding_manager = EmbeddingManager()
vector_store = VectorStoreManager(
    collection_name="pdf_documents", 
    embedding_manager=embedding_manager
)
rag_retriever = RAGRetrieval(vector_store, embedding_manager)
rag_generator = RAGGenerator(api_key=api_key)
llm_engine = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0)

print("✅ Server System Ready on Port 5000!")

# --- 4. ROUTES ---

# This route was getting deleted in your previous code
@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/api/fact-check', methods=['POST'])
def fact_check():
    data = request.json
    user_query = data.get('query')
    category = data.get('category', 'General')
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    print(f"\n📩 [Text Check] Category: {category} | Query: {user_query}")

    try:
        docs = rag_retriever.retrieve(user_query, top_k=5)
        answer = rag_generator.generate_stream(user_query, docs, category=category)
        return jsonify({
            "answer": answer, 
            "sources": [d['metadata'].get('source', 'Unknown') for d in docs]
        })
    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/verify-video', methods=['POST'])
def verify_video():
    data = request.json
    video_url = data.get('url')
    mode = data.get('mode', 'Factual Accuracy Check')
    
    if not video_url:
        return jsonify({"error": "No URL provided"}), 400

    print(f"\n🎥 [Video Check] Mode: {mode} | Processing: {video_url}")

    try:
        summary = vediochecker.run_video_verification(
            video_url, 
            languages=['hi', 'en'], 
            retriever=rag_retriever, 
            llm=llm_engine,
            mode=mode
        )
        return jsonify({"result": summary})
    except Exception as e:
        print(f"Video Check Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/verify-news', methods=['POST'])
def verify_news():
    data = request.json
    news_url = data.get('url')
    filter_mode = data.get('filter_type', 'Factual Accuracy') 
    
    if not news_url:
        return jsonify({"error": "No URL provided"}), 400

    print(f"\n📰 [News Check] Mode: {filter_mode} | URL: {news_url}")

    try:
        content = newschecker.extract_news_content(news_url)
        if not content:
            return jsonify({"error": "Could not extract content."}), 400
        
        lang = newschecker.detect_language_of_text(content)
        if lang and lang.startswith('hi'):
            content = newschecker.translate_hi_to_en(content)
            
        claims = newschecker.extract_claims(content)
        if not claims:
            return jsonify({"result": "No verifiable claims found."})

        top_claims = claims[:5] 
        results = newschecker.verify_claims(
            top_claims, 
            retriever=rag_retriever, 
            llm=llm_engine,
            mode=filter_mode 
        )
        
        summary = newschecker.summarize_verification(results)
        return jsonify({"result": summary})

    except Exception as e:
        print(f"News Check Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- 5. START SERVER ---
if __name__ == '__main__':
    app.run(port=5000, debug=True)