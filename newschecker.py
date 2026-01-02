import trafilatura
import json
import os
from typing import List, Optional
from langdetect import detect, DetectorFactory
from transformers import pipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# make langdetect deterministic
DetectorFactory.seed = 0

_hi_en_translator = None
_retriever = None
_llm = None

def get_retriever():
    """Lazy load retriever"""
    global _retriever
    if _retriever is None:
        from llmrag import RAGRetrieval
        from dataprep import embedding_manager, vector_store
        _retriever = RAGRetrieval(vector_store, embedding_manager)
    return _retriever

def get_llm():
    """Lazy load LLM"""
    global _llm
    if _llm is None:
        from llmrag import ChatGroq
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        _llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0)
    return _llm

def extract_news_content(url: str) -> str:
    try:
        downloaded_data = trafilatura.fetch_url(url)
        if downloaded_data is None:
            return ""
        result = trafilatura.extract(
            downloaded_data,
            include_comments=False,
            include_tables=False,
            no_fallback=False
        )
        return result if result else ""
    except Exception as e:
        print(f"An exception occurred: {e}")
        return ""

def extract_claims(text: str) -> List[str]:
    sents = [s.strip() for s in text.split('.') if s.strip()]
    return [s for s in sents if len(s.split()) >= 5 and any(w.isalpha() for w in s)]

def detect_language_of_text(text: str) -> str:
    try:
        lang = detect(text)
    except Exception:
        lang = 'unknown'
    return lang

def translate_hi_to_en(text: str) -> str:
    global _hi_en_translator
    if _hi_en_translator is None:
        _hi_en_translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")

    max_chunk = 1000
    parts = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    translated = []
    for part in parts:
        out = _hi_en_translator(part)
        if isinstance(out, list) and isinstance(out[0], dict):
            translated.append(out[0].get('translation_text') or out[0].get('text') or str(out[0]))
        else:
            translated.append(str(out))
    return " ".join(translated)

def verify_claim_with_context(claim: str, context: str, llm, max_ctx_chars: int = 4000, mode: str = "Factual Accuracy") -> dict:
    """
    Ask LLM to check the claim. Logic changes based on 'mode'.
    """
    
    if mode == "Political Bias Detection":
        # --- BIAS DETECTION PROMPT ---
        system_instruction = (
            "You are a media bias analyst. "
            "Analyze the provided text excerpt (Claim) for political bias, loaded language, or subjective framing. "
            "Ignore the Context if it's not relevant to bias, focus on the tone of the Claim itself. "
            "Return a JSON object with keys: "
            "verdict (BIASED | NEUTRAL), "
            "evidence (the specific biased words/phrases used), "
            "explanation (why it is biased)."
        )
    else:
        # --- FACT CHECK PROMPT (Default) ---
        system_instruction = (
            "You are an assistant that checks factual claims against the provided CONTEXT. "
            "Return a JSON object with keys: "
            "verdict (SUPPORTED | REFUTED | INSUFFICIENT), "
            "evidence (short snippet from context), "
            "explanation (short reason)."
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "Context:\n{context}\n\nClaim/Text to Analyze: {claim}\n\nReturn JSON as described.")
    ])

    # keep context length bounded
    ctx = context if len(context) <= max_ctx_chars else context[:max_ctx_chars] + " ...[truncated]"

    chain = prompt | llm | StrOutputParser()
    try:
        raw = chain.invoke({"context": ctx, "claim": claim})
    except Exception as e:
        raw = f'{{"verdict":"INSUFFICIENT","evidence":"","explanation":"LLM error: {str(e)}"}}'

    # JSON Parsing
    try:
        parsed = json.loads(raw)
        parsed['verdict'] = str(parsed.get('verdict', '')).strip().upper()
        return {
            "claim": claim,
            "verdict": parsed.get("verdict", "INSUFFICIENT"),
            "evidence": parsed.get("evidence", ""),
            "explanation": parsed.get("explanation", "")
        }
    except Exception:
        # Fallback heuristic
        low = raw.lower()
        if "biased" in low: v = "BIASED"
        elif "neutral" in low: v = "NEUTRAL"
        elif "support" in low: v = "SUPPORTED"
        elif "refut" in low or "contrad" in low or "no" in low: v = "REFUTED"
        else: v = "INSUFFICIENT"
        return {"claim": claim, "verdict": v, "evidence": "", "explanation": raw}


def verify_claims(claims: List[str], retriever: Optional[object] = None, llm=None, top_k: int = 3, mode: str = "Factual Accuracy") -> List[dict]:
    """
    Retrieves docs and verifies claims based on the selected MODE.
    """
    if retriever is None:
        retriever = get_retriever()

    if llm is None:
        llm = get_llm()

    results = []
    for claim in claims:
        # For bias, we might not strictly need retrieval, but context helps identifying misrepresentation.
        # We keep retrieval to maintain consistency.
        docs = retriever.retrieve(claim, top_k=top_k)
        context = "\n\n".join([d['content'] for d in docs]) if docs else ""
        
        # Pass the mode down
        verification = verify_claim_with_context(claim, context, llm, mode=mode)
        
        verification["retrieved_docs"] = docs
        results.append(verification)
    return results


def summarize_verification(results: List[dict]) -> str:
    # Existing Fact Check Buckets
    supported = [r for r in results if r["verdict"] == "SUPPORTED"]
    refuted = [r for r in results if r["verdict"] == "REFUTED"]
    
    # New Bias Buckets
    biased = [r for r in results if r["verdict"] == "BIASED"]
    neutral = [r for r in results if r["verdict"] == "NEUTRAL"]
    
    insufficient = [r for r in results if r["verdict"] not in ("SUPPORTED", "REFUTED", "BIASED", "NEUTRAL")]

    lines = []
    
    # If we have Bias results, show Bias summary
    if biased or neutral:
        lines.append(f"**Bias Analysis Report:**\nFound {len(biased)} biased statements and {len(neutral)} neutral statements.")
        if biased:
            lines.append("\n🔴 **Biased / Loaded Language Detected:**")
            for r in biased:
                lines.append(f"- \"{r['claim']}\"\n  → *Issue:* {r.get('evidence')}\n  → *Note:* {r.get('explanation')}")
        if neutral:
            lines.append("\n✅ **Neutral / Objective Statements:**")
            for r in neutral:
                lines.append(f"- \"{r['claim']}\" (Neutral)")
                
    # If we have Fact Check results, show Fact summary
    elif supported or refuted:
        lines.append(f"**Fact Check Summary:**\n{len(supported)} supported, {len(refuted)} refuted.")
        if supported:
            lines.append("\n✅ **Supported Claims:**")
            for r in supported:
                lines.append(f"- {r['claim']}\n  Evidence: {r.get('evidence')}")
        if refuted:
            lines.append("\n❌ **Refuted / Contradicted Claims:**")
            for r in refuted:
                lines.append(f"- {r['claim']}\n  Evidence: {r.get('evidence')}\n  Correction: {r.get('explanation')}")

    # Show insufficient for both
    if insufficient:
        lines.append("\n⚠️ **Inconclusive / Insufficient Info:**")
        for r in insufficient:
            lines.append(f"- {r['claim']}")

    if not lines:
        return "No verifiable claims or bias detected."

    return "\n".join(lines)

if __name__ == "__main__":
    # Example usage - Only runs when running this file directly
    url = "https://indianexpress.com/article/india/3-kashmiris-us-germany-booked-j-k-weaponising-social-media-srinagar-court-10447105/?ref=hometop_hp"
    content = extract_news_content(url)
    if content:
        claims = extract_claims(content)
        print("Testing Bias Detection Mode:")
        results = verify_claims(claims[:3], mode="Political Bias Detection")
        print(summarize_verification(results))