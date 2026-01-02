from youtube_transcript_api import YouTubeTranscriptApi
import json
import os
from typing import List, Dict, Any, Optional
from langdetect import detect, DetectorFactory
from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DetectorFactory.seed = 0
_hi_en_translator = None
_retriever = None
_llm = None

def get_retriever():
    global _retriever
    if _retriever is None:
        from llmrag import RAGRetrieval
        from dataprep import embedding_manager, vector_store
        _retriever = RAGRetrieval(vector_store, embedding_manager)
    return _retriever

def get_llm():
    global _llm
    if _llm is None:
        from llmrag import ChatGroq
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        _llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant", temperature=0)
    return _llm

def get_youtube_transcript(video_url, languages: Optional[List[str]] = None):
    try:
        video_id = ""
        if "v=" in video_url: video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url: video_id = video_url.split("youtu.be/")[1].split("?")[0]
        elif "/live/" in video_url: video_id = video_url.split("/live/")[1].split("?")[0]
        elif "/shorts/" in video_url: video_id = video_url.split("/shorts/")[1].split("?")[0]
        else: return "Error: Could not find video ID in URL"

        print(f"Fetching transcript for Video ID: {video_id}")
        languages = languages or ['hi', 'en']

        try:
            if hasattr(YouTubeTranscriptApi, 'get_transcript'):
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            else:
                transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=languages)
        except TypeError:
            transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=languages)
        except Exception as e:
             return f"Error fetching transcript: {str(e)}"

        texts = []
        for item in transcript_list:
            if isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])
            else:
                texts.append(getattr(item, 'text', str(item)))

        full_text = " ".join(texts)
        return full_text
        
    except Exception as e:
        return f"Error: {str(e)}"

def extract_claims(text: str) -> List[str]:
    sents = [s.strip() for s in text.split('.') if s.strip()]
    return [s for s in sents if len(s.split()) >= 5 and any(w.isalpha() for w in s)]

def detect_language_of_text(text: str) -> str:
    try: lang = detect(text)
    except Exception: lang = 'unknown'
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

def verify_claim_with_context(claim: str, context: str, llm, mode: str = "Factual Accuracy Check", max_ctx_chars: int = 4000) -> dict:
    
    # --- CUSTOMIZE INSTRUCTION BASED ON MODE ---
    if mode == "Misleading Context Detection":
        instruction = (
            "You are a 'Misleading Context' detector. \n"
            "Analyze if the claim misrepresents the context of the provided document.\n"
            "Return JSON keys: verdict (SUPPORTED|REFUTED|MISLEADING), evidence (snippet), explanation."
        )
    else:
        # Default: Factual Accuracy
        instruction = (
            "You are a factual verifier. \n"
            "Check if the claim is FACTUALLY supported by the CONTEXT.\n"
            "Return JSON keys: verdict (SUPPORTED|REFUTED|INSUFFICIENT), evidence (snippet), explanation."
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("human", "Context:\n{context}\n\nClaim: {claim}\n\nReturn JSON only.")
    ])

    ctx = context if len(context) <= max_ctx_chars else context[:max_ctx_chars] + " ...[truncated]"

    chain = prompt | llm | StrOutputParser()
    try:
        raw = chain.invoke({"context": ctx, "claim": claim})
    except Exception as e:
        raw = f'{{"verdict":"INSUFFICIENT","evidence":"","explanation":"LLM error: {str(e)}"}}'

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
        low = raw.lower()
        if "support" in low: v = "SUPPORTED"
        elif "refut" in low or "no" in low: v = "REFUTED"
        elif "mislead" in low: v = "MISLEADING"
        else: v = "INSUFFICIENT"
        return {"claim": claim, "verdict": v, "evidence": "", "explanation": raw}

def verify_claims(claims: List[str], retriever: Optional[object] = None, llm=None, top_k: int = 3, mode: str = "Factual Accuracy Check") -> List[dict]:
    if retriever is None: retriever = get_retriever()
    if llm is None: llm = get_llm()

    results = []
    for claim in claims:
        docs = retriever.retrieve(claim, top_k=top_k)
        context = "\n\n".join([d['content'] for d in docs]) if docs else ""
        # Pass MODE here
        verification = verify_claim_with_context(claim, context, llm, mode=mode)
        verification["retrieved_docs"] = docs
        results.append(verification)
    return results

def summarize_verification(results: List[dict]) -> str:
    supported = [r for r in results if r["verdict"] == "SUPPORTED"]
    refuted = [r for r in results if r["verdict"] == "REFUTED"]
    misleading = [r for r in results if r["verdict"] == "MISLEADING"]
    insufficient = [r for r in results if r["verdict"] not in ("SUPPORTED", "REFUTED", "MISLEADING")]

    lines = []
    lines.append(f"Summary: {len(supported)} supported, {len(refuted)} refuted, {len(misleading)} misleading, {len(insufficient)} insufficient.")
    
    if supported:
        lines.append("\n✅ Supported claims:")
        for r in supported: lines.append(f"- {r['claim']}\n  Evidence: {r.get('evidence')}")
    
    if refuted:
        lines.append("\n❌ Refuted claims:")
        for r in refuted: lines.append(f"- {r['claim']}\n  Explanation: {r.get('explanation')}")

    if misleading:
        lines.append("\n⚠️ Misleading/Context Missing:")
        for r in misleading: lines.append(f"- {r['claim']}\n  Explanation: {r.get('explanation')}")
        
    if insufficient:
        lines.append("\n❓ Insufficient Data:")
        for r in insufficient: lines.append(f"- {r['claim']}")

    return "\n".join(lines)

def run_video_verification(video_url: str, languages: Optional[List[str]] = None, retriever: Optional[object] = None, llm=None, mode: str = "Factual Accuracy Check"):
    txt = get_youtube_transcript(video_url, languages=languages or ['hi', 'en'])
    if "Error" in txt:
        return txt
    
    lang = detect_language_of_text(txt)
    if lang and lang.startswith('hi'):
        txt = translate_hi_to_en(txt)

    claims = extract_claims(txt)
    if not claims:
        return "No claims extracted from video transcript."

    if retriever is None: retriever = get_retriever()
    if llm is None: llm = get_llm()
    
    # Pass MODE to verification
    results = verify_claims(claims, retriever=retriever, llm=llm, mode=mode)
    summary = summarize_verification(results)
    return summary

if __name__ == "__main__":
    url = input("Enter YouTube video URL: ")
    mode = input("Mode (Factual/Misleading): ")
    result = run_video_verification(url, mode=mode)
    print(result)