"""
FABRIC-AI RAG Query Engine
Dual-source RAG: ChromaDB (scene logs) + FAISS (research papers) → Gemini 2.5 → scene config JSON

FIXED: Swapped target model to gemini-2.5-flash to completely clear free-tier quota limits.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.resolve()
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
FAISS_INDEX_PATH = PROJECT_ROOT / "datasets" / "knowledge_base" / "faiss_index"
OUTPUT_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag_generated_config.json"

CLASS_NAMES = ["car", "auto", "truck", "bus", "cow"]

DEFAULT_CONFIG = {
    "weather": "clear",
    "time_of_day": "midday",
    "object_density": 0.6,
    "auto_rickshaw_count": 3,
    "cow": False,
    "road_type": "asphalt",
    "sun_intensity": 1.0,
    "rain_intensity": 0.0,
    "num_scenes": 5,
    "occlusion_level": 0.3
}

# ── ChromaDB query ────────────────────────────────────────────────────────────
def query_chromadb(user_query: str, n_results: int = 3) -> str:
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        ef = embedding_functions.DefaultEmbeddingFunction()
        collection = client.get_collection(
            name="traffic_scenes",
            embedding_function=ef
        )
        
        count = collection.count()
        if count == 0:
            return "No historical scene logs tracked yet."
            
        results = collection.query(
            query_texts=[user_query],
            n_results=min(n_results, count)
        )
        docs = results["documents"][0]
        print(f"[ChromaDB] Retrieved {len(docs)} scene logs")
        return "\n\n".join(docs)
    except Exception as e:
        print(f"[ChromaDB] Warning: {e}")
        return "No scene logs available."

# ── FAISS query ───────────────────────────────────────────────────────────────
def query_faiss(user_query: str, n_results: int = 3) -> str:
    try:
        if not FAISS_INDEX_PATH.exists():
            print("[FAISS] Index not found — skipping research paper retrieval")
            return "No research paper context available."

        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )
        docs = vectorstore.similarity_search(user_query, k=n_results)
        context = "\n\n".join([d.page_content for d in docs])
        print(f"[FAISS] Retrieved {len(docs)} research paper chunks")
        return context
    except Exception as e:
        print(f"[FAISS] Warning: {e}")
        return "No research paper context available."

# ── Gemini call — Target 2.5 Flash Production ────────────────────────────────
def call_gemini(prompt: str) -> str:
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        print("[Gemini] No GEMINI_API_KEY found in the system environment variables.")
        return ""

    try:
        # Utilizing the official modern Google GenAI library
        import google.genai as genai
        from google.genai import types
        
        # Force production v1 parameters to avoid preview/beta metadata routing drops
        client = genai.Client(
            api_key=gemini_key,
            http_options=types.HttpOptions(api_version="v1")
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text

    except Exception as e1:
        print(f"[Gemini] Modern Client Layer dropped ({e1}). Triaging legacy backup route...")
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=gemini_key)
            model = genai_legacy.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e2:
            print(f"[Gemini] Both SDK pipelines terminated structural routing: {e2}")
            return ""

# ── JSON extractor ────────────────────────────────────────────────────────────
def extract_json(text: str) -> dict:
    if not text:
        return {}
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}

# ── Main pipeline execution loop ──────────────────────────────────────────────
def run(user_query: str) -> dict:
    print("\nFABRIC-AI RAG Query Engine")
    print(f"Query: '{user_query}'")
    print("=" * 70)

    # 1. Retrieve information blocks
    print("\n[Step 1] Querying ChromaDB (scene logs)...")
    scene_context = query_chromadb(user_query)

    print("\n[Step 2] Querying FAISS (research papers)...")
    paper_context = query_faiss(user_query)

    # 2. Structure generative optimization context guidelines
    prompt = f"""You are an expert engine optimizer for FABRIC-AI, generating parameters for synthetic 3D driving scenarios inside BlenderProc.

ACADEMIC LITERATURE RESEARCH CONTEXT (FAISS):
{paper_context}

HISTORICAL SIMULATION VISION MEMORIES (ChromaDB):
{scene_context}

USER TARGET OPTIMIZATION REQUEST:
{user_query}

Analyze the constraints and past data above to produce optimal simulation generation parameters.

IMPORTANT: You must return ONLY a raw, unformatted valid JSON object block structure. No markdown wrappers, descriptions, or commentary text.

Required Target JSON Field Schema:
{{
  "weather": "clear" or "rainy" or "foggy",
  "time_of_day": "morning" or "midday" or "evening" or "night",
  "object_density": <float between 0.3 and 1.0>,
  "auto_rickshaw_count": <integer between 1 and 8>,
  "cow": <true or false>,
  "road_type": "asphalt" or "sand",
  "sun_intensity": <float between 0.0 and 1.5>,
  "rain_intensity": <float between 0.0 and 1.0>,
  "num_scenes": <integer between 3 and 10>,
  "occlusion_level": <float between 0.1 and 0.8>
}}"""

    # 3. Request analysis response
    print("\n[Step 3] Querying Gemini 2.5 Flash Generation Target Engine...")
    raw_response = call_gemini(prompt)

    # 4. Clean structural extraction
    config = {}
    if raw_response:
        config = extract_json(raw_response)
        if config:
            print("[Gemini] JSON config successfully created and structural mapping verified.")
        else:
            print(f"[Gemini] Evaluation syntax error parsing out text block:\n{raw_response[:300]}")

    # 5. Populate configurations
    final_config = DEFAULT_CONFIG.copy()
    final_config.update(config)

    # 6. Sanity parsing corrections
    final_config["object_density"] = float(final_config.get("object_density", 0.6))
    final_config["auto_rickshaw_count"] = int(final_config.get("auto_rickshaw_count", 3))
    final_config["cow"] = bool(final_config.get("cow", False))
    final_config["sun_intensity"] = float(final_config.get("sun_intensity", 1.0))
    final_config["rain_intensity"] = float(final_config.get("rain_intensity", 0.0))
    final_config["num_scenes"] = int(final_config.get("num_scenes", 5))
    final_config["occlusion_level"] = float(final_config.get("occlusion_level", 0.3))

    # 7. Local storage persistency dump
    OUTPUT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CONFIG_PATH, "w") as f:
        json.dump(final_config, f, indent=2)

    print(f"\nConfiguration Engine verified and output written to: {OUTPUT_CONFIG_PATH}")
    print(json.dumps(final_config, indent=2))
    print("=" * 70)

    source = "Gemini 2.5 + Dual-RAG Fusion Network" if config else "System Hardcoded Fallback Profiles"
    print(f"[Info] Simulation pipeline driven by: {source}\n")

    return final_config

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "A dangerous foggy morning in Chennai with stray animals on the asphalt"
    run(query)