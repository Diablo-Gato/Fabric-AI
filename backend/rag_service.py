"""
FABRIC-AI RAG Service
Handles retrieval and scene parameter generation using RAG pipeline.

CHANGES FROM ORIGINAL:
- generate_parameters() now calls the LLM (OpenAI or Gemini) using retrieved context
- Falls back to rule-based only if LLM call fails or API key missing
- Gemini support added as drop-in alternative when OPENAI credits run out
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "datasets" / "knowledge_base"
FAISS_INDEX_PATH = KNOWLEDGE_BASE_DIR / "faiss_index"

DEFAULT_PARAMS = {
    "sun_intensity": 1.0,
    "rain_intensity": 0.0,
    "camera_angle": 25,
    "num_scenes": 3,
    "lighting_condition": "day",
    "time_of_day": "midday",
    "weather": "clear",
    "camera_distance": 15,
    "fov": 60,
    "object_density": 0.5,
    "occlusion_level": 0.3,
}

# ── LLM prompt template ───────────────────────────────────────────────────────
SCENE_PROMPT = """You are an expert in synthetic 3D dataset generation for Indian road scenes.
Using the research context below, generate realistic rendering parameters.

Research Context:
{context}

User Request: {user_prompt}

Return ONLY a valid JSON object (no markdown, no explanation) with these exact keys:
{{
  "weather": "clear" | "rainy" | "foggy",
  "time_of_day": "morning" | "midday" | "evening" | "night",
  "object_density": <float 0.3-1.0>,
  "auto_rickshaw_count": <int 1-8>,
  "cow": <true | false>,
  "road_type": "asphalt" | "sand",
  "sun_intensity": <float 0.0-1.5>,
  "rain_intensity": <float 0.0-1.0>,
  "num_scenes": <int 3-10>,
  "occlusion_level": <float 0.1-0.8>
}}"""


class RAGService:
    """RAG-based scene parameter generation service."""

    def __init__(self, vectorstore: Optional[FAISS] = None):
        self.vectorstore = vectorstore
        self._llm = None

    # ── vector store ──────────────────────────────────────────────────────────
    def _load_vectorstore(self) -> Optional[FAISS]:
        if self.vectorstore is not None:
            return self.vectorstore

        if not FAISS_INDEX_PATH.exists():
            logger.warning("FAISS index not found. Run setup_rag.py first.")
            return None

        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.vectorstore = FAISS.load_local(
                str(FAISS_INDEX_PATH),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            return self.vectorstore
        except Exception as e:
            logger.warning(f"Could not load FAISS: {e}")
            return None

    def retrieve_context(self, query: str, k: int = 5) -> str:
        vectorstore = self._load_vectorstore()
        if vectorstore is None:
            return "Default Indian road scene: mixed traffic, clear weather, midday."
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])
            logger.info(f"Retrieved {len(docs)} documents from FAISS")
            return context
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return "Default Indian road scene parameters."

    # ── LLM loader ───────────────────────────────────────────────────────────
    def _get_llm(self):
        """
        Load LLM. Tries OpenAI first, then Gemini (google-generativeai).
        Returns None if neither is available.
        """
        if self._llm is not None:
            return self._llm

        # 1. Try OpenAI
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key and not openai_key.startswith("sk-your"):
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.2,
                    openai_api_key=openai_key,
                )
                logger.info("Using OpenAI GPT-3.5-turbo")
                return self._llm
            except Exception as e:
                logger.warning(f"OpenAI init failed: {e}")

        # 2. Try Gemini
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        if gemini_key and not gemini_key.startswith("your-"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0.2,
                    google_api_key=gemini_key,
                )
                logger.info("Using Google Gemini 1.5 Flash")
                return self._llm
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}")

        logger.warning("No LLM available — will use rule-based fallback.")
        return None

    # ── parameter generation ──────────────────────────────────────────────────
    def generate_parameters(self, user_prompt: str) -> Dict[str, Any]:
        """
        Generate scene parameters via RAG + LLM.
        Falls back to rule-based if LLM unavailable.
        """
        logger.info(f"Generating parameters for: {user_prompt}")

        context = self.retrieve_context(user_prompt)
        llm = self._get_llm()

        if llm is not None:
            try:
                prompt = PromptTemplate(
                    input_variables=["context", "user_prompt"],
                    template=SCENE_PROMPT,
                )
                chain = prompt | llm
                response = chain.invoke({
                    "context": context,
                    "user_prompt": user_prompt,
                })
                # Extract text from LLM response object
                raw = response.content if hasattr(response, "content") else str(response)
                raw = raw.strip()
                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()
                params = json.loads(raw)
                # Merge with defaults so no key is ever missing
                merged = DEFAULT_PARAMS.copy()
                merged.update(params)
                logger.info("LLM parameter generation succeeded")
                return merged
            except Exception as e:
                logger.warning(f"LLM generation failed ({e}), using rule-based fallback")

        return self._fallback_parameter_generation(user_prompt)

    def _fallback_parameter_generation(self, user_prompt: str) -> Dict[str, Any]:
        """Rule-based fallback — unchanged from original."""
        prompt_lower = user_prompt.lower()
        params = DEFAULT_PARAMS.copy()

        if "rain" in prompt_lower:
            params["rain_intensity"] = 0.6
            params["weather"] = "rainy"
            params["sun_intensity"] = 0.4

        if "night" in prompt_lower:
            params["sun_intensity"] = 0.05
            params["lighting_condition"] = "night"
            params["time_of_day"] = "night"
        elif "evening" in prompt_lower:
            params["sun_intensity"] = 0.3
            params["time_of_day"] = "evening"

        if "morning" in prompt_lower:
            params["time_of_day"] = "morning"
            params["sun_intensity"] = 0.6

        if "fog" in prompt_lower:
            params["weather"] = "foggy"
            params["occlusion_level"] = 0.6

        if "auto" in prompt_lower or "rickshaw" in prompt_lower:
            params["auto_rickshaw_count"] = 5

        if "cow" in prompt_lower:
            params["cow"] = True

        if "pedestrian" in prompt_lower or "people" in prompt_lower:
            params["pedestrian_count"] = 3

        if "busy" in prompt_lower or "traffic" in prompt_lower:
            params["object_density"] = 0.85

        if "quiet" in prompt_lower or "empty" in prompt_lower:
            params["object_density"] = 0.3

        return params

    # ── public API ────────────────────────────────────────────────────────────
    def generate_scenes(
        self, user_prompt: str, num_scenes: Optional[int] = None
    ) -> List[Dict[str, Any]]:

        params = self.generate_parameters(user_prompt)

        if num_scenes is None:
            num_scenes = params.get("num_scenes", DEFAULT_PARAMS["num_scenes"])

        scenes = []
        for i in range(num_scenes):
            scene = params.copy()
            scene["scene_id"] = i + 1
            scene["auto_rickshaw_count"] = params.get("auto_rickshaw_count", 3)
            scene["cow"] = params.get("cow", False)
            scene["pedestrian_count"] = params.get("pedestrian_count", 2)
            scenes.append(scene)

        return scenes


def get_rag_service() -> RAGService:
    return RAGService()


if __name__ == "__main__":
    service = get_rag_service()
    scenes = service.generate_scenes(
        "Rainy evening in Mumbai with auto-rickshaws and pedestrians"
    )
    print(json.dumps(scenes, indent=2))
