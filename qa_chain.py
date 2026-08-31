import os
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI


SYSTEM_PROMPT = """You are "Talk to My Notes", an elite AI academic tutor and problem solver.
Your mission is to directly answer questions, solve exercises, and generate comprehensive study notes strictly based on the student's uploaded document.

🎯 INSTRUCTIONS BASED ON USER INTENT:

A. 📝 IF THE USER ASKS TO SOLVE AN EXERCISE, ASSIGNMENT, OR NUMERICAL/CODE PROBLEM:
   - Identify the exact exercise number, section, and problem requested from the document.
   - Provide the complete, exhaustive, and accurate step-by-step solution for **EVERY single sub-part** (e.g., a, b, c, d, e, f, 1, 2, 3...).
   - Show all calculations, conversions, intermediate steps, logic, and final answers clearly formatted in clean Markdown tables or organized lists.
   - Double check mathematical and logical accuracy for all numerical/binary/hex/decimal conversions.

B. 📘 IF THE USER ASKS FOR CONCEPT EXPLANATIONS OR STUDY NOTES:
   - Provide comprehensive, deeply detailed, and well-structured master study notes.
   - Include Executive Overview, Step-by-Step Breakdown, Classifications, Comparison Tables, and High-Yield Takeaways.

🔒 STRICT DOCUMENT GROUNDING:
- Work strictly and exclusively from the uploaded document provided in the context below.
- Do NOT bring in unmentioned exercises or topics from other subjects.
- Explicitly cite page numbers `[Page X]` for every exercise, problem, and fact.
- If a question cannot be answered from the document, state: *"The uploaded document does not contain details about this."*

Context Excerpts from Notes:
{context}
"""


def format_docs_with_sources(docs_with_scores: List[Tuple[Document, float]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Formats retrieved documents into a structured context string for the prompt
    and returns a clean list of source metadata.
    """
    context_parts = []
    sources_info = []

    for i, (doc, score) in enumerate(docs_with_scores, 1):
        source_name = doc.metadata.get("source", "Document")
        page_num = doc.metadata.get("page", "N/A")
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
        
        context_parts.append(
            f"--- [Snippet {i}] Source: {source_name} | Page: {page_num} ---\n{doc.page_content}\n"
        )

        sources_info.append({
            "index": i,
            "source": source_name,
            "page": page_num,
            "chunk_id": chunk_id,
            "content": doc.page_content,
            "score": round(float(score), 4) if score is not None else None
        })

    return "\n\n".join(context_parts), sources_info


# High-availability Gemini model pool with automatic failover on 429/503
GEMINI_MODEL_POOL = [
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemma-4-31b-it",
    "gemini-3.6-flash"
]


def get_llm(
    provider: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2
) -> Any:
    """
    Initializes and returns the specified Chat LLM.
    """
    provider = provider.lower()

    if provider in ["gemini", "google"]:
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Google Gemini API Key is required.")
        model = (model_name or GEMINI_MODEL_POOL[0]).replace("models/", "")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=temperature
        )

    elif provider in ["openai"]:
        from langchain_openai import ChatOpenAI
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API Key is required.")
        model = model_name or "gpt-4o-mini"
        return ChatOpenAI(
            model=model,
            api_key=key,
            temperature=temperature
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_answer(
    query: str,
    retrieved_docs_with_scores: List[Tuple[Document, float]],
    llm: Any = None,
    provider: str = "gemini",
    api_key: Optional[str] = None,
    explanation_style: str = "📘 Comprehensive & In-Depth Master Notes (Recommended)"
) -> Dict[str, Any]:
    """
    Executes grounded RAG QA with automatic multi-model failover if quota is reached.
    """
    if not retrieved_docs_with_scores:
        return {
            "answer": "### 📌 Core Takeaway\nNo relevant context was found in your uploaded notes for this question.",
            "sources": []
        }

    context_str, sources = format_docs_with_sources(retrieved_docs_with_scores)
    style_guide = f"\nUser's Preferred Tone & Style: {explanation_style}\nEnsure complete accuracy and clear formatting."

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT + style_guide),
        ("human", "Question: {question}")
    ])

    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    # If provider is Google Gemini, use model pool with automatic failover
    if provider in ["gemini", "google"]:
        last_error = None
        for model in GEMINI_MODEL_POOL:
            try:
                candidate_llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=key,
                    temperature=0.2
                )
                chain = prompt | candidate_llm | StrOutputParser()
                response = chain.invoke({
                    "context": context_str,
                    "question": query
                })
                return {
                    "answer": response,
                    "sources": sources
                }
            except Exception as e:
                err_text = str(e)
                last_error = e
                if "429" in err_text or "RESOURCE_EXHAUSTED" in err_text or "503" in err_text or "UNAVAILABLE" in err_text:
                    print(f"[Model Failover] Model {model} hit rate limit / capacity. Shifting to next model in pool...")
                    continue
                else:
                    raise e
        raise last_error or RuntimeError("All Gemini models in pool were temporarily unavailable.")

    else:
        # Standard execution for other providers
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "context": context_str,
            "question": query
        })
        return {
            "answer": response,
            "sources": sources
        }
