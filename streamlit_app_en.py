# streamlit_app_en.py

import re
import asyncio
import threading
from pathlib import Path
from typing import List

import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from async_utils import run_async
from raganything import RAGAnything, RAGAnythingConfig
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

# Qwen3-VL wrapper, located in qwen3_vl_wrapper.py
from qwen3_vl_wrapper import llm_model_func, vision_model_func


# ==================== Basic Path Configuration ====================

BASE_DIR = Path(__file__).resolve().parent

# Working directory for LightRAG / RAGAnything (indexes and caches are stored here)
WORKING_DIR = BASE_DIR / "rag_store"

# Your document path (Modify this line if the path changes)
DOC_PATH = Path(
    "/root/meijie/范滋德——《中国常见蝇类检索表  第二版》——RAG.docx"
)

# Document parsing output directory (intermediate files)
PARSED_OUTPUT_DIR = BASE_DIR / "parsed_output"


# ==================== Simple Tokenizer: Avoid Tiktoken Networking ====================

class SimpleTokenizer:
    """
    A minimalist tokenizer:
    - Splits text by spaces, used only for estimating token counts and chunk control.
    - Prevents LightRAG from using the default TiktokenTokenizer which triggers 
      network downloads for BPE files (avoiding connection issues).
    """

    def encode(self, text: str):
        if not text:
            return []
        tokens = text.split()
        # Return a list of "pseudo token ids", length is what matters
        return list(range(len(tokens)))

    def decode(self, token_ids):
        # Usually not needed for decoding, returning empty string
        return ""

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())


def get_tokenizer() -> SimpleTokenizer:
    """
    Construct (or reuse) a global SimpleTokenizer.
    Stored in session_state to avoid duplicate creation.
    """
    if "simple_tokenizer" not in st.session_state:
        st.session_state["simple_tokenizer"] = SimpleTokenizer()
    return st.session_state["simple_tokenizer"]


# ==================== RAGAnswer Cleaning ====================

def clean_rag_answer(raw: str) -> str:
    """
    Clean the raw answer from RAGAnything:
    - Remove header info like "Document Name:...", "Q&A Area", "You:...", "RAG:".
    - Keep only the actual answer content.
    """
    if not isinstance(raw, str):
        return str(raw)

    text = raw.strip()

    # Prioritize splitting by "RAG:" / "RAG：", keeping only the latter part
    m = re.search(r"RAG[:：]\s*", text)
    if m:
        return text[m.end():].strip()

    # If no "RAG:" found, filter out header lines line by line
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        l = line.strip()
        # Check for both Chinese and English headers to be safe
        if l.startswith("文档名") or l.startswith("Document Name") or l.startswith("Filename"):
            continue
        if l.startswith("问答区域") or l.startswith("Q&A Area"):
            continue
        if l.startswith("你：") or l.startswith("你:") or l.startswith("You:"):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or text


# ==================== Embedding Model: BGE-M3 (Async EmbeddingFunc) ====================

def build_embedding_func() -> EmbeddingFunc:
    """
    Construct EmbeddingFunc (Async version) for the vector model,
    shared by both LightRAG and RAGAnything.
    """
    embed_model_name = "BAAI/bge-m3"
    st.write(f"[build_rag] loading embedding model: {embed_model_name}")

    # For multi-GPU setup: place BGE-M3 on card 2 (cuda:1) to avoid VRAM conflict with Qwen3-VL
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = "cuda:1"
        else:
            device = "cuda:0"
    else:
        device = "cpu"

    st.write(f"[build_rag] embedding model device: {device}")
    embed_model = SentenceTransformer(embed_model_name, device=device)

    # Async version: runs synchronous encode inside a thread pool
    async def embed_texts(texts: List[str]) -> List[List[float]]:
        def _encode(batch: List[str]) -> List[List[float]]:
            embeddings = embed_model.encode(
                batch,
                batch_size=32,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embeddings.tolist()

        return await asyncio.to_thread(_encode, texts)

    embedding_func = EmbeddingFunc(
        func=embed_texts,
        embedding_dim=1024,     # Dimension for BGE-M3
        max_token_size=512,
    )
    return embedding_func


# ==================== Build / Load RAGAnything ====================

def build_rag_for_rebuild() -> RAGAnything:
    """
    Build RAGAnything using RAGAnythingConfig.
    Suitable for the first run or when rebuilding the index 
    (will invoke process_document_complete).
    """
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    config = RAGAnythingConfig(
        working_dir=str(WORKING_DIR),
        # Use mineru parser (change to "docling" if needed)
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,     # Enable image processing
        enable_table_processing=True,     # Enable table processing
        enable_equation_processing=True,  # Enable equation processing
    )

    embedding_func = build_embedding_func()
    _ = get_tokenizer()  # Pre-build tokenizer to avoid lag later, though RAGAnything constructor doesn't take it

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,        # Text LLM: Qwen3-VL
        vision_model_func=vision_model_func,  # Vision VLM: Qwen3-VL
        embedding_func=embedding_func,
        # Note: Do not pass tokenizer=... here, or it will raise a TypeError
    )

    return rag


async def build_rag_from_existing() -> RAGAnything:
    """
    Load from existing LightRAG storage only, without reprocessing documents/multimodal data.
    - Prerequisite: WORKING_DIR must contain a previously built index.
    """
    if not WORKING_DIR.exists() or not any(WORKING_DIR.iterdir()):
        raise RuntimeError(
            f"No existing LightRAG storage found in {WORKING_DIR}. "
            f"Please rebuild the index via command line or UI first."
        )

    st.write(f"[build_rag_from_existing] Loading existing LightRAG storage: {WORKING_DIR}")

    embedding_func = build_embedding_func()
    tokenizer = get_tokenizer()

    # Create LightRAG instance pointing to existing working_dir
    lightrag_instance = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        tokenizer=tokenizer,   # Explicitly pass SimpleTokenizer to avoid tiktoken networking
    )

    # Key: Initialize/Load existing storage only, do not run the document processing pipeline
    await lightrag_instance.initialize_storages()

    # Construct RAGAnything using the existing LightRAG instance
    rag = RAGAnything(
        lightrag=lightrag_instance,           # Use the existing LightRAG
        vision_model_func=vision_model_func,  # Continue using Qwen3-VL for image enhancement (if needed)
        # working_dir / llm_model_func / embedding_func / tokenizer are inherited from lightrag_instance
    )

    return rag


# ==================== Index Construction Workflow ====================

async def build_index(rag: RAGAnything):
    """
    Invoke complete document processing in RAGAnything:
    - Parse docx
    - Text chunking + Entity/Relationship extraction
    - Multimodal (Image) processing
    Results are written to WORKING_DIR (rag_store).
    """
    st.write(">>> Starting index construction (this may take a while for the first time)...")

    if not DOC_PATH.exists():
        raise FileNotFoundError(f"Document not found: {DOC_PATH}")

    PARSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    await rag.process_document_complete(
        file_path=str(DOC_PATH),
        output_dir=str(PARSED_OUTPUT_DIR),
        parse_method="auto",
    )

    st.write(">>> Index construction completed.")


# ==================== Streamlit State Control ====================

def init_rag_first_time():
    """
    On first run, prioritize loading existing index;
    If none exists, automatically build a new one.
    """
    try:
        with st.spinner("Loading index from existing storage (skipping multimodal rebuild)..."):
            rag = run_async(build_rag_from_existing())
            st.session_state["rag"] = rag
            st.session_state["rag_mode"] = "loaded"
            st.success("Index loaded from existing rag_store.")
    except RuntimeError:
        st.warning("No existing index found. Creating new index (includes multimodal processing), please wait...")
        with st.spinner("Building index (this takes time)..."):
            rag = build_rag_for_rebuild()
            run_async(build_index(rag))
            st.session_state["rag"] = rag
            st.session_state["rag_mode"] = "rebuilt"
            st.success("Index construction completed.")


# ==================== Streamlit Page Logic ====================

def main():
    st.set_page_config(page_title="Fly Species Image-Text RAG Q&A", layout="wide")

    st.title("Common Flies in China · Image-Text RAG Q&A")
    st.markdown(
        "Local Knowledge Base Q&A based on **RAGAnything + LightRAG + Qwen3-VL + BGE-M3**.\n\n"
        "Document: Fan Zide — 'Key to Common Flies in China (2nd Edition)'"
    )

    # -------- Initialize RAG Engine --------
    if "rag" not in st.session_state:
        init_rag_first_time()

    # Initialize history and clear flag
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "clear_question" not in st.session_state:
        st.session_state["clear_question"] = False

    # If clear_question was set in the previous run, clear text_input before creation
    if st.session_state.get("clear_question", False):
        st.session_state["question_input"] = ""
        st.session_state["clear_question"] = False

    # -------- Sidebar Control Area --------
    with st.sidebar:
        st.header("Index & System Control")

        # Question Mode Switching
        mode_choice = st.radio(
            "Question Mode",
            [
                "Forward Retrieval (Known Name -> Retrieve path and genus characteristics)",
                "Reverse Identification (Provide features -> Infer possible flies with probability)",
            ],
            index=0,
        )

        # Clear LLM Cache
        if st.button("Clear LLM Answer Cache"):
            if "rag" in st.session_state:
                rag: RAGAnything = st.session_state["rag"]
                try:
                    # Only clear LLM response cache to avoid "Query cache hit" reusing old short answers
                    run_async(rag.aclear_cache(modes=["llm_response"]))
                    st.success("LLM cache cleared. Next question will be regenerated.")
                except Exception as e:
                    st.error(f"Failed to clear cache: {e}")
            else:
                st.warning("RAG Engine not initialized, cannot clear cache.")

        # Display Current Index Mode
        if "rag_mode" in st.session_state:
            mode = st.session_state["rag_mode"]
            if mode == "loaded":
                st.info("Current Mode: Using Existing Index (No Multimodal Rebuild)")
            elif mode == "rebuilt":
                st.info("Current Mode: Index Newly Built")
        else:
            st.info("Initializing RAG Engine...")

        # Clear Chat History
        if st.button("Clear Chat History"):
            st.session_state["history"] = []
            st.success("Chat history cleared.")

    # -------- Q&A Area --------
    st.subheader("Q&A Area")

    # Display History
    for msg in st.session_state["history"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**RAG:** {content}")

    # Question Input
    question = st.text_input(
        "Enter your question (e.g., What are the genus characteristics of Genus X? or: Which fly is X mm long with Y pattern on the thorax?)",
        key="question_input",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_clicked = st.button("Send")

    if ask_clicked and question.strip():
        if "rag" not in st.session_state:
            st.error("RAG Engine not initialized yet, please wait.")
        else:
            rag: RAGAnything = st.session_state["rag"]
            st.session_state["history"].append({"role": "user", "content": question})

            # Add [MODE:FORWARD] / [MODE:REVERSE] tag based on mode selection
            if "Reverse Identification" in mode_choice:
                wrapped_question = f"[MODE:REVERSE]\n{question}"
            else:
                wrapped_question = f"[MODE:FORWARD]\n{question}"

            with st.spinner("Model is thinking, please wait..."):
                try:
                    # Explicitly disable VLM enhancement here, using text retrieval only
                    raw_answer = run_async(
                        rag.aquery(
                            wrapped_question,
                            mode="hybrid",
                            vlm_enhanced=False,
                        )
                    )
                    answer = clean_rag_answer(raw_answer)
                except Exception as e:
                    answer = f"Error during query: {e}"

            st.session_state["history"].append(
                {"role": "assistant", "content": answer}
            )

            # Do not modify question_input directly; set a flag to clear it on next rerun
            st.session_state["clear_question"] = True

            st.rerun()  # Refresh page to show new conversation


if __name__ == "__main__":
    main()