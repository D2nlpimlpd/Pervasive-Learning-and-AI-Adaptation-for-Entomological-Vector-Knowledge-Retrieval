# streamlit_app.py

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

# Qwen3-VL 封装，在 qwen3_vl_wrapper.py 里
from qwen3_vl_wrapper import llm_model_func, vision_model_func


# ==================== 基本路径配置 ====================

BASE_DIR = Path(__file__).resolve().parent

# LightRAG / RAGAnything 的工作目录（索引、缓存都在这里）
WORKING_DIR = BASE_DIR / "rag_store"

# 你的文档路径（如有改动，修改这一行即可）
DOC_PATH = Path(
    "/root/meijie/范滋德——《中国常见蝇类检索表  第二版》——RAG.docx"
)

# 文档解析输出目录（中间文件）
PARSED_OUTPUT_DIR = BASE_DIR / "parsed_output"


# ==================== 简单分词器：避免 tiktoken 联网 ====================

class SimpleTokenizer:
    """
    一个极简 tokenizer：
    - 用空格切分文本，只用于估算 token 数、做 chunk 控制
    - 避免 LightRAG 默认用 TiktokenTokenizer 触发联网下载 BPE 文件
    """

    def encode(self, text: str):
        if not text:
            return []
        tokens = text.split()
        # 返回“伪 token id 列表”，长度对即可
        return list(range(len(tokens)))

    def decode(self, token_ids):
        # 一般用不到 decode，这里简单返回空串
        return ""

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())


def get_tokenizer() -> SimpleTokenizer:
    """
    构建（或复用）一个全局 SimpleTokenizer。
    放在 session_state 里避免重复创建。
    """
    if "simple_tokenizer" not in st.session_state:
        st.session_state["simple_tokenizer"] = SimpleTokenizer()
    return st.session_state["simple_tokenizer"]





# ==================== RAGAnswer 清洗 ====================

def clean_rag_answer(raw: str) -> str:
    """
    清洗 RAGAnything 的原始回答：
    - 去掉“文档名：… / 问答区域 / 你：… / RAG:” 这些头信息
    - 只保留真正的回答内容
    """
    if not isinstance(raw, str):
        return str(raw)

    text = raw.strip()

    # 优先按 "RAG:" / "RAG：" 切分，只取后半部分
    m = re.search(r"RAG[:：]\s*", text)
    if m:
        return text[m.end():].strip()

    # 如果没有 RAG:，逐行过滤掉头部信息
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        l = line.strip()
        if l.startswith("文档名"):
            continue
        if l.startswith("问答区域"):
            continue
        if l.startswith("你：") or l.startswith("你:"):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or text


# ==================== 向量模型：BGE-M3（异步 EmbeddingFunc） ====================

def build_embedding_func() -> EmbeddingFunc:
    """
    构建向量模型的 EmbeddingFunc（异步版本），
    供 LightRAG 和 RAGAnything 共用。
    """
    embed_model_name = "BAAI/bge-m3"
    st.write(f"[build_rag] loading embedding model: {embed_model_name}")

    # 多卡时，把 BGE-M3 放到第 2 张卡 (cuda:1)，避免和 Qwen3-VL 抢同一块显存
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = "cuda:1"
        else:
            device = "cuda:0"
    else:
        device = "cpu"

    st.write(f"[build_rag] embedding model device: {device}")
    embed_model = SentenceTransformer(embed_model_name, device=device)

    # 异步版本：内部用线程池跑同步 encode
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
        embedding_dim=1024,     # BGE-M3 的维度
        max_token_size=512,
    )
    return embedding_func


# ==================== 构建 / 加载 RAGAnything ====================

def build_rag_for_rebuild() -> RAGAnything:
    """
    用 RAGAnythingConfig 的方式构建 RAGAnything，
    适合第一次/需要重建索引时使用（会调用 process_document_complete）。
    """
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    config = RAGAnythingConfig(
        working_dir=str(WORKING_DIR),
        # 使用 mineru 解析器（如需改回 docling，把这里改成 "docling"）
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,     # 启用图像处理
        enable_table_processing=True,     # 表格处理
        enable_equation_processing=True,  # 公式处理
    )

    embedding_func = build_embedding_func()
    _ = get_tokenizer()  # 目前 RAGAnything 构造函数不接收 tokenizer，这里只是提前构建，避免后面首次用时卡顿

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,        # 文本 LLM：Qwen3-VL
        vision_model_func=vision_model_func,  # 图像 VLM：Qwen3-VL
        embedding_func=embedding_func,
        # ⚠ 注意：这里不能传 tokenizer=...，否则会 TypeError
    )

    return rag


async def build_rag_from_existing() -> RAGAnything:
    """
    只加载已有的 LightRAG 存储，不重新处理文档/多模态。
    - 前提：WORKING_DIR 里已经有之前构建好的索引
    """
    if not WORKING_DIR.exists() or not any(WORKING_DIR.iterdir()):
        raise RuntimeError(
            f"工作目录 {WORKING_DIR} 里没有现成的 LightRAG 存储，"
            f"请先在命令行或网页里重建一次索引。"
        )

    st.write(f"[build_rag_from_existing] 加载已有 LightRAG 存储：{WORKING_DIR}")

    embedding_func = build_embedding_func()
    tokenizer = get_tokenizer()

    # 直接创建 LightRAG 实例，指向已有 working_dir
    lightrag_instance = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        tokenizer=tokenizer,   # 显式传入 SimpleTokenizer，避免 tiktoken 联网
    )

    # 关键：只初始化/加载已有存储，不再跑文档处理流水线
    await lightrag_instance.initialize_storages()

    # 用已有 LightRAG 实例构造 RAGAnything
    rag = RAGAnything(
        lightrag=lightrag_instance,           # 直接使用已有的 LightRAG
        vision_model_func=vision_model_func,  # 继续使用 Qwen3-VL 做图像增强（按需）
        # working_dir / llm_model_func / embedding_func / tokenizer 都从 lightrag_instance 继承
    )

    return rag


# ==================== 索引构建流程 ====================

async def build_index(rag: RAGAnything):
    """
    调用 RAGAnything 的完整文档处理：
    - 解析 docx
    - 文本切块 + 实体/关系抽取
    - 多模态（图片）处理
    结果写入 WORKING_DIR（rag_store）。
    """
    st.write(">>> 开始构建索引（第一次会比较慢）...")

    if not DOC_PATH.exists():
        raise FileNotFoundError(f"找不到文档：{DOC_PATH}")

    PARSED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    await rag.process_document_complete(
        file_path=str(DOC_PATH),
        output_dir=str(PARSED_OUTPUT_DIR),
        parse_method="auto",
    )

    st.write(">>> 索引构建完成")


# ==================== Streamlit 状态控制 ====================

def init_rag_first_time():
    """
    页面第一次运行时，优先尝试加载已有索引；
    如果没有，就自动构建一次。
    """
    try:
        with st.spinner("正在从已有存储加载索引（不重建多模态）..."):
            rag = run_async(build_rag_from_existing())
            st.session_state["rag"] = rag
            st.session_state["rag_mode"] = "loaded"
            st.success("已从现有 rag_store 加载索引。")
    except RuntimeError:
        st.warning("未找到已有索引，将创建新索引（包含多模态处理），请稍候...")
        with st.spinner("正在构建索引（第一次会比较慢）..."):
            rag = build_rag_for_rebuild()
            run_async(build_index(rag))
            st.session_state["rag"] = rag
            st.session_state["rag_mode"] = "rebuilt"
            st.success("索引已构建完成。")


# ==================== Streamlit 页面逻辑 ====================

def main():
    st.set_page_config(page_title="蝇类图文 RAG 问答", layout="wide")

    st.title("中国常见蝇类 · 图文 RAG 问答")
    st.markdown(
        "基于 **RAGAnything + LightRAG + Qwen3-VL + BGE-M3** 的本地知识库问答。\n\n"
        "文档：范滋德——《中国常见蝇类检索表  第二版》"
    )

    # -------- 初始化 RAG 引擎 --------
    if "rag" not in st.session_state:
        init_rag_first_time()

    # 初始化历史和清空标记
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "clear_question" not in st.session_state:
        st.session_state["clear_question"] = False

    # 如果上一轮设置了 clear_question，这里在 text_input 创建之前清空它
    if st.session_state.get("clear_question", False):
        st.session_state["question_input"] = ""
        st.session_state["clear_question"] = False

    # -------- 左侧控制区 --------
    with st.sidebar:
        st.header("索引 & 系统控制")

        # 问题模式切换：正向检索 / 反向识别
        mode_choice = st.radio(
            "问题模式",
            [
                "正向检索（已知名称 → 检索路径与属征说明）",
                "反向识别（给出特征 → 推测可能蝇类，给出概率）",
            ],
            index=0,
        )

        # 清空 LLM 答案缓存（避免命中旧的短回答）
        if st.button("清空 LLM 答案缓存"):
            if "rag" in st.session_state:
                rag: RAGAnything = st.session_state["rag"]
                try:
                    # 只清除 LLM 回答缓存，避免“Query cache hit”总是复用旧结果
                    run_async(rag.aclear_cache(modes=["llm_response"]))
                    st.success("已清空 LLM 答案缓存，下次提问会重新生成。")
                except Exception as e:
                    st.error(f"清空缓存失败：{e}")
            else:
                st.warning("RAG 引擎尚未初始化，无法清理缓存。")

        # 显示当前索引模式
        if "rag_mode" in st.session_state:
            mode = st.session_state["rag_mode"]
            if mode == "loaded":
                st.info("当前模式：使用已有索引（不重建多模态）")
            elif mode == "rebuilt":
                st.info("当前模式：索引刚重新构建完成")
        else:
            st.info("正在初始化 RAG 引擎...")

        # 清空对话
        if st.button("清空对话历史"):
            st.session_state["history"] = []
            st.success("已清空对话。")

    # -------- 对话区域 --------
    st.subheader("问答区域")

    # 显示历史对话
    for msg in st.session_state["history"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(f"**你：** {content}")
        else:
            st.markdown(f"**RAG：** {content}")

    # 问题输入
    question = st.text_input(
        "请输入你的问题（例如：齿股蝇属有哪些属征？或：体长约 ×× 毫米、胸背××花纹的是哪种蝇？）",
        key="question_input",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_clicked = st.button("发送")

    if ask_clicked and question.strip():
        if "rag" not in st.session_state:
            st.error("RAG 引擎尚未初始化，请稍后重试。")
        else:
            rag: RAGAnything = st.session_state["rag"]
            st.session_state["history"].append({"role": "user", "content": question})

            # 根据模式在问题前加上 [MODE:FORWARD] / [MODE:REVERSE] 标记
            if "反向识别" in mode_choice:
                wrapped_question = f"[MODE:REVERSE]\n{question}"
            else:
                wrapped_question = f"[MODE:FORWARD]\n{question}"

            with st.spinner("模型思考中，请稍候..."):
                try:
                    # 这里显式关闭 VLM 增强，只用文本检索
                    raw_answer = run_async(
                        rag.aquery(
                            wrapped_question,
                            mode="hybrid",
                            vlm_enhanced=False,
                        )
                    )
                    answer = clean_rag_answer(raw_answer)
                except Exception as e:
                    answer = f"查询时发生错误：{e}"

            st.session_state["history"].append(
                {"role": "assistant", "content": answer}
            )

            # 不直接改 question_input，而是打一个“需要清空”的标记，
            # 让下一轮在 text_input 创建之前去清空
            st.session_state["clear_question"] = True

            st.rerun()  # 刷新页面以显示新对话


if __name__ == "__main__":
    main()