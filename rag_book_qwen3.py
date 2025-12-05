import os
import sys
import asyncio
from pathlib import Path
from typing import List

import torch
from sentence_transformers import SentenceTransformer

from raganything import RAGAnything, RAGAnythingConfig
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

# 同时导入文本 LLM 和 VLM（你原来已有的包装）
from qwen3_vl_wrapper import llm_model_func, vision_model_func


BASE_DIR = Path(__file__).resolve().parent
# 用来保存 LightRAG 的索引（你之前就是这个目录）
WORKING_DIR = BASE_DIR / "rag_store"


def build_embedding_func() -> EmbeddingFunc:
    """
    构建向量模型的 EmbeddingFunc（异步版本），
    供 LightRAG 和 RAGAnything 共用。
    """
    embed_model_name = "BAAI/bge-m3"
    print(f"[build_rag] loading embedding model: {embed_model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[build_rag] embedding model device: {device}")
    embed_model = SentenceTransformer(embed_model_name, device=device)

    # ⭐ 异步版本，内部放到线程池跑同步 encode
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
        embedding_dim=1024,
        max_token_size=512,
    )
    return embedding_func


def build_rag_for_rebuild() -> RAGAnything:
    """
    用 RAGAnythingConfig 的方式构建 RAGAnything，
    适合第一次/需要重建索引时使用（会调用 process_document_complete）。
    """
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    config = RAGAnythingConfig(
        working_dir=str(WORKING_DIR),
        parser="docling",
        parse_method="auto",
        enable_image_processing=True,    # 启用图像管线
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    embedding_func = build_embedding_func()

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,       # 文本 LLM：Qwen3-VL
        vision_model_func=vision_model_func, # 图像 VLM：Qwen3-VL
        embedding_func=embedding_func,
    )

    return rag


async def build_rag_from_existing() -> RAGAnything:
    """
    只加载已有的 LightRAG 存储，不重新处理文档/多模态。
    - 前提：WORKING_DIR 里已经有之前构建好的索引（用 --rebuild 建过一次）。
    """
    if not WORKING_DIR.exists() or not any(WORKING_DIR.iterdir()):
        raise RuntimeError(
            f"工作目录 {WORKING_DIR} 里没有现成的 LightRAG 存储，"
            f"请先运行：python rag_book_qwen3.py --rebuild"
        )

    print(f"[build_rag_from_existing] 加载已有 LightRAG 存储：{WORKING_DIR}")

    embedding_func = build_embedding_func()

    # 直接创建 LightRAG 实例，指向已有 working_dir
    lightrag_instance = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )

    # ⭐ 关键：只初始化/加载已有存储，不再跑文档处理流水线
    await lightrag_instance.initialize_storages()

    # 用已有 LightRAG 实例构造 RAGAnything
    rag = RAGAnything(
        lightrag=lightrag_instance,          # 直接使用已有的 LightRAG
        vision_model_func=vision_model_func, # 继续使用 Qwen3-VL 做图像增强（可按需使用）
        # working_dir / llm_model_func / embedding_func 都从 lightrag_instance 继承
    )

    return rag


async def build_index(rag: RAGAnything):
    """
    调用 RAGAnything 的完整文档处理：
    - 解析 docx
    - 文本切块 + 实体/关系抽取
    - 多模态（图片）处理
    结果写入 WORKING_DIR（rag_store）。
    """
    print(">>> 开始构建索引（第一次会比较慢）...")

    doc_path = Path(r"/root/meijie/范滋德——《中国常见蝇类检索表  第二版》——RAG.docx")
    if not doc_path.exists():
        raise FileNotFoundError(f"找不到文档：{doc_path}")

    output_dir = BASE_DIR / "parsed_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    await rag.process_document_complete(
        file_path=str(doc_path),
        output_dir=str(output_dir),
        parse_method="auto",
    )

    print(">>> 索引构建完成")


async def ask_question(rag: RAGAnything):
    """
    交互式问答循环。
    """
    print(">>> 现在可以开始就文档内容提问了（直接回车退出）。")

    while True:
        try:
            question = input("\n请输入你的问题（回车结束）：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n>>> 结束问答。")
            break

        if not question:
            print(">>> 空问题，结束问答。")
            break

        print(f"\n>>> 正在查询：{question}")

        # ⭐ 这里显式关闭 VLM 增强，暂时只用文本检索，避免当前版本的 bug
        answer = await rag.aquery(
            question,
            mode="hybrid",
            vlm_enhanced=False,
        )

        print("\n>>> RAG 答案：")
        print(answer)


async def main():
    """
    两种启动模式：
    1. python rag_book_qwen3.py --rebuild
       - 解析文档 + 文本 & 图谱 & 多模态建索引
       - 结果写入 /root/meijie/rag_store

    2. python rag_book_qwen3.py
       - 只从已有的 /root/meijie/rag_store 加载 LightRAG 索引
       - 不再重新处理文档/多模态
    """
    if "--rebuild" in sys.argv:
        # 模式 1：重建索引（会重新跑文本 + 图谱 + 多模态）
        rag = build_rag_for_rebuild()
        await build_index(rag)
    else:
        # 模式 2：只加载已有 LightRAG，不再处理文档/多模态
        rag = await build_rag_from_existing()

    await ask_question(rag)


if __name__ == "__main__":
    asyncio.run(main())