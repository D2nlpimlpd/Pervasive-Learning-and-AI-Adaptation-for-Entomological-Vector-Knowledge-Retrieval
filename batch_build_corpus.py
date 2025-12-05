# batch_build_corpus.py

import asyncio
from typing import List

import torch
from sentence_transformers import SentenceTransformer

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

# 使用你已经写好的 Qwen3-VL 封装
from qwen3_vl_wrapper import llm_model_func, vision_model_func


# ===== 本地嵌入模型配置 =====
# 你已经使用的是 BAAI/bge-base-zh-v1.5
EMBED_MODEL_DIR = "/root/.cache/modelscope/hub/models/BAAI/bge-base-zh-v1.5"

# bge-base-zh-v1.5 的维度是 768
EMBED_DIM = 768

print(f"[embedding] loading sentence-transformers model from {EMBED_MODEL_DIR} ...")

embed_device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = SentenceTransformer(EMBED_MODEL_DIR, device=embed_device)

print(f"[embedding] model loaded on {embed_device}.")


def _encode_batch(texts: List[str]) -> List[List[float]]:
    """
    真正做向量计算的同步函数（在线程池里跑）
    """
    if not texts:
        return []

    emb = embed_model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,   # 方便后续用余弦相似度
        show_progress_bar=False,
    )
    return emb.tolist()


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    提供给 LightRAG 的异步嵌入函数。
    LightRAG 会对这个函数结果执行 `await`，所以必须是 async def。
    内部用 run_in_executor 把同步的 encode 扔到线程池里跑，避免阻塞事件循环。
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _encode_batch, texts)


async def main():
    # 1. 配置 RAGAnything —— 使用 mineru
    config = RAGAnythingConfig(
        working_dir="./rag_storage",   # 知识库存储目录，和 Streamlit 里保持一致
        parser="mineru",               # 使用 mineru 解析器
        parse_method="auto",

        # 根据你需求：mineru 通常配合多模态处理
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # 2. 使用本地异步嵌入函数
    embedding_func = EmbeddingFunc(
        embedding_dim=EMBED_DIM,
        max_token_size=8192,
        func=embed_texts,             # 注意：这里现在是 async 函数
    )

    # 3. 实例化 RAGAnything，绑定你自己的 llm_model_func / vision_model_func
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,          # 来自 qwen3_vl_wrapper.py
        vision_model_func=vision_model_func,    # 来自 qwen3_vl_wrapper.py
        embedding_func=embedding_func,
    )

    # 4. 批量处理整个文件夹
    await rag.process_folder_complete(
        folder_path="/root/meijie/dataset",     # 你的数据集目录
        output_dir="./output",                  # 解析结果输出目录
        file_extensions=[
            ".doc", ".docx",
            ".pdf",
        ],
        recursive=True,
        max_workers=1,                          # 先用 1，方便看日志；确认没问题后再调大
    )

    print("✅ 文件夹批量处理完成！")


if __name__ == "__main__":
    asyncio.run(main())