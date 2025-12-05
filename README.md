基于 **RAGAnything + LightRAG + Qwen3‑VL + BGE‑M3** 的本地知识库问答 Demo，主要参考文献：

> 范滋德 — 《中国常见蝇类检索表（第二版）》

本项目可以对该文献进行解析、建立向量索引及知识图谱，并通过 RAG 架构实现问答，支持：

- 文本检索 + 图谱检索（LightRAG）
- Qwen3‑VL 文本 / 多模态大模型
- 正向检索（已知名称 → 检索路径 + 属 / 种特征）
- 反向识别（给出特征 → 推断可能的蝇类及概率）

---

## 1. 环境准备

### 1.1 Python 与操作系统

- Python ≥ 3.10（`async_utils.py` 使用了 `X | None` 语法）
- 推荐环境：
  - Linux / WSL2 / Windows
  - 有 GPU 的机器（特别是运行 Qwen3‑VL‑32B 时）
  - 至少 16 GB 内存

### 1.2 创建虚拟环境

# 以 conda 为例  
conda create -n meijie python=3.10 -y  
conda activate meijie  
1.3 安装依赖
在项目根目录创建 requirements.txt（示例）：

txt
torch
sentence-transformers
streamlit
openai>=1.0.0
pillow
raganything
lightrag
mineru
pyyaml
安装依赖：

bash
pip install -r requirements.txt
# 或
pip install torch sentence-transformers streamlit "openai>=1.0.0" pillow raganything lightrag mineru pyyaml
提示：请根据你机器的 CUDA 版本，选择合适的 torch 安装方式（官网或镜像源）。

2. 模型与后端服务
2.1 Qwen3‑VL + vLLM
本项目通过 vLLM 的 OpenAI 兼容接口调用 Qwen3‑VL，在 qwen3_vl_wrapper.py 中配置：

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
VLLM_API_KEY = "EMPTY"
VLLM_MODEL_NAME = "Qwen3-VL-32B-Instruct"
你需要：

在一台 GPU 机器上启动 vLLM 服务，加载 Qwen3-VL-32B-Instruct（或其他 Qwen3‑VL 模型）。
保证运行本项目的机器能访问到 VLLM_BASE_URL。
启动示例（仅示意）：

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-32B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768
如需远程服务，请修改 qwen3_vl_wrapper.py：

VLLM_BASE_URL = "http://your-server-ip:8000/v1"
VLLM_MODEL_NAME = "Your-Qwen3-VL-Model-Name"
2.2 向量模型 BGE‑M3
项目中统一使用 BAAI/bge-m3 作为嵌入模型，例如：

from sentence_transformers import SentenceTransformer

embed_model_name = "BAAI/bge-m3"
embed_model = SentenceTransformer(embed_model_name, device=device)
设备选择逻辑示例（streamlit_app_en.py）：

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        device = "cuda:1"
    else:
        device = "cuda:0"
else:
    device = "cpu"
多 GPU：默认放在 cuda:1，避免与 Qwen3‑VL 冲突。
单 GPU：放在 cuda:0。
无 GPU：退回 cpu（速度会明显变慢）。
3. 路径与数据配置
3.1 基本目录
代码中常见路径定义方式：

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKING_DIR = BASE_DIR / "rag_store"
PARSED_OUTPUT_DIR = BASE_DIR / "parsed_output"
rag_store/：LightRAG / RAGAnything 的索引与缓存目录（自动创建）。
parsed_output/：解析中间文件目录（自动创建）。
3.2 文档路径（重要）
当前代码中文档路径被写成 Linux 绝对路径，你需要根据自己的环境修改。

streamlit_app_en.py 中：

DOC_PATH = Path(
    "/root/meijie/范滋德——《中国常见蝇类检索表  第二版》——RAG.docx"
)
async_utils.py 中（命令行构建索引用）：

doc_path = Path(r"/root/meijie/范滋德——《中国常见蝇类检索表  第二版》——RAG.docx")
建议统一改为相对路径（推荐）：

# streamlit_app_en.py
DOC_PATH = BASE_DIR / "input" / "范滋德——《中国常见蝇类检索表  第二版》——RAG.docx"

# async_utils.py
doc_path = BASE_DIR / "input" / "范滋德——《中国常见蝇类检索表  第二版》——RAG.docx"
然后在项目根目录创建 input/ 文件夹，将 docx 放入其中：

meijie/
  ├─ input/
  │   └─ 范滋德——《中国常见蝇类检索表  第二版》——RAG.docx
  ├─ streamlit_app_en.py
  ├─ async_utils.py
  ├─ qwen3_vl_wrapper.py
  ...
4. 主要代码文件说明
4.1 streamlit_app_en.py（Streamlit Web 应用）
英文 Web UI 入口文件，核心逻辑包括：

4.1.1 简易 Tokenizer：避免 tiktoken 联网
class SimpleTokenizer:
    def encode(self, text: str):
        if not text:
            return []
        tokens = text.split()
        return list(range(len(tokens)))

    def decode(self, token_ids):
        return ""

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text.split())


def get_tokenizer() -> SimpleTokenizer:
    if "simple_tokenizer" not in st.session_state:
        st.session_state["simple_tokenizer"] = SimpleTokenizer()
    return st.session_state["simple_tokenizer"]
在「从已有索引加载」路径中显式传入：

tokenizer = get_tokenizer()
lightrag_instance = LightRAG(
    working_dir=str(WORKING_DIR),
    llm_model_func=llm_model_func,
    embedding_func=embedding_func,
    tokenizer=tokenizer,
)
4.1.2 嵌入模型构建（BGE‑M3）

def build_embedding_func() -> EmbeddingFunc:
    embed_model_name = "BAAI/bge-m3"
    st.write(f"[build_rag] loading embedding model: {embed_model_name}")

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = "cuda:1"
        else:
            device = "cuda:0"
    else:
        device = "cpu"

    st.write(f"[build_rag] embedding model device: {device}")
    embed_model = SentenceTransformer(embed_model_name, device=device)

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
4.1.3 RAG 构建 / 加载
重建索引：

def build_rag_for_rebuild() -> RAGAnything:
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    config = RAGAnythingConfig(
        working_dir=str(WORKING_DIR),
        parser="mineru",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    embedding_func = build_embedding_func()
    _ = get_tokenizer()

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )
    return rag
从已有索引加载：

async def build_rag_from_existing() -> RAGAnything:
    if not WORKING_DIR.exists() or not any(WORKING_DIR.iterdir()):
        raise RuntimeError(
            f"No existing LightRAG storage found in {WORKING_DIR}. "
            f"Please rebuild the index via command line or UI first."
        )

    st.write(f"[build_rag_from_existing] Loading existing LightRAG storage: {WORKING_DIR}")

    embedding_func = build_embedding_func()
    tokenizer = get_tokenizer()

    lightrag_instance = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        tokenizer=tokenizer,
    )

    await lightrag_instance.initialize_storages()

    rag = RAGAnything(
        lightrag=lightrag_instance,
        vision_model_func=vision_model_func,
    )
    return rag
4.1.4 索引构建流程

async def build_index(rag: RAGAnything):
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
4.1.5 首次初始化逻辑

def init_rag_first_time():
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
4.1.6 问答交互
模式选择（正向 / 反向）：

mode_choice = st.radio(
    "Question Mode",
    [
        "Forward Retrieval (Known Name -> Retrieve path and genus characteristics)",
        "Reverse Identification (Provide features -> Infer possible flies with probability)",
    ],
    index=0,
)
包装问题并调用 RAG：

if ask_clicked and question.strip():
    rag: RAGAnything = st.session_state["rag"]
    st.session_state["history"].append({"role": "user", "content": question})

    if "Reverse Identification" in mode_choice:
        wrapped_question = f"[MODE:REVERSE]\n{question}"
    else:
        wrapped_question = f"[MODE:FORWARD]\n{question}"

    with st.spinner("Model is thinking, please wait..."):
        raw_answer = run_async(
            rag.aquery(
                wrapped_question,
                mode="hybrid",
                vlm_enhanced=False,
            )
        )
        answer = clean_rag_answer(raw_answer)

    st.session_state["history"].append(
        {"role": "assistant", "content": answer}
    )

    st.session_state["clear_question"] = True
    st.rerun()
4.2 qwen3_vl_wrapper.py（Qwen3‑VL 封装）
该文件通过 vLLM 的 OpenAI 接口提供两个入口函数：

文本 LLM：llm_model_func
文本 + 图像 VLM：vision_model_func
关键配置：

from openai import OpenAI

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
VLLM_API_KEY = "EMPTY"
VLLM_MODEL_NAME = "Qwen3-VL-32B-Instruct"

client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)
语言控制（强制回答语言与用户问题语言一致）：

LANGUAGE_INSTRUCTION = (
    "CRITICAL OUTPUT RULE: You must answer in the SAME language as the user's current question. "
    "If the user asks in English, your entire response (including reasoning, descriptions, and conclusions) MUST be in English. "
    "Note: The provided document context is in Chinese. You must translate the relevant information into English when answering an English question. "
    "如果用户用中文提问，请务必用中文回答。"
)
构造完整提示词（模式 + 领域指令 + 上下文 + 用户问题）：

def _build_full_prompt(
    user_prompt: str,
    system_prompt: str = "",
    query_mode: str = "forward",
) -> str:
    if query_mode == "reverse":
        mode_instruction = REVERSE_MODE_INSTRUCTION.strip()
    else:
        mode_instruction = FORWARD_MODE_INSTRUCTION.strip()

    base_prefix_parts: List[str] = [mode_instruction, DOMAIN_INSTRUCTION.strip()]
    base_prefix = "\n\n".join([p for p in base_prefix_parts if p]).strip()

    sys_dyn = (system_prompt or "").strip()
    user = (user_prompt or "").strip()

    parts: List[str] = []
    if base_prefix:
        parts.append(base_prefix)
    if sys_dyn:
        parts.append(sys_dyn)
    if user:
        parts.append(user)

    return "\n\n".join(parts).strip()
供 RAG 调用的文本接口：

async def llm_model_func(
    prompt: str,
    system_prompt: str = "",
    history_messages=None,
    **kwargs,
) -> str:
    query_mode, stripped_prompt = _parse_mode_from_prompt(prompt)

    full_prompt = _build_full_prompt(
        stripped_prompt,
        system_prompt=system_prompt or "",
        query_mode=query_mode,
    )

    gen_args: Dict[str, Any] = {}
    for k in ["temperature", "top_p"]:
        if k in kwargs:
            gen_args[k] = kwargs[k]

    return qwen3_vl_chat_text(
        text=full_prompt,
        system_prompt="",
        max_new_tokens=MAX_NEW_TOKENS_DEFAULT,
        **gen_args,
    )
4.3 async_utils.py（异步工具 + 命令行模式）
4.3.1 run_async：在后台线程跑 asyncio

import asyncio
import threading
from typing import Awaitable, Any

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _loop_runner(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            _loop_thread = threading.Thread(
                target=_loop_runner,
                args=(_loop,),
                daemon=True,
            )
            _loop_thread.start()
        return _loop


def run_async(coro: Awaitable[Any]) -> Any:
    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()
4.3.2 命令行模式：构建 / 加载索引 + 终端问答
（结构与 streamlit_app_en.py 类似，这里只展示核心部分）

import sys
import asyncio
from pathlib import Path
from typing import List

import torch
from sentence_transformers import SentenceTransformer

from raganything import RAGAnything, RAGAnythingConfig
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

from qwen3_vl_wrapper import llm_model_func, vision_model_func

BASE_DIR = Path(__file__).resolve().parent
WORKING_DIR = BASE_DIR / "rag_store"
构建嵌入模型：

def build_embedding_func() -> EmbeddingFunc:
    embed_model_name = "BAAI/bge-m3"
    print(f"[build_rag] loading embedding model: {embed_model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[build_rag] embedding model device: {device}")
    embed_model = SentenceTransformer(embed_model_name, device=device)

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
构建 / 加载 RAG：

def build_rag_for_rebuild() -> RAGAnything:
    WORKING_DIR.mkdir(parents=True, exist_ok=True)

    config = RAGAnythingConfig(
        working_dir=str(WORKING_DIR),
        parser="docling",
        parse_method="auto",
        enable_image_processing=True,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    embedding_func = build_embedding_func()

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    return rag

async def build_rag_from_existing() -> RAGAnything:
    if not WORKING_DIR.exists() or not any(WORKING_DIR.iterdir()):
        raise RuntimeError(
            f"工作目录 {WORKING_DIR} 里没有现成的 LightRAG 存储，"
            f"请先运行：python async_utils.py --rebuild"
        )

    print(f"[build_rag_from_existing] 加载已有 LightRAG 存储：{WORKING_DIR}")

    embedding_func = build_embedding_func()

    lightrag_instance = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )

    await lightrag_instance.initialize_storages()

    rag = RAGAnything(
        lightrag=lightrag_instance,
        vision_model_func=vision_model_func,
    )

    return rag
构建索引与问答：

async def build_index(rag: RAGAnything):
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

        answer = await rag.aquery(
            question,
            mode="hybrid",
            vlm_enhanced=False,
        )

        print("\n>>> RAG 答案：")
        print(answer)
入口：

async def main():
    if "--rebuild" in sys.argv:
        rag = build_rag_for_rebuild()
        await build_index(rag)
    else:
        rag = await build_rag_from_existing()

    await ask_question(rag)


if __name__ == "__main__":
    asyncio.run(main())
5. 运行方式
5.1 命令行模式（可选）
在项目根目录下：


# 重建索引（解析文档 + 建索引）
python async_utils.py --rebuild

# 仅加载已有 rag_store 后问答
python async_utils.py
5.2 启动 Streamlit Web 应用
在项目根目录下：


streamlit run streamlit_app_en.py
终端会显示访问 URL（通常为 http://localhost:8501），浏览器打开即可。

6. 常见问题
6.1 找不到文档（FileNotFoundError）
错误示例：

FileNotFoundError: Document not found: /root/meijie/...
FileNotFoundError: 找不到文档：/root/meijie/...
解决方法：

确认文档路径正确存在；
将 streamlit_app_en.py 中的 DOC_PATH、async_utils.py 中的 doc_path 修改为你的真实路径；
建议使用 BASE_DIR / "input" / ... 的相对路径，并将 docx 放到 input/ 目录。
6.2 CUDA / 显存问题
若遇到 CUDA 初始化失败或显存不足：

确认已安装支持 GPU 的 torch；
确认 vLLM 所在机器有足够显存容纳 Qwen3‑VL；
必要时可以在构建嵌入模型时强制指定设备，例如：

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
6.3 无法连接 vLLM / OpenAI 接口
错误示例：

ConnectionError: Failed to establish a new connection: [Errno 111] Connection refused
排查步骤：

确认 vLLM 服务是否已启动且无报错；
检查 VLLM_BASE_URL 地址和端口是否匹配；
如跨机器访问，确认网络连通、防火墙配置无误。

7. 建议的项目结构

meijie/
  ├─ streamlit_app_en.py      # Web UI 入口（英文界面）
  ├─ qwen3_vl_wrapper.py      # Qwen3-VL + vLLM 封装
  ├─ async_utils.py           # run_async 工具 + 命令行模式
  ├─ input/                   # 放原始 docx 文档
  ├─ rag_store/               # 索引与缓存（运行后自动生成）
  ├─ parsed_output/           # 文档解析中间文件（运行后自动生成）
  ├─ requirements.txt
  └─ README.md
