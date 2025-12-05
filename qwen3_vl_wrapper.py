# qwen3_vl_wrapper.py

import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from openai import OpenAI  # pip install "openai>=1.0.0"

# ===== vLLM / OpenAI API 配置 =====

# vLLM OpenAI 服务器地址（根据你启动时的 --host/--port 修改）
VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
# vLLM 默认不会校验 api_key，可随便填一个占位值
VLLM_API_KEY = "EMPTY"

# vLLM 中注册的模型名称（通常就是原始模型名 / --served-model-name）
VLLM_MODEL_NAME = "Qwen3-VL-32B-Instruct"

client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
)

# ===== 上下文长度 & 生成长度控制 =====

VLLM_MAX_CONTEXT_TOKENS = 32768          # 请与 vLLM --max-model-len 保持一致
RESERVED_TOKENS_FOR_OUTPUT = 1024        # 为输出预留的 token 数

# 默认最大生成长度（新 token 数量）
MAX_NEW_TOKENS_DEFAULT = min(1024, RESERVED_TOKENS_FOR_OUTPUT)

# 【修改点1】增强的语言控制指令（中英双语强指令）
LANGUAGE_INSTRUCTION = (
    "CRITICAL OUTPUT RULE: You must answer in the SAME language as the user's current question. "
    "If the user asks in English, your entire response (including reasoning, descriptions, and conclusions) MUST be in English. "
    "Note: The provided document context is in Chinese. You must translate the relevant information into English when answering an English question. "
    "如果用户用中文提问，请务必用中文回答。"
)

print(
    f"[qwen3_vl_wrapper] Using vLLM OpenAI backend: {VLLM_BASE_URL}, "
    f"model={VLLM_MODEL_NAME}, "
    f"VLLM_MAX_CONTEXT_TOKENS={VLLM_MAX_CONTEXT_TOKENS}, "
    f"MAX_NEW_TOKENS_DEFAULT={MAX_NEW_TOKENS_DEFAULT}"
)


# ========= 领域内固定提示：蝇类检索表使用规则（仅用于文本 LLM） =========

DOMAIN_INSTRUCTION = """
You are an expert assistant answering questions based on a "Key to Common Flies". 
你正在根据一本关于蝇类分类的“检索表”回答问题。请严格按以下规则理解并使用检索表中的编号。

一、检索表条目的结构
1. 检索条目通常写成类似“2 (3)”“3(2)”的格式，后面接一段形态描述和一个分类结果。
2. 括号前面的数字是本条检索条目的编号，括号里的数字是与之配对的对照条编号。

二、使用检索表时的正向推理方式
1. 对于一对条目，先检查标本是否符合第一个条目的形态描述；若符合，则接受该条给出的分类结果；若不符合，则转向与之成对的另一条进行判断。
2. 回答“属于哪一科/族/属/种”等问题时，请基于检索表中形态特征逐步推理。

三、反向识别问题的处理方式（非常重要）
当用户的问题带有“是什么蝇类”、“What fly is this”等反向识别特征时，请按如下步骤思考：

1. 从用户问题中提取全部显式条件（形态、生态、分布等）。
2. 在“检索条目”和“知识库内容”中，列出所有可能的候选蝇类（禁止发明新名称）。
3. 对每个候选蝇类，逐条对比其已知特征与用户给出的条件。
4. 依据匹配程度做出判断：
   - 若只有 1 个候选高度吻合，作为“最可能的答案”。
   - 若有 2–3 个候选均部分符合，给出“可能的候选列表”。
   - 若严重不匹配，明确说明无法确定。

5. 答复内容要求：
   - 给出结论（最可能是哪一科/族/属/种）。
   - 详细说明关键判断依据：哪些特征与文中描述高度吻合，哪些地方存在不确定。

四、总要求
1. 优先依据检索条目的形态特征与标本特征的一致性来判断。
2. 尽量引用所依据的检索条目编号和关键特征。

五、语言与翻译要求 (Language & Translation)
1. **Follow User Language**: If the user asks in English, translate all Chinese context cues into English for the answer. 
2. **Scientific Names**: Keep Latin scientific names (Genus species) as they are, do not translate them.
3. **Terminology**: Use standard entomological terminology in the target language (e.g., "tergite" for 背板, "vein" for 翅脉).
4. **Missing Info**: If the text doesn't have the info, state "Information not found in the provided key" (or Chinese equivalent based on user language).

六、信息来源限制
1. 你只能使用系统提供的“知识库内容”和“上下文”中的信息来回答。
2. 不要编造文本中未出现的属名或种名。
"""

# ========= 模式专用提示：正向检索 vs 反向识别 =========

# 【修改点2】正向模式增加英文输出指引
FORWARD_MODE_INSTRUCTION = """
【当前任务：正向检索 / Forward Retrieval】
用户通常已经给出了某一科、族、属或种的名称。
User has provided a specific taxon name.

Goal:
1. Trace the path from higher taxa down to this specific group using the key.
2. List the "Forward Retrieval Path" (key numbers and features).
3. Summarize the genus/species characteristics.

**Output Format Rule:**
- If User asks in English: Output the path and summary in English.
- If User asks in Chinese: Output in Chinese.
"""

# 【修改点3】反向模式增加英文输出模板指引
REVERSE_MODE_INSTRUCTION = """
【当前任务：反向识别 / Reverse Identification】
用户给出的是若干形态、生态或分布特征，你需要反推可能的蝇类。
User has provided features, you need to infer the fly species.

Please answer in the following structure (adapt language to user's question):

1. **Conclusion / 结论**:
   E.g., "Based on the features, it most likely belongs to Family X..."

2. **Candidate List / 候选列表** (Prioritize 5 candidates):
   Each candidate on a new line:
   - **Candidate 1**: Name (Family/Genus/Species), **Probability**: XX%, **Reason**: ...
   - **Candidate 2**: ...

   (If answering in Chinese, use: 候选1, 概率, 理由...)

3. **Reasoning Details**:
   Briefly explain matching/mismatching features based on the provided text.

**Critical**: If the user asks in English, you MUST translate "Candidate", "Probability", and "Reason" into English, and translate the description of features into English.
"""


def _parse_mode_from_prompt(raw_prompt: str) -> Tuple[str, str]:
    text = raw_prompt.lstrip()
    mode = "forward"
    stripped = raw_prompt

    if text.startswith("[MODE:REVERSE]"):
        mode = "reverse"
        stripped = text[len("[MODE:REVERSE]"):].lstrip()
    elif text.startswith("[MODE:FORWARD]"):
        mode = "forward"
        stripped = text[len("[MODE:FORWARD]"):].lstrip()

    return mode, stripped


def _build_full_prompt(
    user_prompt: str,
    system_prompt: str = "",
    query_mode: str = "forward",
) -> str:
    if query_mode == "reverse":
        mode_instruction = REVERSE_MODE_INSTRUCTION.strip()
    else:
        mode_instruction = FORWARD_MODE_INSTRUCTION.strip()

    # 2. 固定前缀：模式说明 + 领域说明
    base_prefix_parts: List[str] = [mode_instruction, DOMAIN_INSTRUCTION.strip()]
    base_prefix = "\n\n".join([p for p in base_prefix_parts if p]).strip()

    # 3. 动态 system_prompt（主要是 RAG 的上下文）
    sys_dyn = (system_prompt or "").strip()

    # 4. 用户部分
    user = (user_prompt or "").strip()

    # 5. 拼装
    parts: List[str] = []
    if base_prefix:
        parts.append(base_prefix)
    if sys_dyn:
        parts.append(sys_dyn)
    if user:
        parts.append(user)

    return "\n\n".join(parts).strip()


# ========= 工具：安全解码 base64 图片 =========

def _safe_decode_base64_image(image_data: str) -> Optional[Image.Image]:
    try:
        data = image_data or ""
        if "," in data and "base64" in data[:50]:
            data = data.split(",", 1)[1]
        data = data.strip()
        missing = (-len(data)) % 4
        if missing:
            data += "=" * missing
        binary = base64.b64decode(data)
        img = Image.open(BytesIO(binary)).convert("RGB")
        return img
    except Exception as e:
        print(f"[vision_model_func] failed to decode base64 image: {e}")
        return None


# ========= 文本-only LLM（通过 vLLM OpenAI 接口）=========

def qwen3_vl_chat_text(
    text: str,
    system_prompt: str = "",
    max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    文本对话接口
    """
    # 【修改点4】确保 LANGUAGE_INSTRUCTION 是最后一条 System 指令，权重最高
    sys_parts: List[str] = []
    if system_prompt:
        sys_parts.append(system_prompt.strip())
    
    # 强制添加语言指令
    sys_parts.append(LANGUAGE_INSTRUCTION)
    
    combined_system_prompt = "\n\n".join(sys_parts).strip()

    messages: List[Dict[str, Any]] = []
    if combined_system_prompt:
        messages.append({"role": "system", "content": combined_system_prompt})

    messages.append({"role": "user", "content": text})

    resp = client.chat.completions.create(
        model=VLLM_MODEL_NAME,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    return resp.choices[0].message.content.strip()


# ========= 文本 + 图片 多模态 VLM（目前忽略图片，仅文本回答）=========

def qwen3_vl_chat_vision(
    prompt: str,
    image: Optional[Image.Image],
    system_prompt: str = "",
    max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    return qwen3_vl_chat_text(
        text=prompt,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


# ========= 提供给 RAG-Anything 的两个入口 =========

async def llm_model_func(
    prompt: str,
    system_prompt: str = "",
    history_messages=None,
    **kwargs,
) -> str:
    query_mode, stripped_prompt = _parse_mode_from_prompt(prompt)

    # full_prompt 包含模式指令 + 领域指令 + 检索到的上下文
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
        # 注意：这里 system_prompt 传空字符串，因为上面 full_prompt 里已经拼进去了。
        # 但是 qwen3_vl_chat_text 内部会自动 append LANGUAGE_INSTRUCTION，
        # 这样就实现了：[复杂中文背景] + [用户问题] + [SYSTEM: 必须用英文回答]
        system_prompt="", 
        max_new_tokens=MAX_NEW_TOKENS_DEFAULT,
        **gen_args,
    )


async def vision_model_func(
    prompt: str,
    system_prompt: str = "",
    history_messages=None,
    image_data: Optional[str] = None,
    **kwargs,
) -> str:
    gen_args: Dict[str, Any] = {}
    for k in ["temperature", "top_p"]:
        if k in kwargs:
            gen_args[k] = kwargs[k]

    img = _safe_decode_base64_image(image_data) if image_data else None

    return qwen3_vl_chat_vision(
        prompt=prompt,
        image=img,
        system_prompt=system_prompt or "",
        max_new_tokens=MAX_NEW_TOKENS_DEFAULT,
        **gen_args,
    )