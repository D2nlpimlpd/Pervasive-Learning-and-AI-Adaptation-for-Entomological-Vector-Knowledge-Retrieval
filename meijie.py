# qwen3_vl_wrapper.py

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from typing import List, Dict, Any

# ① 这里改成你本地模型的目录
MODEL_DIR = r"/root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    dtype=torch.float16,   
    device_map="auto",     
)
processor = AutoProcessor.from_pretrained(MODEL_DIR)


def qwen3_vl_chat(
    messages: List[Dict[str, Any]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    messages 结构例子：
    [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "你好"},
        ],
      }
    ]
    """
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    # 只取新生成的部分
    generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text.strip()


def llm_model_func(
    prompt: str,
    system_prompt: str = "",
    **kwargs,
) -> str:
    """
    给 RAG-Anything 用的文本 LLM 接口：输入一个 prompt，返回回答文本 [ref:1,4]()
    """
    contents = []
    if system_prompt:
        contents.append({"type": "text", "text": system_prompt})
    contents.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": contents,
        }
    ]
    return qwen3_vl_chat(messages)