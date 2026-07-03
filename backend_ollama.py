import torch
import gc
import json
import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO

def force_vram_garbage_collection(stage_name=""):
    """强行触发 PyTorch 的底层 CUDA 垃圾回收"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(f"[Ollama后端] {stage_name} 成功执行 5090 显存清理。")

def run_ollama_inference(combined_image, model_path_or_name, prompt_or_messages, ollama_url, clean_vram_before_load, clean_vram_after_infer):
    if clean_vram_before_load == "enable":
        force_vram_garbage_collection("模型加载前")

    # 1. 深度降维防护：剥离外围多余的批次墙，确保 100% 契合 PIL Image 像素要求 [INDEX]
    img_tensor = combined_image.detach().cpu()
    while len(img_tensor.shape) > 3:
        img_tensor = img_tensor.squeeze(0)
        
    img_np = (img_tensor.numpy() * 255).astype(np.uint8)
    collage_pil = Image.fromarray(img_np)
    
    buffered = BytesIO()
    collage_pil.save(buffered, format="JPEG", quality=85)
    image_bytes = buffered.getvalue()

    print(f"[Ollama后端] 正在请求本地 Ollama 服务接口: {model_path_or_name}...")
    img_base64 = base64.b64encode(image_bytes).decode('utf-8')
    url = f"{ollama_url.rstrip('/')}/api/chat"
    
    # 2. 【核心自适应解包解耦算法】：如果上级中枢发过来的是专为 27B 准备的 messages 复杂列表结构，
    # 我们通过 Python 循环动态将里面的纯文本和图片剥离，100% 还原并重构为 Ollama API 唯一死锁认领的严格格式！
    final_ollama_messages = []
    
    if isinstance(prompt_or_messages, list):
        for msg in prompt_or_messages:
            role = msg.get("role")
            content = msg.get("content")
            
            # Ollama 要求 content 必须是纯 string。如果是复杂的 list 格式（如最后一轮携带图的 user 气泡），我们进行强行降维提取
            if isinstance(content, list):
                extracted_text = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        extracted_text = item.get("text", "")
                        break
                # 重新封装为 Ollama 认领的纯字符串消息气泡
                final_ollama_messages.append({"role": role, "content": str(extracted_text).strip()})
            else:
                # system 和 assistant 的纯文本历史直接保留
                final_ollama_messages.append({"role": role, "content": str(content).strip()})
    else:
        # 如果是老版本的普通纯文本字符串提示词，直接降级兼容包装入 user 气泡
        final_ollama_messages.append({"role": "user", "content": str(prompt_or_messages).strip()})

    # 3. 强行将单帧 Base64 图片压入最后一轮 user 的消息包裹中（这是 Ollama 多模态 API 的铁律设计规范）
    if final_ollama_messages:
        # 找到最后一个由用户发出的消息气泡，直接挂载 images 字段
        for last_msg in reversed(final_ollama_messages):
            if last_msg.get("role") == "user":
                last_msg["images"] = [img_base64]
                break

    payload = {
        "model": model_path_or_name.strip(),
        "messages": final_ollama_messages,
        "options": {"temperature": 0.2, "num_predict": -1}, # -1彻底解除 Token 长度限制，让大模型把话完整说完 [INDEX]
        "stream": False
    }
    
    output_text = ""
    try:
        res = requests.post(url, json=payload, timeout=300)
        if res.status_code == 200:
            output_text = res.json().get("message", {}).get("content", "").strip()
        else:
            output_text = f"Ollama 内部报错: {res.text}"
    except Exception as e:
        output_text = f"连接 Ollama 失败: {str(e)}"

    if clean_vram_after_infer == "enable":
        force_vram_garbage_collection("模型推理后")

    return output_text
