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
    print(f"[Ollama后端] {stage_name} 成功执行 5090 显存物理大扫除。")

def run_ollama_inference(combined_image, model_path_or_name, prompt, ollama_url, clean_vram_before_load, clean_vram_after_infer):
    # 策略一：加载前清理显存
    if clean_vram_before_load == "enable":
        print("[Ollama后端] 正在执行『加载前显存清理』...")
        force_vram_garbage_collection("模型加载前")

    # 1. 深度降维防护：剥离外围多余的批次墙，确保 100% 契合 PIL Image 像素读取要求 [INDEX]
    img_tensor = combined_image.detach().cpu()
    while len(img_tensor.shape) > 3:
        img_tensor = img_tensor.squeeze(0)
        
    img_np = (img_tensor.numpy() * 255).astype(np.uint8)
    collage_pil = Image.fromarray(img_np)
    
    buffered = BytesIO()
    collage_pil.save(buffered, format="JPEG", quality=85)
    image_bytes = buffered.getvalue()

    # 2. 进行标准的 Base64 编码并组装请求体
    print(f"[Ollama后端] 正在请求本地 Ollama 服务接口: {model_path_or_name}...")
    img_base64 = base64.b64encode(image_bytes).decode('utf-8')
    url = f"{ollama_url.rstrip('/')}/api/chat"
    
    payload = {
        "model": model_path_or_name.strip(),
        "messages": [{"role": "user", "content": prompt, "images": [img_base64]}],
        "options": {"temperature": 0.2, "num_predict": -1}, # -1彻底解锁 Token 长度限制，直至大模型描述完毕为止 [INDEX]
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

    # 策略二：推理后强制清理显存
    if clean_vram_after_infer == "enable":
        force_vram_garbage_collection("模型推理后")

    return output_text
