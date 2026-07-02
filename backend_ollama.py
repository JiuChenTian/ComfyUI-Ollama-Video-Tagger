import torch
import gc
import json
import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO

def force_vram_garbage_collection(stage_name=""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(f"[Ollama分支] {stage_name} 成功执行 5090 显存物理大扫除。")

def run_ollama_inference(combined_image, model_path_or_name, prompt, ollama_url, clean_vram_before_load, clean_vram_after_infer):
    if clean_vram_before_load == "enable":
        print("[Ollama分支] 正在执行『加载前显存清理』...")
        force_vram_garbage_collection("模型加载前")

    # 4维转3维矩阵降维挤压，确保 100% 契合 PIL Image 的输入数据要求 [INDEX]
    img_tensor = combined_image.detach().cpu()
    while len(img_tensor.shape) > 3:
        img_tensor = img_tensor.squeeze(0)
        
    img_np = (img_tensor.numpy() * 255).astype(np.uint8)
    collage_pil = Image.fromarray(img_np)
    
    buffered = BytesIO()
    collage_pil.save(buffered, format="JPEG", quality=85)
    image_bytes = buffered.getvalue()

    print(f"[Ollama分支] 正在请求本地 Ollama 接口: {model_path_or_name}...")
    img_base64 = base64.b64encode(image_bytes).decode('utf-8')
    url = f"{ollama_url.rstrip('/')}/api/chat"
    
    payload = {
        "model": model_path_or_name.strip(),
        "messages": [{"role": "user", "content": prompt, "images": [img_base64]}],
        "options": {"temperature": 0.2, "num_predict": -1}, # -1彻底解除 Token 长度限制 [INDEX]
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
