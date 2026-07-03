import os
import torch
import gc
import numpy as np
from PIL import Image
from pathlib import Path

def force_vram_garbage_collection(stage_name=""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(f"[HF-FP8后端] {stage_name} 成功执行 5090 显存清理。")

def run_hf_fp8_inference(combined_image, model_path_or_name, prompt, clean_vram_before_load, clean_vram_after_infer):
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
    except ImportError:
        return "致命错误: 系统环境缺少 transformers 和 accelerate 推理依赖包！"

    if clean_vram_before_load == "enable":
        force_vram_garbage_collection("模型加载前")

    clean_folder_str = str(model_path_or_name).strip().strip('"').strip("'").strip()
    folder_path = Path(clean_folder_str).resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        return f"Error: 官方原版散文件夹路径不存在:\n{model_path_or_name}"

    img_tensor = combined_image.detach().cpu()
    while len(img_tensor.shape) > 3:
        img_tensor = img_tensor.squeeze(0)
    img_np = (img_tensor.numpy() * 255).astype(np.uint8)
    collage_pil = Image.fromarray(img_np)

    output_text = ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        print(f"[HF-FP8后端] 正在以 FP8 核心载入: {folder_path}...")
        processor = AutoProcessor.from_pretrained(str(folder_path), trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            str(folder_path), torch_dtype=torch.float8_e4m3fn, device_map=device, trust_remote_code=True # 【5090专属硬件级 FP8】 [INDEX]
        )
        
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[collage_pil], padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=4096)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True).strip()

    except Exception as e:
        output_text = f"本地原版散文件夹推理发生致命错误: {str(e)}"
    finally:
        if 'model' in locals(): del model
        if 'processor' in locals(): del processor
        if 'inputs' in locals(): del inputs
        if clean_vram_after_infer == "enable":
            force_vram_garbage_collection("模型推理后")

    return output_text
