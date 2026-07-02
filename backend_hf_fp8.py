import os
import torch
import gc
import numpy as np
from PIL import Image
from pathlib import Path

def force_vram_garbage_collection(stage_name=""):
    """强行触发 PyTorch 的底层 CUDA 垃圾回收"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(f"[HF-FP8分支] {stage_name} 成功执行 5090 显存物理大扫除。")

def run_hf_fp8_inference(combined_image, model_path_or_name, prompt, clean_vram_before_load, clean_vram_after_infer):
    # 动态探测底层依赖，防止未安装时直接卡死
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
    except ImportError:
        return "致命错误: 系统环境缺少 transformers 和 accelerate 推理依赖包，无法使用官方散文件夹后端！"

    # 策略一：加载前清理显存
    if clean_vram_before_load == "enable":
        print("[HF-FP8分支] 正在执行『加载前显存清理』...")
        force_vram_garbage_collection("模型加载前")

    # 路径安全防御机制：剥离 Windows 复制路径时可能夹带的单双引号及前后空格
    clean_folder_str = str(model_path_or_name).strip().strip('"').strip("'").strip()
    folder_path = Path(clean_folder_str).resolve()
    if not folder_path.exists() or not folder_path.is_dir():
        return f"Error: 官方一体原版散文件夹路径不存在或不合法，请检查输入框内容:\n{model_path_or_name}"

    # 1. 图像 Tensor 无损转换为标准的 PIL 对象
    img_tensor = combined_image if len(combined_image.shape) == 4 else combined_image
    img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
    collage_pil = Image.fromarray(img_np)

    output_text = ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        print(f"[HF-FP8分支] 正在以 FP8 硬件加速格式载入本地官方原版散文件夹: {folder_path}...")
        
        # 2. 动态自适应加载本地权重的核心处理器与多模态序列模型
        processor = AutoProcessor.from_pretrained(str(folder_path), trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            str(folder_path), 
            torch_dtype=torch.float8_e4m3fn,  # 【5090专属硬件级 FP8 核心加速格式】
            device_map=device, 
            trust_remote_code=True
        )
        
        # 3. 遵循官方的标准对话模版规则构建输入
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[collage_pil], padding=True, return_tensors="pt").to(device)

        print("[HF-FP8分支] 5090 正在执行 FP8 纯原生多模态张量推理...")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
        
        # 4. 剪掉输入的 Prompt 长度，只保留生成的文本结果并解码
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True).strip()

    except Exception as e:
        output_text = f"本地原版散文件夹推理发生致命错误: {str(e)}"
        print(f"[HF-FP8分支] 错误详情: {str(e)}")
    finally:
        # 无论成功还是失败，均地毯式粉碎显存中高压常驻的张量和模型对象
        if 'model' in locals(): del model
        if 'processor' in locals(): del processor
        if 'inputs' in locals(): del inputs
        
        if clean_vram_after_infer == "enable":
            force_vram_garbage_collection("模型推理后")

    return output_text
