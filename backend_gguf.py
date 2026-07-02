import os
import torch
import gc
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path

def force_vram_garbage_collection(stage_name=""):
    """强行触发 PyTorch 的底层 CUDA 垃圾回收"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(f"[GGUF分支] {stage_name} 成功执行 5090 显存物理大扫除。")

def run_gguf_inference(combined_image, backend_mode, model_path_or_name, optional_mmproj_path, prompt, ctx_vector_length, clean_vram_before_load, clean_vram_after_infer):
    # 动态探测底层依赖，防止未安装时直接卡死
    try:
        from llama_cpp import Llama
    except ImportError:
        return "致命错误: 系统环境未正确编译安装 llama-cpp-python 组件，无法使用 GGUF 后端！"

    # 策略一：加载前清理显存
    if clean_vram_before_load == "enable":
        print("[GGUF分支] 正在执行『加载前显存清理』...")
        force_vram_garbage_collection("模型加载前")

    # 路径安全防御机制：剥离 Windows 复制路径时可能夹带的单双引号及空格
    clean_model_str = str(model_path_or_name).strip().strip('"').strip("'").strip()
    final_model_path = Path(clean_model_str).resolve()
    if not final_model_path.exists():
        return f"Error: 本地大模型文件（.gguf）路径未找到:\n{main_path}"

    # 1. 图像 Tensor 无损转换为 PIL 格式并压为 JPEG 字节流
    img_tensor = combined_image if len(combined_image.shape) == 4 else combined_image
    img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
    collage_pil = Image.fromarray(img_np)
    
    buffered = BytesIO()
    collage_pil.save(buffered, format="JPEG", quality=85)
    image_bytes = buffered.getvalue()

    output_text = ""
    try:
        print("[GGUF分支] 正在装载本地 GGUF 视觉架构处理器...")
        chat_handler = None
        
        # 模式 A：大脑眼睛分开模式（如跑 Qwen3-VL 8B）
        if backend_mode == "Local_GGUF_Separated":
            mm_str = str(optional_mmproj_path).strip().strip('"').strip("'").strip()
            mm_path = Path(mm_str).resolve()
            if not mm_path.exists():
                return f"Error: 分开模式下必须提供有效的眼睛（mmproj）文件路径:\n{mm_path}"
            
            # 【终极核心修复】：动态锁死最新版接口中强制要求的 clip_model_path 位置参数
            try:
                from llama_cpp.llama_chat_format import MMTDChatHandler
                chat_handler = MMTDChatHandler(mmproj_path=str(mm_path), clip_model_path=str(final_model_path))
            except BaseException:
                try:
                    from llama_cpp.llama_chat_format import NanoLlavaChatHandler
                    chat_handler = NanoLlavaChatHandler(mmproj_path=str(mm_path), clip_model_path=str(final_model_path))
                except BaseException:
                    from llama_cpp.llama_chat_format import NanoLlavaChatHandler
                    chat_handler = NanoLlavaChatHandler(mmproj_path=str(mm_path))
        
        # 模式 B：大脑眼睛单文件一体化模式
        else:
            print("[GGUF分支] 检测为单 GGUF 一体化模型，正在调用一体化多模态 Handler...")
            try:
                from llama_cpp.llama_chat_format import NanoLlavaChatHandler
                chat_handler = NanoLlavaChatHandler(mmproj_path=str(final_model_path))
            except BaseException:
                chat_handler = None

        print("[GGUF分支] 正在将 GGUF 模型全量层（-1）轰入 5090 显存...")
        llm = Llama(
            model_path=str(final_model_path),
            chat_handler=chat_handler,
            n_ctx=int(ctx_vector_length),
            n_gpu_layers=-1,  # 5090满血参数
            verbose=False
        )
        
        print("[GGUF分支] 5090 正在全速进行 GGUF 矩阵单图推理...")
        response = llm.create_chat_completion(
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt}, 
                    {"type": "image_url", "image_url": {"url": image_bytes}}
                ]
            }],
            temperature=0.2, 
            max_tokens=256
        )
        output_text = response["choices"]["message"]["content"].strip()

    except Exception as e:
        output_text = f"GGUF本地推理发生严重异常: {str(e)}"
        print(f"[GGUF分支] 错误详情: {str(e)}")
    finally:
        # 无论成功还是失败，均地毯式粉碎指针，强制释放 5090 显存
        if 'llm' in locals(): del llm
        if 'chat_handler' in locals(): del chat_handler
        
        if clean_vram_after_infer == "enable":
            force_vram_garbage_collection("模型推理后")

    return output_text
