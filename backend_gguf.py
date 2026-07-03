import os
import torch
import gc
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path

def force_vram_garbage_collection(stage_name=""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print(f"[GGUF后端] {stage_name} 成功执行 5090 显存清理。")

def run_gguf_inference(combined_image, backend_mode, model_path_or_name, optional_mmproj_path, prompt, ctx_vector_length, clean_vram_before_load, clean_vram_after_infer):
    try:
        from llama_cpp import Llama
    except ImportError:
        return "致命错误: 系统环境未正确编译安装 llama-cpp-python 组件！"

    if clean_vram_before_load == "enable":
        force_vram_garbage_collection("模型加载前")

    clean_model_str = str(model_path_or_name).strip().strip('"').strip("'").strip()
    final_model_path = Path(clean_model_str).resolve()
    if not final_model_path.exists():
        return f"Error: 大脑文件（.gguf）路径未找到:\n{final_model_path}"

    img_tensor = combined_image.detach().cpu()
    while len(img_tensor.shape) > 3:
        img_tensor = img_tensor.squeeze(0)
        
    img_np = (img_tensor.numpy() * 255).astype(np.uint8)
    collage_pil = Image.fromarray(img_np)
    buffered = BytesIO()
    collage_pil.save(buffered, format="JPEG", quality=85)
    image_bytes = buffered.getvalue()

    output_text = ""
    try:
        chat_handler = None
        if backend_mode == "Local_GGUF_Separated":
            mm_str = str(optional_mmproj_path).strip().strip('"').strip("'").strip()
            mm_path = Path(mm_str).resolve()
            if not mm_path.exists():
                return f"Error: 眼睛文件（mmproj）路径未找到:\n{mm_path}"
            
            # 【终极强制对齐】使用顶级基类 BaseException 进行绝对死锁拦截 [INDEX]
            try:
                from llama_cpp.llama_chat_format import MMTDChatHandler
                chat_handler = MMTDChatHandler(mmproj_path=str(mm_path), clip_model_path=str(final_model_path))
            except BaseException:
                try:
                    from llama_cpp.llama_chat_format import NanoLlavaChatHandler
                    chat_handler = NanoLlavaChatHandler(mmproj_path=str(mm_path), clip_model_path=str(final_model_path))
                except BaseException:
                    try:
                        from llama_cpp.llama_chat_format import NanoLlavaChatHandler
                        chat_handler = NanoLlavaChatHandler(mmproj_path=str(mm_path))
                    except BaseException:
                        chat_handler = None
        else:
            try:
                from llama_cpp.llama_chat_format import NanoLlavaChatHandler
                chat_handler = NanoLlavaChatHandler(mmproj_path=str(final_model_path))
            except BaseException:
                chat_handler = None

        print("[GGUF后端] 正在轰入 5090 全量层显存推理...")
        llm = Llama(
            model_path=str(final_model_path), chat_handler=chat_handler,
            n_ctx=int(ctx_vector_length), n_gpu_layers=-1, verbose=False # 吃满32G显存 [INDEX]
        )
        
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_bytes}}]}],
            temperature=0.2, max_tokens=4096
        )
        
        # 【超级自适应解包内核】100% 击穿任何底层库引发的列表/字典解析冲突 [INDEX]
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if isinstance(choices, list) and len(choices) > 0:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    message = first_choice.get("message", {})
                    if isinstance(message, dict):
                        output_text = message.get("content", "").strip()
                    else:
                        output_text = str(message).strip()
                else:
                    output_text = str(first_choice).strip()
            elif isinstance(choices, dict):
                output_text = choices.get("text", choices.get("content", str(choices))).strip()
            else:
                output_text = response.get("content", response.get("text", str(response))).strip()
        else:
            output_text = str(response).strip()

    except BaseException as e:
        output_text = f"GGUF本地推理发生严重异常: {str(e)}"
    finally:
        if 'llm' in locals(): del llm
        if 'chat_handler' in locals(): del chat_handler
        if clean_vram_after_infer == "enable":
            force_vram_garbage_collection("模型推理后")

    return output_text
