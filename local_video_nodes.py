import sys
import importlib
from pathlib import Path
from .collage_builder import VideoFrameCollageBuilder

class UltimateMultimodalTagger:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "combined_image": ("IMAGE",),  # 挂载 5xN 节点拼好的大图
                "backend_mode": (["Ollama_API", "Local_GGUF_Separated", "Local_GGUF_Single_File", "Local_Transformers_FP8_Folder"], {"default": "Ollama_API"}),
                "model_path_or_name": ("STRING", {"default": "模型名字 或 磁盘绝对路径", "multiline": False}),
                "optional_mmproj_path": ("STRING", {"default": "选填：仅GGUF眼睛大脑分开时填路径", "multiline": False}),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "The input image is a 5xN storyboard grid combined from sequential video frames chronologically. Analyze the continuous action, scene, objects, and camera motion from top-left to bottom-right. Output ONLY a detailed English prompt for video generation AI, no chat."
                }),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "ctx_vector_length": ("INT", {"default": 2048, "min": 512, "max": 8192, "step": 512}),
                "clean_vram_before_load": (["enable", "disable"], {"default": "enable"}), 
                "clean_vram_after_infer": (["enable", "disable"], {"default": "enable"}), 
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "route_inference"
    CATEGORY = "Local_LLM_Tools/Infer"

    def route_inference(self, combined_image, backend_mode, model_path_or_name, optional_mmproj_path, prompt, ollama_url, ctx_vector_length, clean_vram_before_load, clean_vram_after_infer):
        # ====================================================================
        # 【全架构热重载矩阵】：点击运行时一键强刷重读所有子模块，彻底告别重启黑窗口
        # ====================================================================
        BASE_MODULE = "custom_nodes.ComfyUI-Ollama-Video-Tagger"
        for sub_mod in ["backend_ollama", "backend_gguf", "backend_hf_fp8"]:
            full_name = f"{BASE_MODULE}.{sub_mod}"
            if full_name in sys.modules:
                try:
                    importlib.reload(sys.modules[full_name])
                except Exception:
                    pass
        print("\n" + "🔥"*25 + "\n  [🚀 中枢通知] 多模态核心矩阵已成功触发全局实时热重载！\n" + "🔥"*25 + "\n")
        # ====================================================================

        # 1. 动态按需加载对应的独立后端子文件
        if backend_mode == "Ollama_API":
            from .backend_ollama import run_ollama_inference
            return (run_ollama_inference(combined_image, model_path_or_name, prompt, ollama_url, clean_vram_before_load, clean_vram_after_infer),)
            
        elif backend_mode in ["Local_GGUF_Separated", "Local_GGUF_Single_File"]:
            from .backend_gguf import run_gguf_inference
            return (run_gguf_inference(combined_image, backend_mode, model_path_or_name, optional_mmproj_path, prompt, ctx_vector_length, clean_vram_before_load, clean_vram_after_infer),)
            
        elif backend_mode == "Local_Transformers_FP8_Folder":
            from .backend_hf_fp8 import run_hf_fp8_inference
            return (run_hf_fp8_inference(combined_image, model_path_or_name, prompt, clean_vram_before_load, clean_vram_after_infer),)

        return ("Error: 未知的后端模式",)
