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
                "combined_image": ("IMAGE",),  # 挂载前置网格节点拼好的多图批次
                "backend_mode": (["Ollama_API", "Local_GGUF_Separated", "Local_GGUF_Single_File", "Local_Transformers_FP8_Folder"], {"default": "Ollama_API"}),
                "model_path_or_name": ("STRING", {"default": "模型名字 或 磁盘绝对路径", "multiline": False}),
                "optional_mmproj_path": ("STRING", {"default": "选填：仅GGUF眼睛大脑分开时填路径", "multiline": False}),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "The input image is a square storyboard grid sequence from a video segment. Analyze the actions, movements, and cinematic scene. Output ONLY a detailed English prompt for text-to-video AI, no intro, no conversational text."
                }),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "ctx_vector_length": ("INT", {"default": 2048, "min": 512, "max": 32768, "step": 512}),
                "clean_vram_before_load": (["enable", "disable"], {"default": "enable"}), 
                "clean_vram_after_infer": (["enable", "disable"], {"default": "enable"}), 
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_list",)
    FUNCTION = "route_inference_batch"
    CATEGORY = "Local_LLM_Tools/Infer"

    def route_inference_batch(self, combined_image, backend_mode, model_path_or_name, optional_mmproj_path, prompt, ollama_url, ctx_vector_length, clean_vram_before_load, clean_vram_after_infer):
        image_shape = list(combined_image.shape)
        num_segments = int(image_shape[0])
        print(f"[中枢反推] 接收到 {num_segments} 个故事板切片画布，开始执行自适应串行循环推理...")
        
        # 动态动态按需载入各独立分支后端
        from .backend_ollama import run_ollama_inference
        from .backend_gguf import run_gguf_inference
        from .backend_hf_fp8 import run_hf_fp8_inference
        
        all_results = []
        
        for idx in range(num_segments):
            print(f"[中枢反推] 正在处理第 {idx+1}/{num_segments} 个视频片段网格...")
            single_grid_tensor = combined_image[idx].unsqueeze(0) 
            
            segment_text = ""
            if backend_mode == "Ollama_API":
                segment_text = run_ollama_inference(single_grid_tensor, model_path_or_name, prompt, ollama_url, clean_vram_before_load, clean_vram_after_infer)
            elif backend_mode in ["Local_GGUF_Separated", "Local_GGUF_Single_File"]:
                segment_text = run_gguf_inference(single_grid_tensor, backend_mode, model_path_or_name, optional_mmproj_path, prompt, ctx_vector_length, clean_vram_before_load, clean_vram_after_infer)
            elif backend_mode == "Local_Transformers_FP8_Folder":
                segment_text = run_hf_fp8_inference(single_grid_tensor, model_path_or_name, prompt, clean_vram_before_load, clean_vram_after_infer)
                
            if segment_text and not segment_text.startswith("Error") and not segment_text.startswith("致命错误"):
                all_results.append(segment_text)
            else:
                all_results.append(f"[Segment {idx+1} Error: {segment_text}]")
                
        final_encoded_string = "|||".join(all_results)
        return (final_encoded_string,)

# 节点三：新加的文本时序拼接累加器
class VideoTextPromptConcatenator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_list": ("STRING", {"forceInput": True}),
                "join_delimiter": (["\n\n", " ", ", ", " | "], {"default": "\n\n"}), 
                "prefix_text": ("STRING", {"default": "", "multiline": False}), 
                "suffix_text": ("STRING", {"default": "", "multiline": False}), 
                "add_segment_index_tag": (["enable", "disable"], {"default": "enable"}), 
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_concated_prompt",)
    FUNCTION = "concatenate_prompts"
    CATEGORY = "Local_LLM_Tools/Processors"

    def concatenate_prompts(self, prompt_list, join_delimiter, prefix_text, suffix_text, add_segment_index_tag):
        if not prompt_list:
            return ("",)
            
        raw_segments = prompt_list.split("|||")
        processed_segments = []
        
        for idx, seg_content in enumerate(raw_segments):
            seg_content = seg_content.strip()
            if not seg_content:
                continue
                
            if add_segment_index_tag == "enable":
                formatted_segment = f"[Segment {idx + 1}]: {seg_content}"
            else:
                formatted_segment = seg_content
                
            processed_segments.append(formatted_segment)
            
        core_concated_body = join_delimiter.join(processed_segments)
        
        final_prompt = core_concated_body
        if prefix_text.strip():
            final_prompt = f"{prefix_text.strip()}\n{final_prompt}"
        if suffix_text.strip():
            final_prompt = f"{final_prompt}\n{suffix_text.strip()}"
            
        print(f"[文本拼接器] 成功合并了 {len(processed_segments)} 段时间轴切片的反推提示词。")
        return (final_prompt,)
