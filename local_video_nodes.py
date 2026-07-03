import sys
from pathlib import Path
from .collage_builder import VideoFrameCollageBuilder

class UltimateMultimodalTagger:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "combined_image": ("IMAGE",), 
                "backend_mode": (["Ollama_API", "Local_GGUF_Separated", "Local_GGUF_Single_File", "Local_Transformers_FP8_Folder"], {"default": "Ollama_API"}),
                "model_path_or_name": ("STRING", {"default": "qwen3-vl:8b 或 磁盘绝对路径", "multiline": False}),
                "optional_mmproj_path": ("STRING", {"default": "选填：仅GGUF眼睛大脑分开时填路径", "multiline": False}),
                
                # 场景属性控制 Widget，100% 抹杀大模型无中生有和人数错乱的幻觉 [INDEX]
                "subject_type": (["Human/Character(人物/角色)", "Animal/Pet(动物/宠物)", "Vehicle/Object(车辆/物体)", "Scenery/Landscape(风景/自然)"], {"default": "Human/Character(人物/角色)"}),
                "subject_count_mode": (["Single_Subject(单一主体)", "Multiple_Subjects(多个主体/群体)", "Auto_Detect(让模型自动判定数量)"], {"default": "Single_Subject(单一主体)"}),
                
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "Describe the subtle chronological action, rich material details, and movements unfolding in this continuous video frame matrix. Output only a dense paragraph of descriptive English prose prompt for text-to-video AI."
                }),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "ctx_vector_length": ("INT", {"default": 4096, "min": 512, "max": 32768, "step": 1024}), 
                "clean_vram_before_load": (["enable", "disable"], {"default": "enable"}), 
                "clean_vram_after_infer": (["enable", "disable"], {"default": "enable"}), 
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_list",)
    FUNCTION = "route_inference_batch"
    CATEGORY = "Local_LLM_Tools/Infer"

    def route_inference_batch(self, combined_image, backend_mode, model_path_or_name, optional_mmproj_path, subject_type, subject_count_mode, prompt, ollama_url, ctx_vector_length, clean_vram_before_load, clean_vram_after_infer):
        image_shape = list(combined_image.shape)
        num_segments = int(image_shape[0]) # 正确解包批次数量维度 [INDEX]
        print(f"[中枢核心] 接收到 {num_segments} 个故事板切片。滚动记忆机制启动...")
        
        from .backend_ollama import run_ollama_inference
        from .backend_gguf import run_gguf_inference
        from .backend_hf_fp8 import run_hf_fp8_inference
        
        # 提取选择特征并动态拼装底座规则
        sub_type = subject_type.split("(")[0].strip()
        count_mode = subject_count_mode.split("(")[0].strip()
        
        system_rules = (
            f"The input image is a square storyboard matrix combined chronologically from ONE continuous video clip (ordered from top-left to bottom-right).\n"
            f"Your absolute goal is to reverse-engineer a detailed English text-to-video prompt. Follow these strict rules:\n"
            f"- TARGET CATEGORY: The main focal subject is a [{sub_type}]. Focus deeply on its unique physical characteristics, materials, and specific features.\n"
        )
        
        if count_mode == "Single_Subject":
            system_rules += "- SUBJECT CONSTRAINT: There is exactly ONE single subject across the timeline. Do NOT interpret different frames as multiple distinct entities. Keep the subject singular and consistent.\n"
        elif count_mode == "Multiple_Subjects":
            system_rules += "- SUBJECT CONSTRAINT: There are multiple interacting subjects or a group in this continuous video timeline. Track and describe their group coordination or individual actions consistently across frames.\n"
        else:
            system_rules += "- SUBJECT CONSTRAINT: Carefully determine the exact count of the main subjects based on the visual sequence. Maintain identity consistency for each subject across the timeline.\n"
            
        system_rules += (
            "- CHRONOLOGICAL KINETICS: Focus on how the action, movements, or environment changes step-by-step from frame to frame. Describe the progression smoothly.\n"
            "- ENVIRONMENT LOCK: The background represents a single location. Accurately describe the textures, lighting, and elements without hallucinating non-existent objects.\n"
            "- OUTPUT FORMAT: Output ONLY the high-quality descriptive paragraph of rich keywords and prose. Do NOT write any conversational chat."
        )

        all_results = []
        previous_segment_memory = ""
        
        for idx in range(num_segments):
            print(f"[中枢核心] 正在串行处理第 {idx+1}/{num_segments} 个片段大图...")
            single_grid_tensor = combined_image[idx].unsqueeze(0) 
            
            # 滚雪球式前情提要注入，100% 抹杀废话和复读 [INDEX]
            if idx == 0:
                dynamic_prompt = f"{system_rules}\n\nUSER PROMPT TASK:\n{prompt}"
            else:
                dynamic_prompt = (
                    f"{system_rules}\n\n"
                    f"USER PROMPT TASK:\n{prompt}\n\n"
                    f"==================================================\n"
                    f"CRITICAL CONTINUITY INSTRUCTIONS (CONTEXT ROLLING):\n"
                    f"This is a continuation of the video timeline. You already described the previous segment as follows:\n"
                    f"\"\"\"{previous_segment_memory}\"\"\"\n\n"
                    f"DO NOT REPEAT the static environment description, unchanged clothing/fur/texture details from the previous text.\n"
                    f"Focus EXCLUSIVELY on describing the NEW changes, chronological action progression, and visual motion from where the previous segment left off.\n"
                    f"Use seamless transitional phrases at the start of your text (e.g., 'Continuing the scene...', 'Seamlessly transitioning to...', 'Following the prior movement...').\n"
                    f"=================================================="
                )
            
            segment_text = ""
            if backend_mode == "Ollama_API":
                segment_text = run_ollama_inference(single_grid_tensor, model_path_or_name, dynamic_prompt, ollama_url, clean_vram_before_load, clean_vram_after_infer)
            elif backend_mode in ["Local_GGUF_Separated", "Local_GGUF_Single_File"]:
                segment_text = run_gguf_inference(single_grid_tensor, backend_mode, model_path_or_name, optional_mmproj_path, dynamic_prompt, ctx_vector_length, clean_vram_before_load, clean_vram_after_infer)
            elif backend_mode == "Local_Transformers_FP8_Folder":
                segment_text = run_hf_fp8_inference(single_grid_tensor, model_path_or_name, dynamic_prompt, clean_vram_before_load, clean_vram_after_infer)
                
            if segment_text and not segment_text.startswith("Error") and not segment_text.startswith("致命错误"):
                all_results.append(segment_text)
                previous_segment_memory = segment_text  
            else:
                all_results.append(f"[Segment {idx+1} Error: {segment_text}]")
                previous_segment_memory = "" 
                
        final_encoded_string = "|||".join(all_results)
        return (final_encoded_string,)

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
        if not prompt_list: return ("",)
        raw_segments = prompt_list.split("|||")
        processed_segments = []
        for idx, seg_content in enumerate(raw_segments):
            seg_content = seg_content.strip()
            if not seg_content: continue
            if add_segment_index_tag == "enable":
                formatted_segment = f"[Segment {idx + 1}]: {seg_content}"
            else:
                formatted_segment = seg_content
            processed_segments.append(formatted_segment)
        core_concated_body = join_delimiter.join(processed_segments)
        final_prompt = core_concated_body
        if prefix_text.strip(): final_prompt = f"{prefix_text.strip()}\n{final_prompt}"
        if suffix_text.strip(): final_prompt = f"{final_prompt}\n{suffix_text.strip()}"
        return (final_prompt,)
