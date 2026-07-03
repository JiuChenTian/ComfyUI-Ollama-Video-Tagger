import sys
import re  # 引入正则表达式清洗器
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
                
                # 自适应属性选择 Widget
                "subject_type": (["Human/Character(人物/角色)", "Animal/Pet(动物/宠物)", "Vehicle/Object(车辆/物体)", "Scenery/Landscape(风景/自然)"], {"default": "Human/Character(人物/角色)"}),
                "subject_count_mode": (["Single_Subject(单一主体)", "Multiple_Subjects(多个主体/群体)", "Auto_Detect(让模型自动判定数量)"], {"default": "Single_Subject(单一主体)"}),
                
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "A crystalline sharp cinematic medium shot. Intensely describe the precise visual elements, clothing fabrics, environment details, and fluid physical choreography evolving from frame to frame chronologically."
                }),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "ctx_vector_length": ("INT", {"default": 8192, "min": 512, "max": 32768, "step": 1024}), 
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
        # 【完美硬核修复行】：必须精确加上 [0] 索引提取，才能将 list 转化为代表 Segment 总段数的纯整数！
        num_segments = int(image_shape[0]) 
        print(f"[27B神装中枢] 成功接收到 {num_segments} 个故事板切片。自适应双轨气泡隔离机制执行中...")
        
        from .backend_ollama import run_ollama_inference
        from .backend_gguf import run_gguf_inference
        from .backend_hf_fp8 import run_hf_fp8_inference
        
        sub_type = subject_type.split("(")[0].strip()
        count_mode = subject_count_mode.split("(")[0].strip()
        
        # 1. 建立标准的 OpenAI system 角色隔离气泡，封死 27B 的思维链自言自语
        SYSTEM_COMMAND = (
            f"You are a video captioning machine. You rewrite video frames into rich English prose prompts for video generation AI.\n"
            f"- TARGET CATEGORY: [{sub_type}] | COUNT MODE: [{count_mode}].\n"
            f"- ABSOLUTE RULES: NEVER output any thinking process, reasoning, annotations, or self-explanations. Never write sentences like 'We need to analyze', 'The image shows', 'The input is a grid'.\n"
            f"- FORMAT: Output ONLY one single massive dense paragraph of rich adjectives and active motion verbs. Start writing description immediately. No preamble, no intro."
        )

        all_results = []
        previous_segment_memory = ""
        
        # 2. 串行流式循环流水线
        for idx in range(num_segments):
            print(f"[27B神装中枢] 5090 正在全速加载并反推第 {idx+1}/{num_segments} 个宫格画布...")
            single_grid_tensor = combined_image[idx].unsqueeze(0) 
            
            # 双轨多气泡隔离：把上一轮吐出来的词当作 assistant 历史记录喂给它，强行掐断复读冲动 [INDEX]
            if idx == 0:
                messages_packet = [
                    {"role": "system", "content": SYSTEM_COMMAND},
                    {"role": "user", "content": prompt}
                ]
            else:
                CONTINUITY_RULE = (
                    f"{prompt}\n\n"
                    f"[CRITICAL CONTINUITY COMPLIANCE]:\n"
                    f"This is a direct continuation of the video timeline. Do NOT repeat the background setup, clothing material, or facial looks you described in your last response.\n"
                    f"Describe ONLY the newly changed choreography movements, actions, and motion progression. Start immediately with a transitional verb link."
                )
                messages_packet = [
                    {"role": "system", "content": SYSTEM_COMMAND},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": previous_segment_memory},
                    {"role": "user", "content": CONTINUITY_RULE}
                ]
            
            segment_text = ""
            if backend_mode == "Ollama_API":
                segment_text = run_ollama_inference(single_grid_tensor, model_path_or_name, messages_packet, ollama_url, clean_vram_before_load, clean_vram_after_infer)
            elif backend_mode in ["Local_GGUF_Separated", "Local_GGUF_Single_File"]:
                segment_text = run_gguf_inference(single_grid_tensor, backend_mode, model_path_or_name, optional_mmproj_path, messages_packet, ctx_vector_length, clean_vram_before_load, clean_vram_after_infer)
            elif backend_mode == "Local_Transformers_FP8_Folder":
                segment_text = run_hf_fp8_inference(single_grid_tensor, model_path_or_name, messages_packet, clean_vram_before_load, clean_vram_after_infer)
                
            if segment_text and not segment_text.startswith("Error") and not segment_text.startswith("致命错误"):
                # 【27B 正则级物理拦截防御清洗器】高级行过滤器
                cleaned_lines = []
                for line in segment_text.split("\n"):
                    clean_line = line.strip()
                    if not clean_line:
                        continue
                    lower_line = clean_line.lower()
                    if any(x in lower_line for x in ["we need to", "the user", "let's describe", "the prompt must", "in order to", "the image shows", "looking at the", "appears to be"]):
                        continue
                    clean_line = re.sub(r'^(here is the prompt:|caption:|description:|output:)\s*', '', clean_line, flags=re.IGNORECASE)
                    if clean_line:
                        cleaned_lines.append(clean_line)
                
                final_cleaned_segment = "\n".join(cleaned_lines).strip()
                
                if final_cleaned_segment:
                    all_results.append(final_cleaned_segment)
                    previous_segment_memory = final_cleaned_segment
                else:
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
