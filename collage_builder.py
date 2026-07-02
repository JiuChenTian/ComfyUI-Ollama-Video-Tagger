import torch
import numpy as np
from PIL import Image

class VideoFrameCollageBuilder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 接收来自 Load Video 的图像批次
                "frame_interval": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}), 
                "max_frames_per_grid": ("INT", {"default": 25, "min": 4, "max": 144, "step": 1}), # 每张图的子图上限
                "thumbnail_size": ("INT", {"default": 384, "min": 128, "max": 768, "step": 64}), 
                "segment_mode": (["All_Segments_Batch", "Single_Specific_Segment"], {"default": "All_Segments_Batch"}), 
                "target_segment_index": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}), 
            }
        }

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("combined_images_batch", "total_segments",)
    FUNCTION = "build_segmented_collage"
    CATEGORY = "Local_LLM_Tools/Processors"

    def build_segmented_collage(self, images, frame_interval, max_frames_per_grid, thumbnail_size, segment_mode, target_segment_index):
        image_shape = list(images.shape)
        total_input_frames = int(image_shape[0])
        
        raw_indices = np.arange(0, total_input_frames, frame_interval)
        if len(raw_indices) == 0:
            raw_indices = np.array([0])
            
        num_raw_frames = len(raw_indices)
        total_segments = int(np.ceil(num_raw_frames / max_frames_per_grid))
        
        segments_to_process = []
        if segment_mode == "Single_Specific_Segment":
            safe_idx = min(target_segment_index, total_segments - 1)
            segments_to_process.append(safe_idx)
        else:
            segments_to_process = list(range(total_segments))
            
        final_collage_tensors = []
        
        # 【核心修正准备】：不论最后一帧剩下多少，整张网格的网格间距（列数/行数）必须以配置的 max_frames_per_grid 为硬标准死锁！
        # 这样可以确保每个 Segment 拼出来的大画布尺寸绝对恒定
        grid_size = int(np.ceil(np.sqrt(max_frames_per_grid)))
        cols = grid_size
        rows = grid_size
        
        for seg_idx in segments_to_process:
            start_f = seg_idx * max_frames_per_grid
            end_f = min(start_f + max_frames_per_grid, num_raw_frames)
            seg_indices = raw_indices[start_f:end_f]
            
            pil_images = []
            for idx in seg_indices:
                img_tensor = images[idx]
                img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                img_pil.thumbnail((thumbnail_size, thumbnail_size))
                pil_images.append(img_pil)
                
            if not pil_images:
                continue
                
            # 统一采用死锁的 cell 宽高，根据首帧来做特征测量
            widths, heights = zip(*(i.size for i in pil_images))
            cell_w = max(widths)
            cell_h = max(heights)
            
            # 建立物理尺寸绝对相等的巨型黑底画布
            grid_width = cell_w * cols
            grid_height = cell_h * rows
            collage_pil = Image.new('RGB', (grid_width, grid_height), color=(0, 0, 0))
            
            # 贴图循环，即便后面格子填不满，也是在相等尺寸的黑底画布上填，完美解决了 entry 尺寸不一致痛点！
            for i, im in enumerate(pil_images):
                r = i // cols
                c = i % cols
                x = c * cell_w + (cell_w - im.size[0]) // 2
                y = r * cell_h + (cell_h - im.size[1]) // 2
                collage_pil.paste(im, (x, y))
                
            out_np = np.array(collage_pil).astype(np.float32) / 255.0
            out_tensor = torch.from_numpy(out_np)
            final_collage_tensors.append(out_tensor)
            
        if not final_collage_tensors:
            empty_np = np.zeros((thumbnail_size * rows, thumbnail_size * cols, 3), dtype=np.float32)
            final_batch = torch.from_numpy(empty_np).unsqueeze(0)
        else:
            final_batch = torch.stack(final_collage_tensors, dim=0)
            
        print(f"[拼图切片中枢] 视频处理完毕。总分段数: {total_segments}，每一分段尺寸完全对齐！本次实际输出大图数: {final_batch.shape}")
        return (final_batch, total_segments,)
