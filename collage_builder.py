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
                "images": ("IMAGE",),  # 接收来自 VHS Load Video 的图像批次
                "frame_interval": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}), # 抽帧间隔
                
                # 【全自由 M × N 核心 Widget 升级】
                "cols": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}), # 允许自由指定横向有多少列
                "rows": ("INT", {"default": 3, "min": 1, "max": 20, "step": 1}), # 允许自由指定纵向有多少行
                
                "thumbnail_size": ("INT", {"default": 384, "min": 128, "max": 768, "step": 64}), # 单张子图分辨率
                "segment_mode": (["All_Segments_Batch", "Single_Specific_Segment"], {"default": "All_Segments_Batch"}), 
                "target_segment_index": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}), 
            }
        }

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("combined_images_batch", "total_segments",)
    FUNCTION = "build_segmented_collage"
    CATEGORY = "Local_LLM_Tools/Processors"

    def build_segmented_collage(self, images, frame_interval, cols, rows, thumbnail_size, segment_mode, target_segment_index):
        image_shape = list(images.shape)
        # 精准解包提取总输入帧数
        total_input_frames = int(image_shape[0])
        
        raw_indices = np.arange(0, total_input_frames, frame_interval)
        if len(raw_indices) == 0:
            raw_indices = np.array([0])
            
        # 根据前台设定的 M × N，自动计算出每一张大图里应该容纳多少个子格子上限
        max_frames_per_grid = cols * rows
        
        num_raw_frames = len(raw_indices)
        total_segments = int(np.ceil(num_raw_frames / max_frames_per_grid))
        
        segments_to_process = []
        if segment_mode == "Single_Specific_Segment":
            safe_idx = min(target_segment_index, total_segments - 1)
            segments_to_process.append(safe_idx)
        else:
            segments_to_process = list(range(total_segments))
            
        final_collage_tensors = []
        
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
                
            # 统一采用首帧特征测量，死锁网格 cell 的宽高
            widths, heights = zip(*(i.size for i in pil_images))
            cell_w = max(widths)
            cell_h = max(heights)
            
            # 【物理对齐核心】强行以用户指定的固定 cols 和 rows 建立大画布，绝对恒定
            grid_width = cell_w * cols
            grid_height = cell_h * rows
            collage_pil = Image.new('RGB', (grid_width, grid_height), color=(0, 0, 0))
            
            # 有条不紊地贴入自由定义的 M × N 空间中。填不满的格子自动留黑背景，完美契合多图 Batch stack 格式
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
            
        print(f"[自由宫格矩阵] 视频处理完毕。总分段数: {total_segments}，已完美构筑 {cols}列 × {rows}行 自由切片画布。实际输出 Batch 维度: {final_batch.shape}")
        return (final_batch, total_segments,)
