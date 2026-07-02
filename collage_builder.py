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
                "images": ("IMAGE",),
                "frame_interval": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}), 
                "max_total_frames": ("INT", {"default": 25, "min": 5, "max": 1000, "step": 5}),
                "thumbnail_size": ("INT", {"default": 384, "min": 128, "max": 768, "step": 64}), 
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("combined_image",)
    FUNCTION = "build_collage"
    CATEGORY = "Local_LLM_Tools/Processors"

    def build_collage(self, images, frame_interval, max_total_frames, thumbnail_size):
        image_shape = list(images.shape)
        total_frames = int(image_shape[0])
        
        raw_indices = np.arange(0, total_frames, frame_interval)
        if len(raw_indices) > max_total_frames:
            indices = np.linspace(0, len(raw_indices) - 1, max_total_frames, dtype=int)
            indices = raw_indices[indices]
        else:
            indices = raw_indices

        pil_images = []
        for idx in indices:
            img_tensor = images[idx]
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_pil.thumbnail((thumbnail_size, thumbnail_size))
            pil_images.append(img_pil)

        if not pil_images:
            img_np = (images[0].cpu().numpy() * 255).astype(np.uint8)
            pil_images.append(Image.fromarray(img_np))

        cols = 5  
        num_frames = len(pil_images)
        rows = int(np.ceil(num_frames / cols))  

        widths, heights = zip(*(i.size for i in pil_images))
        cell_w = max(widths)
        cell_h = max(heights)

        grid_width = cell_w * cols
        grid_height = cell_h * rows
        collage_pil = Image.new('RGB', (grid_width, grid_height), color=(0, 0, 0))

        for i, im in enumerate(pil_images):
            r = i // cols
            c = i % cols
            x = c * cell_w + (cell_w - im.size[0]) // 2  
            y = r * cell_h + (cell_h - im.size[1]) // 2
            collage_pil.paste(im, (x, y))

        out_np = np.array(collage_pil).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(out_np).unsqueeze(0)  # 输出标准的 4 维 [1, H, W, 3]

        return (out_tensor,)
