from .local_video_nodes import VideoFrameCollageBuilder, UltimateMultimodalTagger

NODE_CLASS_MAPPINGS = {
    "VideoFrameCollageBuilder": VideoFrameCollageBuilder,
    "UltimateMultimodalTagger": UltimateMultimodalTagger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameCollageBuilder": "Video Frame Collage Builder (5xN Grid)",
    "UltimateMultimodalTagger": "Ultimate Multimodal Tagger (Ollama/GGUF/HF-FP8)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
