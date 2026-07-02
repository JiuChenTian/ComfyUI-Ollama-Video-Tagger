from .local_video_nodes import VideoFrameCollageBuilder, UltimateMultimodalTagger, VideoTextPromptConcatenator

NODE_CLASS_MAPPINGS = {
    "VideoFrameCollageBuilder": VideoFrameCollageBuilder,
    "UltimateMultimodalTagger": UltimateMultimodalTagger,
    "VideoTextPromptConcatenator": VideoTextPromptConcatenator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameCollageBuilder": "Video Frame Collage Builder (Adaptive & Segmented)",
    "UltimateMultimodalTagger": "Ultimate Multimodal Tagger (Ollama/GGUF/HF-FP8)",
    "VideoTextPromptConcatenator": "Video Text Prompt Concatenator (Auto-Append)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
