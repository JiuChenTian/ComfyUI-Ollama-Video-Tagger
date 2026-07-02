## 🛠️ 安装方法
1. 将本插件文件夹克隆或直接下载解压至你的 `ComfyUI/custom_nodes/` 目录下。
2. 确保你的 Python 环境中安装了核心多模态依赖。如果是 Windows 便携整合包，请在 ComfyUI 根目录下打开终端执行：
   ```powershell
   .\python_embeded\python.exe -m pip install transformers accelerate sentencepiece Pillow numpy torch
   ```
3. **重要（仅针对使用本地 GGUF 后端的用户）**：
   若要完美吃满 RTX 5090 显卡加速，请务必带上 CUDA 编译参数手动安装或重新编译 `llama-cpp-python`：
   ```powershell
   \$env:CMAKE_ARGS="-GGUIDE -DGGML_CUDA=on"; .\python_embeded\python.exe -m pip install llama-cpp-python --force-reinstall --no-cache-dir
   ```
4. 重启 ComfyUI。

连接图示：
<img width="1000" height="207" alt="image" src="https://github.com/user-attachments/assets/dff42c1f-96b8-48cc-949b-6ce8ed2eaeda" />

## 🛠️ 使用节点说明
1.Video Frame Collage Builder (5xN Grid)
作用：从加载的视频中传入图像，并按5*N进行拼接（最大拼接1000张，即5*200）
frame_interval：每隔*帧抽取一帧。

2.Ultimate Multimodal Tagger (Ollama/GGUF/HF-FP8)
作用：接收拼接好的图像，传入ollama/本地模型进行反推
可选形式：本地ollama转发、本地gguf（需要大脑和眼睛-mmproj）、本地.safetensors
注：提示词需要人工按所选模型自行更改
