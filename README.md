## ⚠️ 观前提示！！！！
目前仅尝试ollama转发成功，本地gguf（需要大脑和眼睛-mmproj）和 本地safetensors 暂未尝试


## 🛠️ 安装方法（一切环境安装前需要自行判断环境是否缺失）
1. 将本插件文件夹克隆或直接下载解压至你的 `ComfyUI/custom_nodes/` 目录下。
2. 确保你的 Python 环境中安装了核心多模态依赖。如果是 Windows 便携整合包，请在 ComfyUI 根目录下打开终端执行：
   ```powershell
   .\python\python.exe -m pip install transformers accelerate sentencepiece Pillow numpy torch
   ```
3. **重要（仅针对使用本地 GGUF 后端的用户）**：
   若要完美吃满 RTX 5090 显卡加速，请务必带上 CUDA 编译参数手动安装或重新编译 `llama-cpp-python`：
   ```powershell
   \$env:CMAKE_ARGS="-GGUIDE -DGGML_CUDA=on"; .\python\python.exe -m pip install llama-cpp-python --force-reinstall --no-cache-dir
   ```
4. 重启 ComfyUI。

连接图示：
<img width="1339" height="284" alt="image" src="https://github.com/user-attachments/assets/b87e41c8-5c20-4dcf-9598-8b0716902f91" />


## 🛠️ 使用节点说明
1.Video Frame Collage Builder (Adaptive & Segmented)
作用：从加载的视频中传入图像，并按5*5（25）或6*6（36）进行循环拼接
<img width="239" height="273" alt="image" src="https://github.com/user-attachments/assets/0d7131a2-fbe6-44d5-ba9d-d6fcf04ce14c" />

1.0 frame_interval：每隔*帧抽取一帧。

1.1 segment_mode（分段处理模式开关）：All_Segments_Batch（全自动批次模式-默认）：它会自动把长视频按设置的 max_frames_per_grid: 25 分割成若干个正方形故事板大图，如果一共有75帧，它会一口气做成3张正方形大图组成一个批次（Batch），通过 combined_images_batch 端口打包全部传给下游。
适用场景：日常全自动化生产流。
Single_Specific_Segment（指定单一片段模式）：它不再输出多图批次，而是变成只输出单张图片。视频再长，它也只会去切你指定的那一小段。适用场景：调试工作流、或者做精准画面反推。比如你想单独看长视频中某一秒的跳舞动作，或者生图显存不够、不想让大模型一次性跑完一整段长视频，就切换到这个模式。

1.2 target_segment_index（目标片段索引序号）这个参数只有在 segment_mode 切换为 Single_Specific_Segment 时才会生效。
作用：用来指定想单独看视频的第几段。填 0：代表切出视频的第 1 段（比如第 0~25 帧构成的那个 5x5 正方形）。填 1：代表切出视频的第 2 段（第 25~50 帧构成的正方形）。填 2：代表切出视频的第 3 段（第 50~75 帧），以此类推。
注：如果你填的数字（比如填了 10）超过了视频的总分段数，代码内部已经写了安全防御，会自动卡死在最后一段，绝对不会报错崩溃。

1.3 输出端口 total_segments（总分段数）：它是节点根据你视频的总长度以及 max_frames_per_grid 自动计算出来的“总段数”数字。具体用处：日常观察：你可以从这个紫色的 total_segments 端口向右拉出一根线，在画布空白处放一个普通的 ShowText（展示文本） 节点（或者数字显示节点），点击生成后，它就会直接在画布上显示出一个数字（比如 3 或 5），让你心里有个底：“哦，我这个视频被切成了 3 段”。高阶联动（可选）：你以后如果需要编写更复杂的自定义节点（比如用 ComfyUI 原生的 Loop 循环去重复执行某个任务），可以用这个数字作为循环的总上限参数。

2.Ultimate Multimodal Tagger (Ollama/GGUF/HF-FP8)
作用：接收拼接好的图像，传入ollama/本地模型进行反推
可选形式：本地ollama转发、本地gguf（需要大脑和眼睛-mmproj）、本地safetensors
注：描述提示词需要按所选模型自行更改
<img width="270" height="199" alt="image" src="https://github.com/user-attachments/assets/a12f0ac2-9ee5-4e6f-9a66-cc10d80a32a3" />

3.Video Text Prompt Concatenator (Auto-Append)
<img width="254" height="149" alt="image" src="https://github.com/user-attachments/assets/92ca9fa6-f69a-45bf-a85f-9daaf95dee05" />

3.1 join_delimiter（衔接分隔符）：决定每两段分镜描述之间，用什么符号来隔开。
选项代表的含义：\n\n（当前默认-最推荐）：代表换行两次（即空一行）。每一段描述会变成一个独立的段落，层次分明，最适合你日常在 Show Text 框里肉眼阅读，或者喂给支持分段解析的国产长视频大模型。
 （空格）：把所有段落首尾相连成一大堆密密麻麻的文字。
, （逗号）：用逗号衔接，适合纯 Tag 标签流。
|（竖线）：用竖线隔开，适合某些特定视频模型的权重切分。

3.2 prefix_text（全局总前缀）：在整篇长剧本的最开头，强行插入一段你自定义的固定文字。
用法：比如你正在跑的都是电影写实流视频，你可以直接在这里填入：masterpiece, best quality, ultra-photorealistic, 8k resolution, cinematic lighting，
这样，不管后面大模型吐出多少段舞蹈描述，最终输出的最开头，永远会稳稳地带着这串顶级画质修饰词。

3.3 suffix_text（全局总后缀）：在整篇长剧本的最末尾，强行追加一段你自定义的固定文字。
用法：比如你希望视频生成的结尾都是淡出或者特定的负面绕开词，可以填入：, smooth motion, flawless choreography, hyper detailed.。它会被全自动挂在全文最后。

3.4 add_segment_index_tag（场景编号自动标记）：控制是否要在每一段文本前，自动打上类似时间轴的分镜标签。
enable（当前配置-最推荐）：开启标记，最终吐出的总提示词会变得极度有条理：[Segment 1]: A young woman begins dancing...[Segment 2]: The woman continues with fluid arm waves...disable（关闭标记）：关闭后，它会把打碎的描述直接揉成一坨纯文本，不再带有 [Segment X] 的前缀。如果你的下游视频模型（比如早期版本的 SVD）不认识这个中括号前缀、或者会因为这个前缀产生文字幻觉，就切到 disable。
