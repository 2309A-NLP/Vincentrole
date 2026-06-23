# 数字人智能对话系统 - Linly-Talker — “数字人交互，与虚拟的自己互动”

<div align="center">
<h1>Linly-Talker WebUI</h1>


[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange)](https://github.com/Kedreamix/Linly-Talker)

<img src="https://github.com/Kedreamix/Linly-Talker/raw/main/docs/linly_logo.png" /><br>

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/Kedreamix/Linly-Talker/blob/main/colab_webui.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/Kedreamix/Linly-Talker/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Models%20Repo-yellow.svg?style=for-the-badge)](https://huggingface.co/Kedreamix/Linly-Talker)

[**English**](https://github.com/Kedreamix/Linly-Talker/blob/main/README.md) | [**中文简体**](https://github.com/Kedreamix/Linly-Talker/blob/main/README_zh.md)

</div>


**2023.12 更新** 📆

**用户可以上传任意图片进行对话**

**2024.01 更新** 📆

- **令人兴奋的消息！我现在已经将强大的GeminiPro和Qwen大模型融入到我们的对话场景中。用户现在可以在对话中上传任何图片，为我们的互动增添了全新的层面。**
- **更新了FastAPI的部署调用方法。** 
- **更新了微软TTS的高级设置选项，增加声音种类的多样性，以及加入视频字幕加强可视化。**
- **更新了GPT多轮对话系统，使得对话有上下文联系，提高数字人的交互性和真实感。**

**2024.02 更新** 📆

- **更新了Gradio的版本为最新版本4.16.0，使得界面拥有更多的功能，比如可以摄像头拍摄图片构建数字人等。**
- **更新了ASR和THG，其中ASR加入了阿里的FunASR，具体更快的速度；THG部分加入了Wav2Lip模型，ER-NeRF在准备中(Comming Soon)。**
- **加入了语音克隆方法GPT-SoVITS模型，能够通过微调一分钟对应人的语料进行克隆，效果还是相当不错的，值得推荐。**
- **集成一个WebUI界面，能够更好的运行Linly-Talker。**

**2024.04 更新** 📆

- **更新了除 Edge TTS的 Paddle TTS的离线方式。**
- **更新了ER-NeRF作为Avatar生成的选择之一。**
- **更新了app_talk.py，在不基于对话场景可自由上传语音和图片视频生成。**

---

<details>
<summary>目录</summary>


<!-- TOC -->

- [数字人对话系统 - Linly-Talker —— “数字人交互，与虚拟的自己互动”](#数字人对话系统---linly-talker--数字人交互与虚拟的自己互动)
  - [介绍](#介绍)
  - [TO DO LIST](#to-do-list)
  - [示例](#示例)
  - [创建环境](#创建环境)
  - [ASR - Speech Recognition](#asr---speech-recognition)
    - [Whisper](#whisper)
    - [FunASR](#funasr)
  - [TTS - Edge TTS](#tts---edge-tts)
  - [Voice Clone](#voice-clone)
    - [GPT-SoVITS（推荐）](#gpt-sovits推荐)
    - [XTTS](#xtts)
  - [THG - Avatar](#thg---avatar)
    - [SadTalker](#sadtalker)
    - [Wav2Lip](#wav2lip)
    - [ER-NeRF（Comming Soon）](#er-nerfcomming-soon)
  - [LLM - Conversation](#llm---conversation)
    - [Linly-AI](#linly-ai)
    - [Qwen](#qwen)
    - [Gemini-Pro](#gemini-pro)
    - [LLM 多模型选择](#llm-多模型选择)
  - [优化](#优化)
  - [Gradio](#gradio)
  - [启动WebUI](#启动webui)
  - [文件夹结构](#文件夹结构)
  - [参考](#参考)
  - [Star History](#star-history)

<!-- /TOC -->

</details>



## 介绍

Linly-Talker是一款创新的数字人对话系统，它融合了最新的人工智能技术，包括大型语言模型（LLM）、自动语音识别（ASR）、文本到语音转换（TTS）和语音克隆技术。这个系统通过Gradio平台提供了一个交互式的Web界面，允许用户上传图片与AI进行个性化的对话交流。

系统的核心特点包括：

1. **多模型集成**：Linly-Talker整合了Linly、GeminiPro、Qwen等大模型，以及Whisper、SadTalker等视觉模型，实现了高质量的对话和视觉生成。
2. **多轮对话能力**：通过GPT模型的多轮对话系统，Linly-Talker能够理解并维持上下文相关的连贯对话，极大地提升了交互的真实感。
3. **语音克隆**：利用GPT-SoVITS等技术，用户可以上传一分钟的语音样本进行微调，系统将克隆用户的声音，使得数字人能够以用户的声音进行对话。
4. **实时互动**：系统支持实时语音识别和视频字幕，使得用户可以通过语音与数字人进行自然的交流。
5. **视觉增强**：通过数字人生成等技术，Linly-Talker能够生成逼真的数字人形象，提供更加沉浸式的体验。

Linly-Talker的设计理念是创造一种全新的人机交互方式，不仅仅是简单的问答，而是通过高度集成的技术，提供一个能够理解、响应并模拟人类交流的智能数字人。

![The system architecture of multimodal human–computer interaction.](https://github.com/Kedreamix/Linly-Talker/raw/main/docs/HOI.png)

> 查看我们的介绍视频 [demo video](https://www.bilibili.com/video/BV1rN4y1a76x/)
>
> 在B站上我录了一系列视频，也代表我更新的每一步与使用方法，详细查看[数字人智能对话系统 - Linly-Talker合集](https://space.bilibili.com/241286257/channel/collectiondetail?sid=2065753)
>
> -  [🔥🔥🔥数字人对话系统 Linly-Talker🔥🔥🔥](https://www.bilibili.com/video/BV1rN4y1a76x/)
> - [🚀数字人的未来：Linly-Talker+GPT-SoVIT语音克隆技术的赋能之道](https://www.bilibili.com/video/BV1S4421A7gh/)
> - [AutoDL平台部署Linly-Talker (0基础小白超详细教程)](https://www.bilibili.com/video/BV1uT421m74z/)
> - [Linly-Talker 更新离线TTS集成及定制数字人方案](https://www.bilibili.com/video/BV1Mr421u7NN/)

## TO DO LIST

- [x] 基本完成对话系统流程，能够`语音对话`
- [x] 加入了LLM大模型，包括`Linly`，`Qwen`和`GeminiPro`的使用
- [x] 可上传`任意数字人照片`进行对话
- [x] Linly加入`FastAPI`调用方式
- [x] 利用微软`TTS`加入高级选项，可设置对应人声以及音调等参数，增加声音的多样性
- [x] 视频生成加入`字幕`，能够更好的进行可视化
- [x] GPT`多轮对话`系统（提高数字人的交互性和真实感，增强数字人的智能）
- [x] 优化Gradio界面，加入更多模型，如Wav2Lip，FunASR等
- [x] `语音克隆`技术，加入GPT-SoVITS，只需要一分钟的语音简单微调即可（语音克隆合成自己声音，提高数字人分身的真实感和互动体验）
- [x] 加入离线TTS以及NeRF-based的方法和模型
- [ ] `实时`语音识别（人与数字人之间就可以通过语音进行对话交流)

🔆 该项目 Linly-Talker 正在进行中 - 欢迎提出PR请求！如果您有任何关于新的模型方法、研究、技术或发现运行错误的建议，请随时编辑并提交 PR。您也可以打开一个问题或通过电子邮件直接联系我。📩⭐ 如果您发现这个Github Project有用，请给它点个星！🤩

> 如果在部署的时候有任何的问题，可以关注[常见问题汇总.md](https://github.com/Kedreamix/Linly-Talker/blob/main/常见问题汇总.md)部分，我已经整理了可能出现的所有问题，另外交流群也在这里，我会定时更新，感谢大家的关注与使用！！！

###### 模型文件和权重，请浏览“模型文件”页面获取。


接下来还需要安装对应的模型，有以下下载方式，下载后安装文件架结构放置，文件夹结构在本文最后有说明。

- [Baidu (百度云盘)](https://pan.baidu.com/s/1eF13O-8wyw4B3MtesctQyg?pwd=linl) (Password: `linl`)
- [huggingface](https://huggingface.co/Kedreamix/Linly-Talker)
- [modelscope](https://www.modelscope.cn/models/Kedreamix/Linly-Talker/summary) 

**HuggingFace下载**

如果速度太慢可以考虑镜像，参考[简便快捷获取 Hugging Face 模型（使用镜像站点）](https://kedreamix.github.io/2024/01/05/Note/HuggingFace/?highlight=镜像)

```bash
# 从huggingface下载预训练模型
git lfs install
git clone https://huggingface.co/Kedreamix/Linly-Talker
```

**ModelScope下载**

```bash
# 从modelscope下载预训练模型
# 1. git 方法
git lfs install
git clone https://www.modelscope.cn/Kedreamix/Linly-Talker.git

# 2. Python 代码下载
pip install modelscope
from modelscope import snapshot_download
model_dir = snapshot_download('Kedreamix/Linly-Talker')
```

**移动所有模型到当前目录**

如果百度网盘下载后，可以参考文档最后目录结构来移动

```bash
# 移动所有模型到当前目录
# checkpoint中含有SadTalker和Wav2Lip
mv Linly-Talker/checkpoints/* ./checkpoints

# SadTalker的增强GFPGAN
# pip install gfpgan
# mv Linly-Talker/gfpan ./

# 语音克隆模型
mv Linly-Talker/GPT_SoVITS/pretrained_models/* ./GPT_SoVITS/pretrained_models/

# Qwen大模型
mv Linly-Talker/Qwen ./
```

为了大家的部署使用方便，更新了一个`configs.py`文件，可以对其进行一些超参数修改即可

```bash
# 设备运行端口 (Device running port)
port = 7860
# api运行端口及IP (API running port and IP)
mode = 'api' # api 需要先运行Linly-api-fast.py，暂时仅仅适用于Linly

# 本地端口localhost:127.0.0.1 全局端口转发:"0.0.0.0"
ip = '127.0.0.1' 
api_port = 7871

# L模型路径 (Linly model path)
mode = 'offline'
model_path = 'Qwen/Qwen-1_8B-Chat'

# ssl证书 (SSL certificate) 麦克风对话需要此参数
# 最好调整为绝对路径
ssl_certfile = "./https_cert/cert.pem"
ssl_keyfile = "./https_cert/key.pem"
```

## 启动WebUI

之前我将很多个版本都是分开来的，实际上运行多个会比较麻烦，所以后续我增加了变成WebUI一个界面即可体验，后续也会不断更新

现在已加入WebUI的功能如下

- [x] 文本/语音数字人对话（固定数字人，分男女角色）
- [x] 任意图片数字人对话（可上传任意数字人）
- [x] 多轮GPT对话（加入历史对话数据，链接上下文）
- [x] 语音克隆对话（基于GPT-SoVITS设置进行语音克隆，内置烟嗓音，可根据语音对话的声音进行克隆）

```bash
# WebUI
python webui.py
```

![](https://github.com/Kedreamix/Linly-Talker/raw/main/docs/WebUI.png)



现在的启动一共有几种模式，可以选择特定的场景进行设置

第一种只有固定了人物问答，设置好了人物，省去了预处理时间

```bash
python app.py
```

![](https://github.com/Kedreamix/Linly-Talker/raw/main/docs/UI.png)

最近更新了第一种模式，加入了Wav2Lip模型进行对话

```bash
python appv2.py
```

第二种是可以任意上传图片进行对话

```bash
python app_img.py
```

![](https://github.com/Kedreamix/Linly-Talker/raw/main/docs/UI2.png)

第三种是在第一种的基础上加入了大语言模型，加入了多轮的GPT对话

```bash
python app_multi.py
```

![](https://github.com/Kedreamix/Linly-Talker/raw/main/docs/UI3.png)

现在加入了语音克隆的部分，可以自由切换自己克隆的声音模型和对应的人图片进行实现，这里我选择了一个烟嗓音和男生图片

```bash
python app_vits.py
```

加入了第四种方式，不固定场景进行对话，直接输入语音或者生成语音进行数字人生成，内置了Sadtalker，Wav2Lip，ER-NeRF等方式

> ER-NeRF是针对单独一个人的视频进行训练的，所以需要替换特定的模型才能进行渲染得到正确的结果，内置了Obama的权重，可直接用

```bash
python app_talk.py
```

![](https://github.com/Kedreamix/Linly-Talker/raw/main/docs/UI4.png)

## 文件夹结构

所有的权重部分可以从这下载

- [Baidu (百度云盘)](https://pan.baidu.com/s/1eF13O-8wyw4B3MtesctQyg?pwd=linl) (Password: `linl`)
- [huggingface](https://huggingface.co/Kedreamix/Linly-Talker)
- [modelscope](https://www.modelscope.cn/models/Kedreamix/Linly-Talker/files) comming soon

权重文件夹结构如下

```bash
Linly-Talker/ 
├── checkpoints
│   ├── hub
│   │   └── checkpoints
│   │       └── s3fd-619a316812.pth
│   ├── lipsync_expert.pth
│   ├── mapping_00109-model.pth.tar
│   ├── mapping_00229-model.pth.tar
│   ├── SadTalker_V0.0.2_256.safetensors
│   ├── visual_quality_disc.pth
│   ├── wav2lip_gan.pth
│   └── wav2lip.pth
├── gfpgan
│   └── weights
│       ├── alignment_WFLW_4HG.pth
│       └── detection_Resnet50_Final.pth
├── GPT_SoVITS
│   └── pretrained_models
│       ├── chinese-hubert-base
│       │   ├── config.json
│       │   ├── preprocessor_config.json
│       │   └── pytorch_model.bin
│       ├── chinese-roberta-wwm-ext-large
│       │   ├── config.json
│       │   ├── pytorch_model.bin
│       │   └── tokenizer.json
│       ├── README.md
│       ├── s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
│       ├── s2D488k.pth
│       ├── s2G488k.pth
│       └── speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
├── Qwen
│   └── Qwen-1_8B-Chat
│       ├── assets
│       │   ├── logo.jpg
│       │   ├── qwen_tokenizer.png
│       │   ├── react_showcase_001.png
│       │   ├── react_showcase_002.png
│       │   └── wechat.png
│       ├── cache_autogptq_cuda_256.cpp
│       ├── cache_autogptq_cuda_kernel_256.cu
│       ├── config.json
│       ├── configuration_qwen.py
│       ├── cpp_kernels.py
│       ├── examples
│       │   └── react_prompt.md
│       ├── generation_config.json
│       ├── LICENSE
│       ├── model-00001-of-00002.safetensors
│       ├── model-00002-of-00002.safetensors
│       ├── modeling_qwen.py
│       ├── model.safetensors.index.json
│       ├── NOTICE
│       ├── qwen_generation_utils.py
│       ├── qwen.tiktoken
│       ├── README.md
│       ├── tokenization_qwen.py
│       └── tokenizer_config.json
└── README.md
```


