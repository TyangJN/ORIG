# ORIG - 多模态检索增强图像生成评估系统

ORIG (Optimized Retrieval for Image Generation) 是一个基于检索增强的多模态图像生成评估系统，支持多种检索模型和生成模型的组合，用于评估和比较不同模态下的图像生成质量。

## 🎯 项目概述

本项目实现了一个完整的检索增强图像生成流水线，包含以下核心功能：

- **多模态检索**: 支持文本检索和图像检索，增强生成提示的准确性
- **多模型支持**: 集成 GPT 和 Qwen 检索模型，支持多种生成模型
- **并行处理**: 使用多线程并行处理，提高处理效率
- **全面评估**: 提供详细的图像质量评估和比较分析

## 🏗️ 系统架构

```
ORIG/
├── data/                    # 数据目录
│   ├── FIG-Eval/           # 评估数据集
│   ├── gpt_based_search/   # GPT检索结果
│   └── qwen_based_search/  # Qwen检索结果
├── eval/                   # 评估模块
│   ├── evaluation_single_modal.py  # 单模态评估器
│   └── reference/          # 参考数据和图像
├── gpt_retrieval/          # GPT检索模块
├── qwen_retrieval/         # Qwen检索模块
├── utils/                  # 工具函数
├── main_mp.py             # GPT检索主程序
├── main_qwen_mp.py        # Qwen检索主程序
└── pipeline.py            # 核心流水线
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 必要的依赖包（见 requirements.txt）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 API 密钥

在使用前，请确保配置了相应的 API 密钥：

```python
# 在相关文件中设置
OPENAI_API_KEY = "your_openai_api_key"
```

## 📖 使用方法

### 1. GPT 检索流水线

```bash
python main_mp.py \
    --search_model gpt \
    --gen_model openai_gen \
    --dataset data/FIG-Eval/prompts.jsonl \
    --meta_path data \
    --max_rounds 3 \
    --modality mm
```

### 2. Qwen 检索流水线

```bash
python main_qwen_mp.py \
    --search_model qwen \
    --gen_model qwen_gen \
    --dataset data/FIG-Eval/prompts.jsonl \
    --meta_path data \
    --max_rounds 3 \
    --modality mm
```

### 参数说明

- `--search_model`: 检索模型，可选 "gpt" 或 "qwen"
- `--gen_model`: 生成模型，可选 "gemini_gen", "openai_gen", "qwen_gen"
- `--dataset`: 数据集路径
- `--meta_path`: 结果保存路径
- `--max_rounds`: 最大检索轮数
- `--modality`: 任务类型，可选 "mm", "txt", "img", "cot"

## 🎭 支持的模态

系统支持以下五种模态：

1. **mm** (Multimodal): 多模态检索增强
2. **txt** (Text): 纯文本检索增强
3. **img** (Image): 图像检索增强
4. **cot** (Chain of Thought): 思维链检索增强
5. **dir** (Direct): 直接生成（无检索增强）

## 📊 评估系统

### 单模态评估

```bash
python eval/evaluation_single_modal.py
```

评估器支持：
- 多线程并行评估
- 详细的评估报告生成
- 支持大小写变体的图像路径
- 统计信息汇总

### 评估指标

- 图像质量评分
- 与参考图像的相似度
- 多维度评估结果

## 📁 数据结构

### 输入数据格式

```json
{
    "id": "animal_0",
    "category": "animal", 
    "prompt_zh": "生成青蛙生命周期的照片",
    "prompt_en": "Generate a photo of frog lifecircle"
}
```

### 输出数据格式

检索结果包含：
- 检索计划和查询
- 检索结果摘要
- 增强后的生成提示

## 🔧 核心组件

### 检索模块

- **warm_up_search**: 预热检索，识别关键实体
- **loop_search**: 迭代检索，逐步完善信息
- **content_refine**: 内容精炼，生成最终提示

### 生成模块

- 支持多种生成模型
- 自动提示优化
- 批量图像生成

### 评估模块

- 单模态质量评估
- 多维度评分系统
- 详细结果分析

## 📈 性能优化

- **并行处理**: 使用 ThreadPoolExecutor 实现多线程并行
- **结果缓存**: 支持断点续传，避免重复计算
- **内存优化**: 分批处理大量数据
- **错误处理**: 完善的异常处理和日志记录

## 🎨 数据集

项目包含 FIG-Eval 数据集，涵盖以下类别：

- **动物** (animal): 动物相关图像生成
- **文化** (culture): 文化元素图像生成  
- **事件** (event): 历史事件场景生成
- **食物** (food): 美食图像生成
- **地标** (landmarks): 著名地标生成
- **人物** (people): 人物肖像生成
- **植物** (plant): 植物图像生成
- **产品** (product): 产品图像生成
- **运动** (sports): 运动场景生成
- **交通** (transportation): 交通工具生成

## 🔍 检索策略

### 文本检索
- 实体定义和特征描述
- 历史背景和文化含义
- 过程步骤和阶段信息

### 图像检索  
- 视觉风格和外观特征
- 材质纹理和空间布局
- 具体示例和参考图像

## 📝 使用示例

### 基本使用流程

1. **准备数据**: 将提示数据放入 `data/FIG-Eval/` 目录
2. **运行检索**: 选择检索模型和生成模型
3. **生成图像**: 系统自动生成增强提示并创建图像
4. **评估结果**: 使用评估器分析生成质量

### 高级配置

```python
# 自定义检索轮数
max_rounds = 5

# 自定义线程数
max_workers = 16

# 自定义批处理大小
batch_size = 50
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目。

## 📄 许可证

本项目采用 MIT 许可证。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 使用前请确保已正确配置所有必要的 API 密钥和依赖环境。