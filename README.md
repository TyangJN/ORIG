# ORIG - Multi-Modal Retrieval-Enhanced Image Generation Evaluation System

ORIG (Optimized Retrieval for Image Generation) is a multi-modal image generation evaluation system based on retrieval augmentation, supporting various combinations of retrieval models and generation models for assessing and comparing image generation quality across different modalities.

## 🎯 Project Overview

This project implements a complete retrieval-enhanced image generation pipeline with the following core features:

- **Multi-Modal Retrieval**: Supports text retrieval and image retrieval to enhance prompt accuracy
- **Multi-Model Support**: Integrates GPT and Qwen retrieval models with various generation models
- **Parallel Processing**: Uses multi-threading for parallel processing to improve efficiency
- **Comprehensive Evaluation**: Provides detailed image quality assessment and comparative analysis

## 🏗️ System Architecture

```
ORIG/
├── data/                    # Data directory
│   ├── FIG-Eval/           # Evaluation dataset
│   ├── gpt_based_search/   # GPT retrieval results
│   └── qwen_based_search/  # Qwen retrieval results
├── eval/                   # Evaluation module
│   ├── evaluation_single_modal.py  # Single modal evaluator
│   └── reference/          # Reference data and images
├── gpt_retrieval/          # GPT retrieval module
├── qwen_retrieval/         # Qwen retrieval module
├── utils/                  # Utility functions
├── main_mp.py             # GPT retrieval main program
├── main_qwen_mp.py        # Qwen retrieval main program
└── pipeline.py            # Core pipeline
```

## 🚀 Quick Start

### Requirements

- Python 3.8+
- Required dependencies (see requirements.txt)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure API Keys

Before using, ensure you have configured the necessary API keys:

```python
# Set in relevant files
OPENAI_API_KEY = "your_openai_api_key"
```

## 📖 Usage

### 1. GPT Retrieval Pipeline

```bash
python main_mp.py \
    --search_model gpt \
    --gen_model openai_gen \
    --dataset data/FIG-Eval/prompts.jsonl \
    --meta_path data \
    --max_rounds 3 \
    --modality mm
```

### 2. Qwen Retrieval Pipeline

```bash
python main_qwen_mp.py \
    --search_model qwen \
    --gen_model qwen_gen \
    --dataset data/FIG-Eval/prompts.jsonl \
    --meta_path data \
    --max_rounds 3 \
    --modality mm
```

### Parameter Description

- `--search_model`: Retrieval model, options: "gpt" or "qwen"
- `--gen_model`: Generation model, options: "gemini_gen", "openai_gen", "qwen_gen"
- `--dataset`: Dataset path
- `--meta_path`: Result save path
- `--max_rounds`: Maximum number of retrieval rounds
- `--modality`: Task type, options: "mm", "txt", "img", "cot"

## 🎭 Supported Modalities

The system supports the following five modalities:

1. **mm** (Multimodal): Multi-modal retrieval enhancement
2. **txt** (Text): Pure text retrieval enhancement
3. **img** (Image): Image retrieval enhancement
4. **cot** (Chain of Thought): Chain of thought retrieval enhancement
5. **dir** (Direct): Direct generation (no retrieval enhancement)

## 📊 Evaluation System

### Single Modal Evaluation

```bash
python eval/evaluation_single_modal.py
```

The evaluator supports:
- Multi-threaded parallel evaluation
- Detailed evaluation report generation
- Support for case-variant image paths
- Statistical information summary

### Evaluation Metrics

- Image quality scoring
- Similarity with reference images
- Multi-dimensional evaluation results

## 📁 Data Structure

### Input Data Format

```json
{
    "id": "animal_0",
    "category": "animal", 
    "prompt_zh": "生成青蛙生命周期的照片",
    "prompt_en": "Generate a photo of frog lifecircle"
}
```

### Output Data Format

Retrieval results include:
- Retrieval plans and queries
- Retrieval result summaries
- Enhanced generation prompts

## 🔧 Core Components

### Retrieval Module

- **warm_up_search**: Warm-up retrieval to identify key entities
- **loop_search**: Iterative retrieval to gradually improve information
- **content_refine**: Content refinement to generate final prompts

### Generation Module

- Support for multiple generation models
- Automatic prompt optimization
- Batch image generation

### Evaluation Module

- Single modal quality evaluation
- Multi-dimensional scoring system
- Detailed result analysis

## 📈 Performance Optimization

- **Parallel Processing**: Uses ThreadPoolExecutor for multi-threaded parallel processing
- **Result Caching**: Supports checkpoint resumption to avoid redundant computation
- **Memory Optimization**: Batch processing for large datasets
- **Error Handling**: Comprehensive exception handling and logging

## 🎨 Dataset

The project includes the FIG-Eval dataset covering the following categories:

- **Animal**: Animal-related image generation
- **Culture**: Cultural element image generation
- **Event**: Historical event scene generation
- **Food**: Food image generation
- **Landmarks**: Famous landmark generation
- **People**: Portrait generation
- **Plant**: Plant image generation
- **Product**: Product image generation
- **Sports**: Sports scene generation
- **Transportation**: Vehicle generation

## 🔍 Retrieval Strategy

### Text Retrieval
- Entity definitions and characteristic descriptions
- Historical background and cultural meanings
- Process steps and stage information

### Image Retrieval
- Visual style and appearance features
- Material textures and spatial layouts
- Specific examples and reference images

## 📝 Usage Examples

### Basic Usage Workflow

1. **Prepare Data**: Place prompt data in `data/FIG-Eval/` directory
2. **Run Retrieval**: Select retrieval model and generation model
3. **Generate Images**: System automatically generates enhanced prompts and creates images
4. **Evaluate Results**: Use evaluator to analyze generation quality

### Advanced Configuration

```python
# Custom retrieval rounds
max_rounds = 5

# Custom thread count
max_workers = 16

# Custom batch size
batch_size = 50
```

## 🔄 Retrieval Process

### Warm-up Phase
1. Identify key entities and concepts from input prompts
2. Generate 1-3 sub-questions for clarification
3. Create corresponding search queries

### Iterative Phase
1. Analyze missing visual/semantic information
2. Plan text and image retrievals strategically
3. Execute searches and refine results
4. Continue until sufficient information is gathered

### Content Refinement
1. Synthesize all retrieved information
2. Generate enhanced prompts for image generation
3. Optimize prompt structure and clarity

## 🎯 Key Features

### Multi-Modal Retrieval
- **Text Retrieval**: For factual information, definitions, and context
- **Image Retrieval**: For visual references, styles, and layouts
- **Hybrid Approach**: Combines both modalities for comprehensive enhancement

### Flexible Model Support
- **Retrieval Models**: GPT-5, Qwen
- **Generation Models**: GPT-Image, Gemini-Image, Qwen-Image
- **Easy Extension**: Modular design for adding new models

### Comprehensive Evaluation
- **Quality Metrics**: Multiple evaluation dimensions
- **Reference Comparison**: Against ground truth images
- **Statistical Analysis**: Detailed performance reports

## 🛠️ Development

### Adding New Retrieval Models

1. Create new module in `{model}_retrieval/` directory
2. Implement required interface methods
3. Update main pipeline to support new model

### Adding New Generation Models

1. Extend generation module with new model support
2. Implement model-specific prompt formatting
3. Add configuration options


## 🤝 Contributing

We welcome contributions to improve the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Contribution Guidelines

- Follow existing code style
- Add tests for new features
- Update documentation as needed
- Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License.

## 📞 Contact

For questions or suggestions, please contact us through:

- Submit GitHub Issues
- Send email to project maintainers

## 🙏 Acknowledgments

- Thanks to the open-source community for various tools and libraries
- Special thanks to contributors who have helped improve the project

---

**Note**: Please ensure all necessary API keys and dependencies are properly configured before use.
