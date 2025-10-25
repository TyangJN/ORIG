# Installation Guide

This guide provides detailed installation instructions for the ORIG project.

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (recommended: 8+ cores)
- **RAM**: Minimum 8GB, recommended 16GB+
- **GPU**: Optional but recommended for local Qwen model inference
  - NVIDIA GPU with CUDA support (for torch GPU acceleration)
  - Minimum 8GB VRAM for Qwen2.5-VL-7B model

### Software Requirements
- **Python**: 3.8 or higher (tested with Python 3.8, 3.9, 3.10, 3.11)
- **Operating System**: Linux, macOS, or Windows
- **CUDA**: 11.8 or higher (if using GPU acceleration)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ORIG
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv orig_env
source orig_env/bin/activate  # On Windows: orig_env\Scripts\activate

# Or using conda
conda create -n orig_env python=3.10
conda activate orig_env
```

### 3. Install Dependencies

#### Basic Installation
```bash
pip install -r requirements.txt
```

#### GPU Support (Optional)
If you have a CUDA-compatible GPU and want to use local Qwen models:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install flash-attention for faster inference (optional)
pip install flash-attn --no-build-isolation
```

#### Development Dependencies (Optional)
```bash
pip install -r requirements.txt[dev]
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import openai; print(f'OpenAI version: {openai.__version__}')"
```

## API Key Configuration

### Required API Keys

1. **OpenAI API Key** (for GPT retrieval and generation)
2. **Google API Key** (for Gemini generation)
3. **SerpAPI Key** (for web search)
4. **Qwen API Key** (for Qwen cloud inference)

### Configuration Methods

#### Method 1: Environment Variables (Recommended)
```bash
export OPENAI_API_KEY="your_openai_api_key"
export GOOGLE_API_KEY="your_google_api_key"
export SERPAPI_API_KEY="your_serpapi_key"
export QWEN_API_KEY="your_qwen_api_key"
```

#### Method 2: Direct Configuration
Edit the relevant files and set the API keys directly:

```python
# In utils/search_component.py, gpt_retrieval/call_gpt.py, etc.
OPENAI_API_KEY = "your_openai_api_key"
GOOGLE_API_KEY = "your_google_api_key"
SERPAPI_API_KEY = "your_serpapi_key"
QWEN_API_KEY = "your_qwen_api_key"
```

## Model Downloads

### Local Qwen Model (Optional)

If you want to use local Qwen models instead of API calls:

```bash
# Download Qwen2.5-VL-7B model (requires ~15GB disk space)
python -c "
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
model = Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
print('Model downloaded successfully!')
"
```

## Troubleshooting

### Common Issues

#### 1. CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If CUDA is not available, install CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Flash Attention Installation Issues
```bash
# If flash-attn installation fails, skip it (it's optional)
pip install -r requirements.txt --no-deps
pip install torch transformers openai pillow requests
```

#### 3. Memory Issues
- Reduce batch size in configuration
- Use CPU-only inference for large models
- Close other memory-intensive applications

#### 4. API Key Issues
- Verify API keys are correctly set
- Check API key permissions and quotas
- Ensure network connectivity

### Platform-Specific Notes

#### Windows
- Use PowerShell or Command Prompt
- May need Visual Studio Build Tools for some packages
- Use `orig_env\Scripts\activate` for virtual environment

#### macOS
- May need Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for system dependencies if needed

#### Linux
- Ensure Python development headers are installed
- May need additional system packages for image processing

## Performance Optimization

### For GPU Users
1. Install CUDA-compatible PyTorch
2. Use flash-attention for faster inference
3. Enable mixed precision training/inference

### For CPU Users
1. Use smaller models when possible
2. Reduce batch sizes
3. Consider using cloud APIs instead of local models

## Testing Installation

Run the following test script to verify everything is working:

```bash
python -c "
# Test basic imports
import torch
import transformers
import openai
from PIL import Image
import requests

print('✅ All basic dependencies imported successfully')

# Test API connectivity (if keys are set)
try:
    from openai import OpenAI
    client = OpenAI()
    print('✅ OpenAI client initialized')
except Exception as e:
    print(f'⚠️  OpenAI client test failed: {e}')

print('Installation test completed!')
"
```

## Next Steps

After successful installation:

1. Configure your API keys
2. Download the FIG-Eval dataset
3. Run a test generation: `python main_mp.py --help`
4. Check the README for usage examples

## Support

If you encounter issues during installation:

1. Check the troubleshooting section above
2. Verify your Python version and system requirements
3. Create an issue on the project repository
4. Include your system information and error messages
