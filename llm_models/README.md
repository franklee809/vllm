# LLM Models

This directory contains model files, configurations, and utilities for managing Large Language Models.

## Purpose

The `llm_models/` directory manages:
- Model weights and checkpoints
- Model configuration files
- Model loading and initialization utilities
- Fine-tuned model variants
- Model evaluation and benchmarking tools

## Structure

```
llm_models/
├── pretrained/       # Pre-trained model files and weights
├── fine_tuned/       # Custom fine-tuned models
├── configs/          # Model-specific configuration files
├── loaders/          # Model loading utilities and adapters
├── benchmarks/       # Model evaluation and benchmark results
└── utils/            # Model utilities and helper functions
```

## Supported Models

### Open Source Models
- **Llama 2** (7B, 13B, 70B parameters)
- **Code Llama** - Specialized for code generation
- **Mistral 7B** - Efficient and performant
- **Phi-3** - Microsoft's small language model
- **Qwen** - Multilingual capabilities

### Model Formats
- **GGUF** - Quantized models for CPU inference
- **Safetensors** - Safe model weight format
- **PyTorch** - Native PyTorch model files
- **ONNX** - Cross-platform model format

## Model Management

### Downloading Models
```bash
# Using Ollama
ollama pull llama2
ollama pull codellama

# Using Hugging Face Hub
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/DialoGPT-medium')"
```

### Model Configuration
```yaml
# Example model config
model:
  name: "llama2-7b"
  path: "llm_models/pretrained/llama2/"
  format: "gguf"
  
parameters:
  temperature: 0.7
  top_p: 0.9
  top_k: 40
  max_tokens: 2048
  
hardware:
  device: "cuda"  # or "cpu", "mps"
  precision: "fp16"
  batch_size: 1
```

### Loading Models
```python
# Example model loader
from llm_models.loaders import ModelLoader

loader = ModelLoader()
model = loader.load_model(
    name="llama2-7b",
    config_path="llm_models/configs/llama2.yaml"
)

# Generate text
response = model.generate(
    prompt="What is machine learning?",
    temperature=0.7,
    max_tokens=100
)
```

## Model Optimization

### Quantization
- **4-bit quantization** for memory efficiency
- **8-bit quantization** for balanced performance
- **Dynamic quantization** for inference optimization

### Hardware Acceleration
- **GPU acceleration** with CUDA
- **Apple Silicon** optimization with Metal Performance Shaders
- **CPU optimization** with Intel MKL or OpenBLAS

## Model Evaluation

### Benchmarks
- **MMLU** - Massive Multitask Language Understanding
- **HellaSwag** - Common sense reasoning
- **HumanEval** - Code generation capabilities
- **GSM8K** - Mathematical reasoning

### Evaluation Scripts
```bash
# Run standard benchmarks
python llm_models/benchmarks/run_eval.py --model llama2-7b --benchmark mmlu

# Custom evaluation
python llm_models/benchmarks/custom_eval.py --config evaluation_config.yaml
```

## Fine-tuning

### Supported Techniques
- **LoRA** (Low-Rank Adaptation)
- **QLoRA** (Quantized LoRA)
- **Full Fine-tuning**
- **PEFT** (Parameter-Efficient Fine-tuning)

### Fine-tuning Example
```python
from llm_models.fine_tune import LoRATrainer

trainer = LoRATrainer(
    model_name="llama2-7b",
    dataset_path="data/training/custom_dataset.jsonl",
    output_dir="llm_models/fine_tuned/my_model"
)

trainer.train(epochs=3, learning_rate=1e-4)
```

## Storage and Organization

### File Naming Convention
```
llm_models/
├── pretrained/
│   ├── llama2-7b-chat.gguf
│   ├── llama2-13b-instruct.gguf
│   └── mistral-7b-v0.1.gguf
├── fine_tuned/
│   ├── custom-assistant-v1/
│   └── code-helper-v2/
```

### Model Metadata
```json
{
  "model_name": "llama2-7b-chat",
  "version": "1.0",
  "parameters": "7B",
  "file_size": "3.8GB",
  "quantization": "q4_0",
  "license": "Custom License",
  "use_cases": ["chat", "instruction_following"],
  "benchmark_scores": {
    "mmlu": 0.456,
    "hellaswag": 0.774
  }
}
```

## Security Considerations

⚠️ **Important Notes:**
- Model files are typically large (GB to TB)
- Use `.gitignore` to exclude model files from version control
- Consider using Git LFS for smaller model files
- Implement access controls for proprietary models
- Verify model checksums to ensure integrity

## Performance Optimization

### Memory Management
```python
import torch

# Clear GPU memory
torch.cuda.empty_cache()

# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Enable memory-efficient attention
model.config.use_memory_efficient_attention = True
```

### Inference Optimization
- Use smaller batch sizes for memory constraints
- Implement model parallelism for large models
- Cache frequent prompts and responses
- Use streaming for long text generation

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size or use quantization
2. **Slow Inference**: Check hardware acceleration settings
3. **Model Loading Errors**: Verify file paths and formats
4. **Poor Quality**: Adjust temperature and sampling parameters

### Debugging Tools
```bash
# Check model information
python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('model_path'))"

# Monitor GPU usage
nvidia-smi

# Profile memory usage
python -m memory_profiler your_script.py
```