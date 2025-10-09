# Data

This directory contains datasets, training data, and data processing utilities for the LLM project.

## Purpose

The `data/` directory is organized to store:
- Training and fine-tuning datasets
- Evaluation and benchmark data
- Preprocessed data files
- Data processing scripts and utilities
- Sample data for testing and development

## Structure

```
data/
├── raw/              # Original, unprocessed datasets
├── processed/        # Cleaned and preprocessed data
├── training/         # Data specifically for model training
├── evaluation/       # Test and validation datasets
├── samples/          # Small sample datasets for development
├── scripts/          # Data processing and transformation scripts
└── schemas/          # Data schema definitions and validation
```

## Data Types

### Text Data
- Conversation logs
- Document corpora
- Question-answer pairs
- Code repositories

### Structured Data
- JSON/JSONL files
- CSV datasets
- Parquet files
- Database exports

## Data Processing Pipeline

1. **Raw Data Ingestion** → `data/raw/`
2. **Cleaning & Preprocessing** → `data/scripts/`
3. **Processed Data** → `data/processed/`
4. **Training Split** → `data/training/`
5. **Evaluation Split** → `data/evaluation/`

## Common Data Formats

### Conversation Data
```json
{
  "id": "conv_001",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"}
  ],
  "metadata": {
    "timestamp": "2024-01-01T00:00:00Z",
    "source": "chat_logs"
  }
}
```

### Training Data
```jsonl
{"input": "What is Python?", "output": "Python is a programming language..."}
{"input": "How do I install packages?", "output": "You can use pip to install packages..."}
```

## Data Processing Scripts

Common data processing operations:
```bash
# Clean and preprocess raw data
python data/scripts/preprocess.py --input data/raw/ --output data/processed/

# Split data into train/validation sets
python data/scripts/split_data.py --input data/processed/ --train-ratio 0.8

# Validate data format
python data/scripts/validate.py --schema data/schemas/conversation.json
```

## Data Privacy and Security

⚠️ **Important Guidelines:**
- Never commit sensitive or personal data
- Use `.gitignore` to exclude large datasets
- Implement data anonymization for personal information
- Follow GDPR and privacy regulations
- Use secure methods for data transfer

## Dataset Sources

Document your data sources:
- Public datasets (e.g., Common Crawl, OpenWebText)
- Proprietary datasets (with proper licensing)
- Synthetic data generation
- User-generated content (with consent)

## Data Quality Checks

Implement quality assurance:
- Data validation scripts
- Duplicate detection
- Format consistency checks
- Statistical analysis
- Manual spot checks

## Usage Examples

```python
# Load processed data
import json
from pathlib import Path

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Load training data
training_data = load_jsonl('data/training/conversations.jsonl')
```

## Contributing

When working with data:
1. Document data sources and processing steps
2. Implement data validation
3. Use consistent naming conventions
4. Add appropriate metadata
5. Respect privacy and licensing requirements