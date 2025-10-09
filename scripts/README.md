# Scripts

This directory contains utility scripts, automation tools, and command-line utilities for the LLM project.

## Purpose

The `scripts/` directory provides:
- Data processing and transformation scripts
- Model training and evaluation automation
- Deployment and infrastructure scripts
- Utility functions and helper scripts
- Batch processing and scheduled tasks

## Structure

```
scripts/
├── data/              # Data processing and transformation
├── training/          # Model training and fine-tuning scripts
├── evaluation/        # Model evaluation and benchmarking
├── deployment/        # Deployment and infrastructure automation
├── utils/             # General utility scripts
├── monitoring/        # Logging and monitoring scripts
└── maintenance/       # System maintenance and cleanup
```

## Script Categories

### Data Processing Scripts
- `data/preprocess.py` - Clean and prepare raw data
- `data/split_dataset.py` - Split data into train/test sets
- `data/validate_format.py` - Validate data format and quality
- `data/convert_formats.py` - Convert between different data formats

### Training Scripts
- `training/fine_tune.py` - Fine-tune models with custom data
- `training/evaluate_model.py` - Evaluate model performance
- `training/hyperparameter_search.py` - Automated hyperparameter tuning
- `training/resume_training.py` - Resume interrupted training sessions

### Deployment Scripts
- `deployment/setup_server.py` - Server setup and configuration
- `deployment/deploy_model.py` - Model deployment automation
- `deployment/health_check.py` - System health monitoring
- `deployment/backup_data.py` - Data backup automation

## Usage Examples

### Data Processing
```bash
# Preprocess raw data
python scripts/data/preprocess.py \
  --input data/raw/conversations.json \
  --output data/processed/conversations.jsonl \
  --format jsonl

# Split dataset
python scripts/data/split_dataset.py \
  --input data/processed/conversations.jsonl \
  --train-ratio 0.8 \
  --output-dir data/training/
```

### Model Training
```bash
# Fine-tune a model
python scripts/training/fine_tune.py \
  --model llama2-7b \
  --dataset data/training/train.jsonl \
  --output-dir models/fine_tuned/custom_model \
  --epochs 3 \
  --learning-rate 1e-4

# Evaluate model performance
python scripts/evaluation/benchmark.py \
  --model models/fine_tuned/custom_model \
  --test-data data/evaluation/test.jsonl \
  --metrics accuracy,perplexity,bleu
```

### Deployment
```bash
# Deploy model to production
python scripts/deployment/deploy_model.py \
  --model-path models/fine_tuned/custom_model \
  --environment production \
  --replicas 3

# Health check
python scripts/deployment/health_check.py \
  --endpoint http://localhost:8000/health \
  --timeout 30
```

## Script Templates

### Basic Script Template
```python
#!/usr/bin/env python3
"""
Script Description: Brief description of what this script does

Usage:
    python script_name.py --arg1 value1 --arg2 value2

Author: Your Name
Date: 2024-01-01
"""

import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument('--input', required=True, help='Input file path')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Script logic here
    logger.info("Starting processing...")
    
    try:
        # Main processing
        result = process_data(args.input, args.output)
        logger.info(f"Processing completed successfully: {result}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1
    
    return 0

def process_data(input_path, output_path):
    """Process data from input to output."""
    # Implementation here
    pass

if __name__ == "__main__":
    exit(main())
```

### Configuration Management
```python
# config.py - Centralized configuration
import os
from pathlib import Path

class Config:
    """Configuration management class."""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "llm_models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Model settings
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama2-7b")
    MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # API settings
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
    
    # Database settings
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app.db")
    
    @classmethod
    def validate(cls):
        """Validate configuration."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
```

## Utility Functions

### Common Utilities
```python
# utils/common.py
import json
import yaml
from pathlib import Path

def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    """Save data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_yaml(file_path):
    """Load YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)
```

### Progress Tracking
```python
# utils/progress.py
from tqdm import tqdm
import time

class ProgressTracker:
    """Progress tracking utility."""
    
    def __init__(self, total, description="Processing"):
        self.pbar = tqdm(total=total, description=description)
        self.start_time = time.time()
    
    def update(self, n=1):
        """Update progress."""
        self.pbar.update(n)
    
    def close(self):
        """Close progress bar and show summary."""
        self.pbar.close()
        elapsed = time.time() - self.start_time
        print(f"Completed in {elapsed:.2f} seconds")
```

## Scheduling and Automation

### Cron Jobs
```bash
# Example crontab entries
# Run daily data backup at 2 AM
0 2 * * * /usr/bin/python3 /path/to/scripts/maintenance/backup_data.py

# Run weekly model evaluation on Sundays at 3 AM
0 3 * * 0 /usr/bin/python3 /path/to/scripts/evaluation/weekly_benchmark.py

# Health check every 5 minutes
*/5 * * * * /usr/bin/python3 /path/to/scripts/monitoring/health_check.py
```

### Systemd Services
```ini
# /etc/systemd/system/llm-monitoring.service
[Unit]
Description=LLM Monitoring Service
After=network.target

[Service]
Type=simple
User=llm-user
WorkingDirectory=/path/to/project
ExecStart=/usr/bin/python3 scripts/monitoring/monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Error Handling and Logging

### Robust Error Handling
```python
import logging
import traceback
from functools import wraps

def handle_errors(func):
    """Decorator for error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            logging.error(traceback.format_exc())
            raise
    return wrapper

@handle_errors
def risky_operation():
    """Operation that might fail."""
    pass
```

### Structured Logging
```python
import logging
import json

class StructuredLogger:
    """Structured logging utility."""
    
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def log_event(self, event, level=logging.INFO, **kwargs):
        """Log structured event."""
        log_data = {
            'event': event,
            'timestamp': time.time(),
            **kwargs
        }
        self.logger.log(level, json.dumps(log_data))
```

## Performance Optimization

### Parallel Processing
```python
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def parallel_process(func, items, max_workers=None):
    """Process items in parallel."""
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logging.error(f"Task failed: {e}")
    
    return results
```

## Testing Scripts

### Unit Tests for Scripts
```python
# tests/test_scripts.py
import unittest
import tempfile
from pathlib import Path
import sys

# Add scripts to path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from data.preprocess import preprocess_data

class TestPreprocessScript(unittest.TestCase):
    """Test data preprocessing script."""
    
    def test_preprocess_data(self):
        """Test data preprocessing function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_file = Path(temp_dir) / "input.json"
            output_file = Path(temp_dir) / "output.jsonl"
            
            # Create test input
            test_data = [{"text": "Hello world"}]
            with open(input_file, 'w') as f:
                json.dump(test_data, f)
            
            # Run preprocessing
            result = preprocess_data(str(input_file), str(output_file))
            
            # Assert output exists
            self.assertTrue(output_file.exists())
```

## Contributing

When adding scripts:
1. Follow the established directory structure
2. Include comprehensive argument parsing
3. Add proper error handling and logging
4. Write unit tests for complex logic
5. Document usage and examples
6. Use consistent naming conventions
7. Include configuration validation