# Tests

This directory contains the test suite for the LLM project, including unit tests, integration tests, and end-to-end testing.

## Purpose

The `tests/` directory provides:
- Unit tests for individual components
- Integration tests for component interactions
- End-to-end tests for complete workflows
- Performance and load testing
- Test utilities and fixtures

## Structure

```
tests/
├── unit/              # Unit tests for individual modules
├── integration/       # Integration tests for component interactions
├── e2e/              # End-to-end tests for complete workflows
├── performance/      # Performance and load testing
├── fixtures/         # Test data and fixtures
├── utils/            # Testing utilities and helpers
├── conftest.py       # Pytest configuration and shared fixtures
└── requirements.txt  # Testing dependencies
```

## Testing Framework

This project uses **pytest** as the primary testing framework with additional tools:
- `pytest` - Core testing framework
- `pytest-cov` - Code coverage reporting
- `pytest-mock` - Mocking utilities
- `pytest-asyncio` - Async test support
- `pytest-xdist` - Parallel test execution

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_models.py

# Run specific test function
pytest tests/unit/test_models.py::test_model_loading

# Run tests with verbose output
pytest -v

# Run tests with coverage
pytest --cov=src --cov-report=html
```

### Test Categories
```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only end-to-end tests
pytest tests/e2e/

# Run performance tests
pytest tests/performance/ --benchmark-only
```

### Parallel Execution
```bash
# Run tests in parallel
pytest -n auto  # Use all available CPUs
pytest -n 4     # Use 4 workers
```

## Test Configuration

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    e2e: marks tests as end-to-end tests
    performance: marks tests as performance tests
```

### conftest.py
```python
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock
import pandas as pd

@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]

@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    mock_client = Mock()
    mock_client.generate.return_value = {"response": "Mocked response"}
    return mock_client

@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        "model_name": "test-model",
        "temperature": 0.7,
        "max_tokens": 100,
    }
```

## Unit Tests

### Testing Individual Functions
```python
# tests/unit/test_utils.py
import pytest
from src.utils import preprocess_text, validate_input

class TestPreprocessing:
    """Test text preprocessing utilities."""
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing."""
        input_text = "Hello, World!"
        expected = "hello world"
        result = preprocess_text(input_text)
        assert result == expected
    
    def test_preprocess_text_empty(self):
        """Test preprocessing with empty input."""
        result = preprocess_text("")
        assert result == ""
    
    def test_preprocess_text_none(self):
        """Test preprocessing with None input."""
        with pytest.raises(TypeError):
            preprocess_text(None)

class TestValidation:
    """Test input validation functions."""
    
    @pytest.mark.parametrize("input_data,expected", [
        ({"text": "Hello"}, True),
        ({"text": ""}, False),
        ({}, False),
        (None, False),
    ])
    def test_validate_input(self, input_data, expected):
        """Test input validation with various inputs."""
        result = validate_input(input_data)
        assert result == expected
```

### Testing Classes
```python
# tests/unit/test_models.py
import pytest
from unittest.mock import patch, Mock
from src.models import LLMModel

class TestLLMModel:
    """Test LLM model class."""
    
    @pytest.fixture
    def model(self, test_config):
        """Create model instance for testing."""
        return LLMModel(test_config)
    
    def test_model_initialization(self, model, test_config):
        """Test model initialization."""
        assert model.config == test_config
        assert model.model_name == test_config["model_name"]
    
    @patch('src.models.ollama.Client')
    def test_generate_text(self, mock_client, model):
        """Test text generation."""
        # Setup mock
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        mock_instance.generate.return_value = {"response": "Test response"}
        
        # Test
        result = model.generate("Test prompt")
        
        # Assert
        assert result == "Test response"
        mock_instance.generate.assert_called_once()
    
    def test_generate_with_invalid_prompt(self, model):
        """Test generation with invalid prompt."""
        with pytest.raises(ValueError):
            model.generate("")
```

## Integration Tests

### Testing Component Interactions
```python
# tests/integration/test_api_integration.py
import pytest
import requests
from unittest.mock import patch
from src.api import create_app
from src.models import LLMModel

@pytest.fixture
def client():
    """Create test client."""
    app = create_app(testing=True)
    return app.test_client()

class TestAPIIntegration:
    """Test API integration with models."""
    
    @patch('src.models.ollama.Client')
    def test_chat_endpoint(self, mock_client, client):
        """Test chat endpoint integration."""
        # Setup
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        mock_instance.generate.return_value = {"response": "Hello!"}
        
        # Test
        response = client.post('/chat', json={
            'message': 'Hi there',
            'model': 'test-model'
        })
        
        # Assert
        assert response.status_code == 200
        data = response.get_json()
        assert data['response'] == 'Hello!'
```

## End-to-End Tests

### Testing Complete Workflows
```python
# tests/e2e/test_workflows.py
import pytest
import subprocess
from pathlib import Path

@pytest.mark.e2e
class TestDataProcessingWorkflow:
    """Test complete data processing workflow."""
    
    def test_full_pipeline(self, temp_dir):
        """Test complete data processing pipeline."""
        # Create test data
        input_file = temp_dir / "input.json"
        input_file.write_text('{"text": "Hello world"}')
        
        # Run preprocessing script
        result = subprocess.run([
            "python", "scripts/data/preprocess.py",
            "--input", str(input_file),
            "--output", str(temp_dir / "output.jsonl")
        ], capture_output=True, text=True)
        
        # Assert success
        assert result.returncode == 0
        assert (temp_dir / "output.jsonl").exists()

@pytest.mark.e2e
class TestModelDeployment:
    """Test model deployment workflow."""
    
    def test_model_deployment(self):
        """Test complete model deployment."""
        # This would test the full deployment pipeline
        pass
```

## Performance Tests

### Load and Performance Testing
```python
# tests/performance/test_performance.py
import pytest
import time
from concurrent.futures import ThreadPoolExecutor
from src.models import LLMModel

@pytest.mark.performance
class TestModelPerformance:
    """Test model performance characteristics."""
    
    def test_generation_latency(self, mock_llm_client):
        """Test text generation latency."""
        model = LLMModel({"model_name": "test"})
        model.client = mock_llm_client
        
        start_time = time.time()
        model.generate("Test prompt")
        latency = time.time() - start_time
        
        assert latency < 1.0  # Should respond within 1 second
    
    def test_concurrent_requests(self, mock_llm_client):
        """Test handling concurrent requests."""
        model = LLMModel({"model_name": "test"})
        model.client = mock_llm_client
        
        def make_request():
            return model.generate("Test prompt")
        
        # Test with 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in futures]
        
        assert len(results) == 10
        assert all(result for result in results)
```

## Test Data and Fixtures

### Sample Data
```python
# tests/fixtures/sample_data.py
SAMPLE_CONVERSATIONS = [
    {
        "id": "conv_001",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    },
    {
        "id": "conv_002",
        "messages": [
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"}
        ]
    }
]

SAMPLE_TRAINING_DATA = [
    {"input": "What is Python?", "output": "Python is a programming language."},
    {"input": "How do I install packages?", "output": "Use pip to install packages."},
]
```

### Data Generation
```python
# tests/utils/data_generators.py
import random
from faker import Faker

fake = Faker()

def generate_conversation(num_turns=5):
    """Generate a fake conversation."""
    messages = []
    for i in range(num_turns):
        role = "user" if i % 2 == 0 else "assistant"
        content = fake.sentence()
        messages.append({"role": role, "content": content})
    return {"messages": messages}

def generate_training_data(num_samples=100):
    """Generate fake training data."""
    data = []
    for _ in range(num_samples):
        input_text = fake.question()
        output_text = fake.sentence()
        data.append({"input": input_text, "output": output_text})
    return data
```

## Test Utilities

### Assertion Helpers
```python
# tests/utils/assertions.py
import json

def assert_valid_json(text):
    """Assert that text is valid JSON."""
    try:
        json.loads(text)
    except json.JSONDecodeError:
        raise AssertionError(f"Invalid JSON: {text}")

def assert_response_format(response, expected_keys):
    """Assert response has expected format."""
    assert isinstance(response, dict)
    for key in expected_keys:
        assert key in response, f"Missing key: {key}"

def assert_model_output(output, min_length=10):
    """Assert model output meets quality criteria."""
    assert isinstance(output, str)
    assert len(output.strip()) >= min_length
    assert output.strip() != ""
```

### Mock Helpers
```python
# tests/utils/mocks.py
from unittest.mock import Mock

def create_mock_llm_response(text="Mock response"):
    """Create a mock LLM response."""
    return {"response": text, "done": True}

def create_mock_client():
    """Create a mock LLM client."""
    mock_client = Mock()
    mock_client.generate.return_value = create_mock_llm_response()
    return mock_client
```

## Coverage and Reporting

### Coverage Configuration
```ini
# .coveragerc
[run]
source = src
omit = 
    */tests/*
    */venv/*
    */env/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

### Generating Reports
```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Generate XML coverage report (for CI)
pytest --cov=src --cov-report=xml

# Show missing lines
pytest --cov=src --cov-report=term-missing
```

## Continuous Integration

### GitHub Actions Example
```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r tests/requirements.txt
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## Best Practices

### Writing Good Tests
1. **Descriptive names**: Test names should clearly describe what is being tested
2. **Single responsibility**: Each test should test one specific behavior
3. **Independent tests**: Tests should not depend on each other
4. **Fast execution**: Unit tests should run quickly
5. **Reliable**: Tests should be deterministic and not flaky

### Test Organization
1. **Follow naming conventions**: Use clear, consistent naming patterns
2. **Group related tests**: Use classes to group related test methods
3. **Use fixtures**: Share common setup code using fixtures
4. **Parameterize tests**: Use parametrization for testing multiple inputs
5. **Mock external dependencies**: Isolate units under test

### Debugging Tests
```bash
# Run specific test with debugging
pytest tests/unit/test_models.py::test_model_loading -v -s

# Drop into debugger on failure
pytest --pdb

# Run last failed tests only
pytest --lf

# Show local variables on failure
pytest --tb=long
```