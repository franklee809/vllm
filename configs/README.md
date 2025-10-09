# Configs

This directory contains configuration files for various components of the LLM project.

## Purpose

The `configs/` directory stores:
- Model configuration files
- Environment-specific settings
- Application configuration templates
- Deployment configurations
- Feature flags and switches

## Structure

```
configs/
├── models/          # LLM model configurations
├── environments/    # Environment-specific configs
├── agents/         # Agent configuration files
├── deployment/     # Docker and deployment configs
└── templates/      # Configuration templates
```

## Configuration Files

### Model Configurations
- `models/default.yaml` - Default model settings
- `models/llama.yaml` - Llama-specific configurations
- `models/custom.yaml` - Custom model configurations

### Environment Configurations
- `environments/development.yaml` - Development settings
- `environments/production.yaml` - Production settings
- `environments/testing.yaml` - Testing environment settings

## Configuration Format

Configurations use YAML format for readability:

```yaml
# Example model configuration
model:
  name: "llama2"
  temperature: 0.7
  max_tokens: 2048
  top_p: 0.9
  
api:
  host: "localhost"
  port: 11434
  timeout: 30
  
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Environment Variables

Configuration files can reference environment variables:

```yaml
database:
  url: ${DATABASE_URL}
  pool_size: ${DB_POOL_SIZE:-10}
```

## Usage

Load configuration in Python:

```python
import yaml
from pathlib import Path

def load_config(config_name):
    config_path = Path("configs") / f"{config_name}.yaml"
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load development config
config = load_config('environments/development')
```

## Best Practices

1. **Secrets**: Never store sensitive data in config files
2. **Environment**: Use environment variables for sensitive values
3. **Validation**: Validate configuration on application startup
4. **Documentation**: Document all configuration options
5. **Defaults**: Provide sensible default values

## Contributing

When adding new configurations:
- Follow the existing YAML structure
- Add documentation for new options
- Provide example values
- Consider environment-specific variations