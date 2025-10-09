# Backend

This directory contains the backend API and server components for the LLM project.

## Purpose

The `backend/` directory houses:
- REST API endpoints for LLM interactions
- WebSocket connections for real-time communication
- Authentication and authorization logic
- Request/response handling and validation
- Database connections and ORM models

## Structure

```
backend/
├── api/           # API route definitions and handlers
├── models/        # Database models and schemas
├── services/      # Business logic and service classes
├── middleware/    # Custom middleware components
├── utils/         # Backend utilities and helpers
├── config/        # Backend configuration files
└── tests/         # Backend-specific tests
```

## Technologies

- **Framework**: FastAPI / Flask (depending on implementation)
- **Database**: PostgreSQL / SQLite
- **Authentication**: JWT tokens
- **Validation**: Pydantic models
- **Testing**: pytest

## Getting Started

1. Install dependencies: `uv install`
2. Set up environment variables (see `.env.example`)
3. Initialize the database: `python -m backend.db.init`
4. Run the development server: `uvicorn backend.main:app --reload`

## API Endpoints

### Health Check
- `GET /health` - Server health status

### LLM Endpoints
- `POST /chat` - Chat with the LLM
- `POST /generate` - Generate text completion
- `GET /models` - List available models

### Authentication
- `POST /auth/login` - User authentication
- `POST /auth/register` - User registration
- `POST /auth/logout` - User logout

## Configuration

Environment variables:
```bash
DATABASE_URL=sqlite:///./app.db
SECRET_KEY=your-secret-key
OLLAMA_HOST=http://localhost:11434
```

## Development

Run in development mode:
```bash
uvicorn backend.main:app --reload --port 8000
```

Run tests:
```bash
pytest backend/tests/
```