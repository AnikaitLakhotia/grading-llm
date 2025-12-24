# Grading API

This document describes how to call the grading API.

## Endpoints

### GET `/health`
Returns basic service metadata.

Example:
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "version": "0.0.0",
  "git_sha": "abc123",
  "env": "local"
}
```

### POST `/grade`
Request: JSON body with fields:
- `essay_id` (string, required)
- `full_text` (string, required)
- `assignment` (string, optional)
- `prompt_name` (string, optional)
- `grade_level` (int, optional)

Example:
```bash
curl -sS -X POST http://localhost:8000/grade -H "Content-Type: application/json"   -d '{"essay_id":"e1","full_text":"This is a sample essay...", "assignment":"asm1", "prompt_name":"p1", "grade_level":10}'
```

Successful response (200):
```json
{
  "score": 4,
  "feedback": "Short feedback string",
  "evidence": "Optional evidence snippet",
  "confidence": 0.72
}
```

Errors:
- `413` — payload too large (full_text exceeded allowed length).
- `500` — internal grading error.

Notes:
- The API returns Pydantic-validated JSON and includes OpenAPI docs at `/docs`.
- Do not send raw PII to the public instance; use sanitized essays only.
