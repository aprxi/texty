# Text Classification Web Service

A FastAPI-based web interface for the AI text risk classifier that provides real-time classification with detailed analysis similar to the `make evaluate FILE=...` command.

## Features

- **Web Interface**: Clean, responsive HTML interface with large text input area
- **Real-time Classification**: Fast text analysis with sub-50ms performance goals
- **Detailed Analysis**: Sentence-by-sentence breakdown with confidence scores
- **Debug Mode**: Comprehensive debugging information including feature analysis
- **Configurable Parameters**: Adjustable confidence, count, and percentage thresholds
- **Classification Reasoning**: Step-by-step explanation of classification logic

## Quick Start

### Using Make Commands (Recommended)

1. Build and train the model:
```bash
make build
make train
```

2. Start the web service:
```bash
make web                    # Production mode
# OR
make web-dev               # Development mode with auto-reload
```

3. Open your browser to: http://localhost:8000

4. Test the service (optional):
```bash
python test_web.py         # Run automated tests
```

### Using Docker/Podman Directly

Alternatively, run the web service directly:
```bash
# Copy model and run container
podman run -it --rm \
  -p 8000:8000 \
  -v /path/to/artifacts/models/base_latest.pkl:/app/model.pkl:ro \
  -e MODEL_PATH=/app/model.pkl \
  texty-classifier \
  python src/web/main.py
```

### Local Development

1. Install dependencies:
```bash
cd src/web
pip install -r requirements.txt
```

2. Set model path and run:
```bash
export MODEL_PATH=/path/to/your/base_model.pkl
python run.py --reload
```

3. Open browser to: http://localhost:8000

## API Endpoints

### Web Interface
- `GET /` - Main web interface

### Classification API
- `POST /classify` - Classify text with form data
  - `text` (required): Text to classify
  - `debug` (optional): Enable detailed analysis
  - `confidence_threshold` (optional): Confidence threshold (0-1, default: 0.7)
  - `count_threshold` (optional): Count threshold (default: 1)
  - `percentage_threshold` (optional): Percentage threshold (default: 10.0)

### Service Info
- `GET /health` - Health check and model status
- `GET /api/info` - Service information and available endpoints

## Response Format

The classification endpoint returns detailed information including:

```json
{
  "classification": "high-risk|neutral|low-risk",
  "confidence": 0.85,
  "high_risk_sentences": 2,
  "total_sentences": 10,
  "percentage_high_risk": 20.0,
  "processing_time_ms": 45.2,
  "text_length_chars": 1250,
  "text_length_words": 245,
  "classification_reasoning": "Step-by-step classification logic...",
  "debug_info": {
    "sentences": [...],
    "thresholds": {...},
    "metrics": {...},
    "classification_logic": {...}
  }
}
```

## Debug Mode

When debug mode is enabled, the response includes:

- **Sentence-by-Sentence Analysis**: Individual sentence classifications with confidence scores
- **Feature Analysis**: Top contributing words/n-grams for each sentence
- **Risk vs Safe Indicators**: Words that contribute to high-risk vs low-risk classification
- **Classification Logic**: Detailed breakdown of threshold checks and decision process

## Configuration

Environment variables:
- `MODEL_PATH`: Path to the trained model file (default: `/data/models/base_model.pkl`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

## Development

The service includes:
- **Auto-reload**: Use `--reload` flag for development
- **Error Handling**: Comprehensive error messages and HTTP status codes  
- **Logging**: Structured logging for debugging and monitoring
- **Health Checks**: Model loading status and service health endpoints

## Performance

- Target: <50ms classification for 1000-word texts
- Memory: <150MB including model and dependencies
- Optimized for real-time web usage with sentence-level caching