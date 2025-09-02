import os
import sys
import time
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.evaluator.classify import classify_text
except ImportError:
    # Fallback for container environment
    sys.path.append('/app')
    from src.evaluator.classify import classify_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Text Classification Service", version="1.0.0")

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "/data/models/base.model.pkl")
model = None

class ClassificationRequest(BaseModel):
    text: str
    debug: bool = False
    confidence_threshold: float = 0.7
    count_threshold: int = 1
    percentage_threshold: float = 10.0

class ClassificationResponse(BaseModel):
    classification: str
    confidence: float
    high_risk_sentences: int
    total_sentences: int
    percentage_high_risk: float
    processing_time_ms: float
    text_length_chars: int
    text_length_words: int
    classification_reasoning: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        logger.info(f"Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/", response_class=HTMLResponse)
async def index():
    # Try multiple possible paths for the HTML file
    html_paths = [
        "html/index.html",                              # Local development
        "src/web/html/index.html",                      # From app root
        "/app/src/web/html/index.html",                 # Container absolute path
        str(Path(__file__).parent / "html" / "index.html")  # Relative to this file
    ]
    
    for html_path in html_paths:
        try:
            with open(html_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            continue
    
    # If no HTML file found, return a basic page
    return """
    <html>
        <head><title>Text Classification Service</title></head>
        <body>
            <h1>Text Classification Service</h1>
            <p>HTML interface file not found. Please check the installation.</p>
            <p>Available endpoints:</p>
            <ul>
                <li><a href="/api/info">/api/info</a> - Service information</li>
                <li><a href="/health">/health</a> - Health check</li>
                <li><strong>POST /classify</strong> - Text classification</li>
            </ul>
        </body>
    </html>
    """

@app.post("/classify", response_model=ClassificationResponse)
async def classify(
    text: str = Form(...),
    debug: str = Form("false"),  # Receive as string from form
    confidence_threshold: float = Form(0.7),
    count_threshold: int = Form(1),
    percentage_threshold: float = Form(10.0)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        start_time = time.time()
        
        # Always run with debug=True to get sentence details
        classification, high_risk_count, total_sentences, sentence_details = classify_text(
            model=model,
            text=text,
            threshold_count=count_threshold,
            threshold_percent=percentage_threshold / 100.0,  # Convert percentage to decimal
            confidence_threshold=confidence_threshold,
            use_weighted_scoring=False,
            debug=True  # Always get detailed output for visual display
        )
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Calculate additional metrics
        percentage_high_risk = (high_risk_count / total_sentences * 100) if total_sentences > 0 else 0
        text_length_chars = len(text)
        text_length_words = len(text.split()) if text.strip() else 0
        
        # Get average confidence of high-risk sentences
        high_risk_sentences_list = [s for s in sentence_details if s.get('effective_prediction') == 'HIGH-RISK']
        avg_confidence = sum(s.get('high_risk_prob', 0) for s in high_risk_sentences_list) / len(high_risk_sentences_list) if high_risk_sentences_list else 0
        
        # Generate classification reasoning similar to classify.py
        weighted_score = sum(s.get('high_risk_prob', 0) for s in sentence_details) if sentence_details else 0
        weighted_percent = weighted_score / total_sentences if total_sentences > 0 else 0
        
        reasoning_parts = [
            f"Text Analysis: {text_length_chars} characters, {text_length_words} words, {total_sentences} sentences",
            f"Stage 1 - Sentence Level: Each sentence needs High-risk probability >= {confidence_threshold}",
            f"  Sentences meeting threshold: {high_risk_count}/{total_sentences} ({percentage_high_risk:.1f}%)",
            f"Stage 2 - Text Level: Final classification based on aggregation:",
            f"  COUNT MODE: Qualifying sentences: {high_risk_count}",
            f"  Thresholds: count > {count_threshold} OR percentage > {percentage_threshold / 100.0}",
        ]
        
        if high_risk_count > count_threshold or (percentage_high_risk / 100.0) > (percentage_threshold / 100.0):
            reasoning_parts.append(f"  RESULT: Threshold exceeded → {classification.upper()}")
        else:
            reasoning_parts.append(f"  RESULT: Threshold not met → {classification.upper()}")
            
        classification_reasoning = "\n".join(reasoning_parts)
        
        # Find high-risk sentences for summary
        high_risk_sentences_details = [
            {
                "index": i + 1,
                "text": s['sentence'],
                "confidence": s['high_risk_prob']
            }
            for i, s in enumerate(sentence_details) 
            if s.get('effective_prediction') == 'HIGH-RISK'
        ]
        
        response = ClassificationResponse(
            classification=classification,
            confidence=avg_confidence,
            high_risk_sentences=high_risk_count,
            total_sentences=total_sentences,
            percentage_high_risk=percentage_high_risk,
            processing_time_ms=processing_time,
            text_length_chars=text_length_chars,
            text_length_words=text_length_words,
            classification_reasoning=classification_reasoning,
            debug_info={
                "sentences": sentence_details,
                "thresholds": {
                    "confidence": confidence_threshold,
                    "count": count_threshold,
                    "percentage": percentage_threshold
                },
                "metrics": {
                    "weighted_score": weighted_score,
                    "weighted_percent": weighted_percent,
                    "avg_confidence": avg_confidence
                },
                "high_risk_sentences": high_risk_sentences_details,
                "classification_logic": {
                    "stage1_qualifying_sentences": high_risk_count,
                    "stage2_count_check": high_risk_count > count_threshold,
                    "stage2_percent_check": (percentage_high_risk / 100.0) > (percentage_threshold / 100.0),
                    "final_result": classification
                }
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.get("/api/info")
async def info():
    return {
        "service": "Text Classification Service",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "/": "Web interface",
            "/classify": "Text classification endpoint (POST)",
            "/health": "Health check",
            "/api/info": "Service information"
        }
    }

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get host and port from environment or use defaults
    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_PORT", "8000"))
    
    print(f"🌐 Starting Text Classification Web Service")
    print(f"📡 Server: http://{host}:{port}")
    print(f"🤖 Model: {MODEL_PATH}")
    print(f"🛑 Press Ctrl+C to stop")
    print("")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )