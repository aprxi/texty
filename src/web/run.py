#!/usr/bin/env python3
"""
Development runner for the FastAPI web service.
Usage: python run.py [--host HOST] [--port PORT] [--model MODEL_PATH]
"""

import argparse
import os
import sys
import uvicorn
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run the text classification web service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", help="Path to the model file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Set model path if provided
    if args.model:
        if not os.path.exists(args.model):
            print(f"Error: Model file not found at {args.model}")
            sys.exit(1)
        os.environ["MODEL_PATH"] = args.model
        print(f"Using model: {args.model}")
    elif "MODEL_PATH" not in os.environ:
        # Default model path
        default_model = "/data/models/base_model.pkl"
        if os.path.exists(default_model):
            os.environ["MODEL_PATH"] = default_model
            print(f"Using default model: {default_model}")
        else:
            print(f"Warning: No model specified and default model not found at {default_model}")
            print("Set MODEL_PATH environment variable or use --model parameter")
    
    print(f"Starting server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()