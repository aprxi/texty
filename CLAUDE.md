# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a Python prototype for a real-time text classification system that identifies high-risk AI applications based on vocabulary analysis. The system uses a unified TF-IDF + Naive Bayes approach optimized for performance-critical applications requiring sub-50ms classification of 1000-word texts.

**IMPORTANT**: All training and evaluation data is completely artificial and synthetic. See DISCLAIMER.md files in vocabulary/ and tests/evaluate/ directories.

## Key Technical Requirements

- **Performance**: Must classify 1000-word texts in < 50ms on a 4-core CPU
- **Memory**: Model and dependencies must use < 150 MB memory
- **Architecture**: Multinomial Naive Bayes classifier with TF-IDF vectorization
- **Classification Logic**: Sentence-level classification with configurable aggregation thresholds

## Container Runtime Support

This project supports both Docker and Podman container runtimes. The Makefile automatically detects which runtime is available on your system.

## Common Development Commands

```bash
# Build the container image
make build

# Train unified base model
make train

# Training is configured via ./vocabulary/config.yaml
# Edit config.yaml to adjust:
# - Feature count (tfidf.max_features)
# - N-gram settings (tfidf.ngram_range)
# - Vocabulary sources (vocabulary_sources)
# - Synthetic data generation settings

# Evaluate model on test dataset
make evaluate
make evaluate VERBOSE=true
make evaluate DEBUG=true  # Show detailed debug analysis

# Evaluate single file
make evaluate FILE=./test.txt

# Build and run web interface
make web        # Build web container
make web-run    # Run web server

# Clean up container and image
make clean
```

## Current Directory Structure

```
./artifacts/
├── models/                      # Trained models
│   └── base.latest.pkl          # Latest trained model

./vocabulary/                    # Training vocabulary (SYNTHETIC DATA)
├── function/                    # AI function/capability terms
│   ├── hr/                      # High-risk functions
│   └── lr/                      # Low-risk functions
├── what/                        # What is being processed
│   ├── hr/                      # High-risk data types
│   └── lr/                      # Low-risk data types
├── target/                      # Who is affected
│   ├── hr/                      # High-risk target groups
│   └── lr/                      # Low-risk target groups
├── patch/                       # Contextual patches
└── DISCLAIMER.md                # Artificial data notice

./tests/
└── evaluate/                    # Evaluation dataset (SYNTHETIC DATA)
    ├── high_risk/               # High-risk test texts
    ├── low_risk/                # Low-risk test texts
    ├── RULES.md                 # Classification rules
    └── DISCLAIMER.md            # Artificial data notice

./src/
├── trainer/                     # Training module
│   ├── run.py                   # Main training script
│   └── export_onnx.py           # ONNX export utility
├── evaluator/                   # Evaluation module
│   ├── classify.py              # Single text classification
│   ├── evaluate.py              # Dataset evaluation
│   └── run.py                   # Base model evaluation
├── web/                         # Web interface
│   ├── main.py                  # FastAPI server
│   ├── html/index.html          # Web UI
│   └── Dockerfile               # Web container config
└── helpers/                     # Utility modules
    ├── stopwords_combined.py    # Dutch + English stopwords
    └── vocabulary_vectorizer.py # Vocabulary processing
```

## Inside Container Commands

```bash
# Train base unified model
python -m src.trainer.run

# Train with custom parameters
python -m src.trainer.run --features 7500 --ngrams 3 --exclude function,what

# Classify a single text
python -m src.evaluator.classify --model /data/artifacts/models/base.latest.pkl --input /app/test.txt

# Evaluate on test set
python -m src.evaluator.run --model /data/artifacts/models/base.latest.pkl --eval-dir /app/tests/evaluate

# Evaluate with verbose output
python -m src.evaluator.run --model /data/artifacts/models/base.latest.pkl --eval-dir /app/tests/evaluate --verbose
```

## Architecture Overview

The system uses a unified vocabulary-based classification approach:

### Classification Pipeline

1. **Vocabulary-Based Training**: The system learns from categorized vocabulary terms:
   - **Function**: What the AI system does (e.g., facial recognition, automated decision-making)
   - **What**: What data is processed (e.g., biometric data, personal information)
   - **Target**: Who is affected (e.g., employees, citizens, students)
   - **Generated**: Synthetic combinations for improved coverage

2. **Sentence-Level Classification**: Each sentence is classified using TF-IDF features and Naive Bayes
   - Configurable n-gram range (default: unigrams + bigrams)
   - Dutch + English stopword filtering
   - Configurable feature count (default: 5000, max: 10000)

3. **Text-Level Aggregation**: Final classification based on configurable thresholds
   - Count threshold: Number of high-risk sentences (default: 1)
   - Percentage threshold: Proportion of high-risk sentences (default: 10%)

### Key Components

- **Feature Engineering**: TF-IDF with configurable n-grams (default: unigrams + bigrams, configurable up to trigrams)
- **Preprocessing Pipeline**: Lowercase conversion, punctuation removal, stopword filtering
- **Classification Confidence**: 
  - Confidence threshold filtering (default 0.7, configurable)
  - Weighted scoring option that uses probability scores instead of binary counting
  - Adjustable thresholds for both count and percentage-based classification
- **Performance Optimizations**:
  - Vocabulary capping (configurable, default: 5,000 features, max recommended: 10,000)
  - Early stopping mechanism for long texts
  - Optional sentence capping for very long texts

### Implementation Stack

- **scikit-learn**: Naive Bayes classifier and TF-IDF vectorization
- **NLTK**: Sentence segmentation
- **NumPy**: Numerical operations
- **FastAPI**: Web interface framework
- **joblib**: Model serialization

## Development Guidelines

1. Always prioritize performance - profile any changes that might impact latency
2. Maintain the sentence-level classification architecture
3. Keep classification thresholds configurable
4. Document performance benchmarks for any model changes
5. Include timing measurements in any performance-critical functions
6. Use the containerized environment for consistent development and testing
7. The Makefile supports both Docker and Podman - it will auto-detect your runtime
8. Use the unified base model (`make train`) for best results
9. Synthetic data is generated automatically during training (see config.yaml)
10. Adjust vocabulary sources if needed: Edit `./vocabulary/config.yaml`
11. All data is synthetic - treat it as training/test data only, not real scenarios

## Container Environment

- **Base Image**: Python 3.11-slim for minimal footprint
- **Working Directory**: `/app` inside the container
- **Volume Mounts**: 
  - `./` mounted to `/app` (code and scripts)
  - `./artifacts` mounted to `/data/artifacts` (models and results)
  - `./vocabulary` mounted to `/app/vocabulary` (training vocabulary)
  - `./tests` mounted to `/app/tests` (evaluation data)
- **Environment Variables**:
  - `MODEL_PATH`: Default location for trained model (default: `/data/artifacts/models/base.latest.pkl`)
  - `LOG_LEVEL`: Controls logging verbosity (default: `INFO`)

## Web Interface

The web interface provides:
- Real-time text classification
- Visual sentence-by-sentence analysis with color coding
- Classification reasoning and metrics
- Simple interface with just Validate and Clear buttons
- Always shows detailed analysis (no toggles needed)

To use:
1. `make web` - Build the web container
2. `make web-run` - Start the server at http://localhost:8000

## Creating Evaluation Texts

For rules and guidelines on creating evaluation prompts and understanding classification criteria, see `./tests/evaluate/RULES.md`. This document contains:
- The 8 high-risk AI categories according to EU AI Act
- Classification criteria for high-risk vs low-risk AI systems
- Quality guidelines for evaluation prompts
- Examples of well-formed evaluation texts

Remember: All evaluation texts are synthetic and created for testing purposes only.