# Text Classification System

A high-performance real-time text classification system that identifies high-risk AI applications based on vocabulary analysis. The system uses a unified TF-IDF + Naive Bayes approach optimized for sub-50ms classification performance.

## Features

- **Performance**: Classifies 1000-word texts in < 50ms on 4-core CPU
- **Vocabulary-Based**: Learns from categorized AI terminology (function, data, targets)
- **Configurable**: Adjustable features, n-grams, and classification thresholds
- **Container-based**: Works with both Docker and Podman
- **Modular**: Organized into trainer, evaluator, and generator modules
- **Web Interface**: Visual classification interface with sentence-by-sentence analysis
- **100% Synthetic Data**: All training and test data is artificially generated

## Quick Start

```bash
# Build the container
make build

# Train the unified model
make train

# Evaluate model performance
make evaluate

# Evaluate with detailed debug output
make evaluate DEBUG=true

# Classify a single text file
make evaluate FILE=./test.txt

# Clean up
make clean
```

## Training Configuration

Training parameters are configured in `./vocabulary/config.yaml`. Edit this file to:
- Adjust TF-IDF features (default: 5000)
- Change n-gram settings (default: unigrams + bigrams)
- Enable/disable vocabulary sources
- Configure synthetic data generation
- Set classification thresholds

```bash
# Train with default settings from config.yaml
make train

# View/edit configuration
cat ./vocabulary/config.yaml
```


## Project Structure

```
texty/
├── artifacts/                   # Training artifacts
│   └── models/                  # Trained models
│       └── base.latest.pkl      # Latest trained model
├── src/                         # Source code
│   ├── trainer/                 # Training module
│   │   ├── run.py              # Main training script
│   │   └── export_onnx.py      # ONNX export utility
│   ├── evaluator/               # Evaluation module
│   │   ├── classify.py         # Single text classification
│   │   └── evaluate.py         # Dataset evaluation
│   ├── generator/               # Synthetic data generation
│   ├── web/                     # Web interface
│   │   └── main.py             # Web server
│   └── helpers/                 # Utility functions
│       ├── stopwords_combined.py    # Dutch + English stopwords
│       └── vocabulary_vectorizer.py # Vocabulary processing
├── vocabulary/                  # Training vocabulary data (SYNTHETIC)
│   ├── function/                # AI capabilities vocabulary
│   │   ├── hr/                  # High-risk functions
│   │   └── lr/                  # Low-risk functions
│   ├── what/                    # Data types vocabulary
│   │   ├── hr/                  # High-risk data types
│   │   └── lr/                  # Low-risk data types
│   ├── target/                  # Affected groups vocabulary
│   │   ├── hr/                  # High-risk targets
│   │   └── lr/                  # Low-risk targets
│   └── DISCLAIMER.md            # Artificial data notice
└── tests/                       # Test data
    └── evaluate/                # Evaluation dataset (SYNTHETIC)
        ├── high_risk/           # High-risk test texts
        ├── low_risk/            # Low-risk test texts
        ├── RULES.md             # Classification rules
        └── DISCLAIMER.md        # Artificial data notice
```

## Important Notice: Synthetic Data

**ALL DATA IN THIS PROJECT IS COMPLETELY ARTIFICIAL AND SYNTHETIC**

Both the training vocabulary in `./vocabulary/` and the evaluation test cases in `./tests/evaluate/` are 100% artificially generated for machine learning purposes. This data:
- ✅ Is created specifically for training and testing ML models
- ✅ Does not represent real AI systems or implementations
- ✅ Contains no sensitive or real-world information
- ❌ Should not be interpreted as documentation of actual systems

See `DISCLAIMER.md` files in the respective directories for full details.

## System Requirements

- **Performance**: Classifies 1000-word texts in <50ms on 4-core CPU
- **Memory**: Model and dependencies use <150MB RAM
- **Accuracy**: Optimized for high precision to minimize false positives
- **Architecture**: Multinomial Naive Bayes with TF-IDF features
- **Container Runtime**: Docker or Podman (auto-detected)

## Training Data Organization

The system uses a vocabulary-based approach with three main dimensions:

### Vocabulary Dimensions

1. **Function** (`vocabulary/function/`): AI system capabilities
   - High-risk (hr/): facial recognition, automated decision-making, biometric analysis
   - Low-risk (lr/): recommendations, content generation, translation

2. **What** (`vocabulary/what/`): Data types being processed
   - High-risk (hr/): biometric data, medical records, financial information
   - Low-risk (lr/): public content, product information, general text

3. **Target** (`vocabulary/target/`): Who is affected
   - High-risk (hr/): employees, citizens, patients, students
   - Low-risk (lr/): consumers, users, general public

### Evaluation Data

The `tests/evaluate/` directory contains full-text evaluation examples:
- **high_risk/**: Texts describing high-risk AI applications
- **low_risk/**: Texts describing low-risk AI applications

For detailed classification rules, see `tests/evaluate/RULES.md`

**IMPORTANT**: All data in both `vocabulary/` and `tests/evaluate/` is completely artificial and synthetic. See DISCLAIMER.md files in each directory.

### Classification Logic

A text is classified as **high-risk** if:
- Count threshold: More than 1 high-risk sentence (configurable)
- Percentage threshold: More than 10% high-risk sentences (configurable)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd texty

# Build the container (auto-detects Docker/Podman)
make build
```

## Testing and Validation

### Basic Testing

```bash
# Test on a high-risk sample
make evaluate FILE=./tests/evaluate/high_risk/biometric/01108973d97e4fa2.txt

# Test on a low-risk sample  
make evaluate FILE=./tests/evaluate/low_risk/research/35a6e7909c4733a2.txt

# Test multiple samples
for file in tests/evaluate/high_risk/*/*.txt; do
    echo "Testing: $file"
    make evaluate FILE="$file"
done
```

### Performance Benchmarks

```bash
# Run comprehensive benchmarks
make test

# This will show:
# - Unit test results (4 tests)
# - Performance metrics (inference time)
# - Memory usage
# - Accuracy metrics (precision, recall, F1-score)
```

### Expected Results

**High-Risk Text Example:**
```
==================================================
File: ./tests/evaluate/high_risk/biometric/example.txt
Classification: HIGH-RISK
High-risk sentences: 23/23
Inference time: 36.27ms
==================================================
```

**Low-Risk Text Example:**
```
==================================================
File: ./tests/evaluate/low_risk/research/example.txt
Classification: LOW-RISK
High-risk sentences: 0/26
Inference time: 36.31ms
==================================================
```

### Validate Dataset Quality

```bash
# Check file counts
echo "Vocabulary files: $(find vocabulary -name "*.txt" -type f | wc -l)"
echo "Evaluation files: $(find tests/evaluate -name "*.txt" | wc -l)"

# Check evaluation file word counts
for file in tests/evaluate/high_risk/*/*.txt; do
    words=$(wc -w < "$file")
    if [ $words -lt 100 ]; then
        echo "Warning: $file has only $words words (minimum: 100)"
    fi
done
```

## Development Workflow

### Web Interface

```bash
# Build the web container
make web

# Start the web server
make web-run

# The web interface provides:
# - Real-time text classification  
# - Visual sentence-by-sentence risk analysis
# - Color-coded risk visualization
# - Classification reasoning and metrics
# - Available at http://localhost:8000
```

### Custom Classification

```bash
# Classify your own text file
echo "Your text content here..." > custom_text.txt
make evaluate FILE=./custom_text.txt

# The classification uses default thresholds from config.yaml:
# - Confidence threshold: 0.7
# - Count threshold: 1 high-risk sentence
# - Percentage threshold: 10% high-risk sentences
```


## Architecture Details

### Container Runtime Detection

The Makefile automatically detects and uses either Docker or Podman:
- **Docker**: Standard container runtime
- **Podman**: Rootless container alternative

### Volume Mounts

- `./artifacts/` → `/data/` (models and results)
- `./vocabulary/` → `/app/vocabulary/` (training vocabulary data)
- `./tests/` → `/app/tests/` (evaluation test data)
- `./` → `/app/` (application code)

### Environment Variables

- `FEATURES`: Number of TF-IDF features (default: `5000`)
- `NGRAMS`: Maximum n-gram size (default: `2`)
- `EXCLUDE`: Categories to exclude from training
- `MODEL_PATH`: Path to trained model (default: `/data/models/base.latest.pkl`)
- `LOG_LEVEL`: Logging verbosity (default: `INFO`)

## Troubleshooting

### Common Issues

**Model not found error:**
```bash
# Ensure model is trained first
make train

# Check if model exists
ls -la artifacts/models/base.latest.pkl
```

**Permission errors with Podman:**
```bash
# Check SELinux labels in volume mounts
# The Makefile handles this with :Z flags
```

**NLTK data errors:**
```bash
# Rebuild container to update NLTK data
make build
```

### Performance Issues

If inference time exceeds 50ms:
1. Check text length (optimized for 1000 words)
2. Verify system resources (4-core CPU recommended)
3. Reduce features in `./vocabulary/config.yaml` (tfidf.max_features)
4. Disable vocabulary sources in config.yaml

### Accuracy Issues

If classification accuracy is poor:
1. Increase features in `./vocabulary/config.yaml` (tfidf.max_features)
2. Enable trigrams in config.yaml (tfidf.ngram_range: [1, 3])
3. Adjust vocabulary sources in config.yaml (disable problematic ones)
4. Increase synthetic samples in config.yaml (synthetic_generation.samples_per_combination)

## File Structure

```
texty/
├── Makefile              # Build and run commands
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── CLAUDE.md             # AI assistant guidance
├── artifacts/            # Training artifacts
│   └── models/           # Trained models
├── src/                  # Source code
│   ├── trainer/          # Training module
│   ├── evaluator/        # Evaluation module
│   ├── generator/        # Synthetic data generation
│   ├── web/              # Web interface
│   └── helpers/          # Utility functions
├── vocabulary/           # Training vocabulary (SYNTHETIC DATA)
│   ├── function/         # AI capabilities
│   ├── what/             # Data types
│   ├── target/           # Affected groups
│   └── DISCLAIMER.md     # Artificial data notice
└── tests/
    └── evaluate/         # Evaluation texts (SYNTHETIC DATA)
        ├── high_risk/    # High-risk test cases
        ├── low_risk/     # Low-risk test cases
        ├── RULES.md      # Classification rules
        └── DISCLAIMER.md # Artificial data notice
```

## Performance Targets

- ✅ **Inference Time**: <50ms for 1000-word texts
- ✅ **Memory Usage**: <150MB total
- ✅ **Training Time**: <5 minutes for 1000-10000 samples
- ✅ **Accuracy**: High precision and recall (F1-score >0.95)

## License

This is a prototype system for text classification research and development.
