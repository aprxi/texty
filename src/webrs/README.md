# Texty WebRS - Rust/WASM Text Classification

A high-performance Rust implementation of the text classification system that can compile to WebAssembly (WASM) for client-side use.

## Features

- 🦀 **Native Rust Performance**: Fast text classification with minimal overhead
- 🌐 **WebAssembly Support**: Run in browsers without server dependencies
- 🔄 **ONNX Integration**: Uses ONNX models exported from the Python scikit-learn pipeline
- 📊 **Same API**: Compatible results with the Python implementation
- ⚡ **Sub-50ms Performance**: Optimized for real-time classification
- 🎯 **Sentence-Level Analysis**: Detailed sentence-by-sentence classification

## Quick Start

### 1. Export Python Model to ONNX

First, train a Python model and export it to ONNX format:

```bash
# From the project root
make train                    # Train the Python model
make webrs-export            # Export to ONNX format
```

### 2. Build Rust Library

```bash
# Build both native and WASM versions
make webrs-build

# Or build separately
cd src/webrs
make build-native            # Native Rust library
make build-wasm             # WASM for browsers
```

### 3. Use in Your Application

#### Native Rust

```rust
use texty_webrs::{TextClassifier, ClassifierConfig, classifier::ClassifierBuilder};
use std::fs;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the exported model
    let classifier = ClassifierBuilder::new()
        .confidence_threshold(0.7)
        .count_threshold(1)
        .percentage_threshold(0.1)
        .build_from_files(
            "./models/base_model_classifier.onnx",
            "./models/base_model_vectorizer.json"
        )?;
    
    // Classify text
    let text = "We use facial recognition to identify employees automatically.";
    let result = classifier.classify(text)?;
    
    println!("Classification: {}", result.classification);
    println!("Confidence: {:.3}", result.confidence);
    println!("High-risk sentences: {}/{}", result.high_risk_sentences, result.total_sentences);
    
    Ok(())
}
```

#### WebAssembly (Browser)

```javascript
import init, { WasmTextClassifier } from './pkg/texty_webrs.js';

async function main() {
    // Initialize WASM module
    await init();
    
    // Load model files (you'll need to fetch these)
    const onnxBytes = await fetch('./models/base_model_classifier.onnx')
        .then(r => r.arrayBuffer())
        .then(b => new Uint8Array(b));
    
    const vectorizerConfig = await fetch('./models/base_model_vectorizer.json')
        .then(r => r.text());
    
    // Create classifier
    const classifier = new WasmTextClassifier();
    classifier.load_model_from_bytes(onnxBytes, vectorizerConfig);
    
    // Classify text
    const text = "We use facial recognition to identify employees automatically.";
    const result = classifier.classify(text);
    
    console.log('Classification:', result.classification);
    console.log('Confidence:', result.confidence);
}

main();
```

## Architecture

### Components

1. **lib.rs**: Main library interface and WASM bindings
2. **classifier.rs**: Core classification logic and ONNX inference
3. **preprocessing.rs**: Text preprocessing and TF-IDF vectorization
4. **utils.rs**: Utility functions for logging and performance monitoring

### Model Pipeline

```
Text Input
    ↓
Sentence Tokenization
    ↓
Text Preprocessing (normalize, clean)
    ↓
TF-IDF Vectorization (Rust implementation)
    ↓
ONNX Model Inference
    ↓
Aggregation & Thresholding
    ↓
Classification Result
```

## Performance

- **Target**: <50ms for 1000-word texts
- **Memory**: <50MB including model
- **WASM Size**: ~2MB compressed
- **Browser Support**: All modern browsers with WASM support

## Development

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install required tools
cd src/webrs
make install-tools
```

### Commands

```bash
# Development
make build-dev              # Fast development build
make test                   # Run tests
make check                  # Check without building
make lint                   # Run clippy linter
make fmt                    # Format code

# Production builds
make build-native           # Optimized native build
make build-wasm            # WASM for browsers
make build-wasm-bundler    # WASM for bundlers (webpack, etc.)

# Documentation
make docs                   # Generate documentation

# Utilities
make watch                  # Watch for changes and rebuild
make bench                  # Run benchmarks
make clean                  # Clean build artifacts
```

## Configuration

### ClassifierConfig

```rust
pub struct ClassifierConfig {
    pub confidence_threshold: f32,    // Minimum confidence for high-risk (0.0-1.0)
    pub count_threshold: usize,       // Number of high-risk sentences needed
    pub percentage_threshold: f32,    // Percentage of high-risk sentences needed (0.0-1.0)
    pub use_weighted_scoring: bool,   // Use confidence-weighted scoring
}
```

### Default Values

- **confidence_threshold**: 0.7
- **count_threshold**: 1
- **percentage_threshold**: 0.1 (10%)
- **use_weighted_scoring**: false

## ONNX Model Requirements

The Rust implementation expects:

1. **ONNX Model**: Exported scikit-learn Naive Bayes classifier
   - Input: `[batch_size, n_features]` (Float32)
   - Outputs: 
     - Index 0: Predicted classes
     - Index 1: Class probabilities

2. **Vectorizer Config**: JSON file with TF-IDF vectorizer parameters
   - Vocabulary mapping
   - Feature names
   - Preprocessing parameters
   - N-gram settings

## Limitations

- Currently supports Naive Bayes classifiers only
- TF-IDF vectorization is simplified (no IDF weighting yet)
- Limited text preprocessing compared to full NLTK pipeline
- No custom tokenizers (uses regex-based approach)

## Roadmap

- [ ] Full IDF weighting support
- [ ] Advanced tokenization (spaCy-like)
- [ ] Support for other model types (SVM, Random Forest)
- [ ] Streaming classification for large texts
- [ ] Model quantization for smaller WASM size
- [ ] WebWorker integration for non-blocking UI

## Contributing

1. Follow Rust conventions and formatting
2. Add tests for new features
3. Update documentation
4. Ensure WASM compatibility for browser features