# Rust WebRS Makefile for building native and WASM versions

.PHONY: help build build-wasm build-native test test-wasm clean install-tools export-onnx default copy-onnx run

# Default target - copy ONNX and build
default: copy-onnx build

# Copy ONNX artifacts from root artifacts directory
copy-onnx:
	@echo "📋 Copying ONNX artifacts..."
	@if [ ! -d "../../artifacts/models/onnx" ]; then \
		echo "❌ Error: ONNX artifacts not found at ../../artifacts/models/onnx/"; \
		echo "⚠️  Please run 'make onnx' in the root of the repository first."; \
		exit 1; \
	fi
	@if [ ! -f "../../artifacts/models/onnx/base_model_classifier.onnx" ]; then \
		echo "❌ Error: Required ONNX model files not found."; \
		echo "⚠️  Please run 'make onnx' in the root of the repository first."; \
		exit 1; \
	fi
	@mkdir -p ./build/models
	@cp -r ../../artifacts/models/onnx/* ./build/models/
	@echo "✅ ONNX artifacts copied to ./build/models/"
	@echo "Files copied:"
	@ls -la ./build/models/

# Help target
help:
	@echo "Texty WebRS - Rust/WASM Text Classification"
	@echo "==========================================="
	@echo ""
	@echo "🦀 BUILD TARGETS:"
	@echo "  build         - Build both native and WASM versions"
	@echo "  build-native  - Build native Rust library"
	@echo "  build-wasm    - Build WASM version for web"
	@echo ""
	@echo "🧪 TESTING:"
	@echo "  test          - Run all tests"
	@echo "  test-wasm     - Test WASM build"
	@echo ""
	@echo "🔧 TOOLS:"
	@echo "  install-tools - Install required tools (wasm-pack, etc.)"
	@echo "  copy-onnx     - Copy ONNX artifacts from ../../artifacts/models/onnx/"
	@echo "  run           - Start development server (requires build first)"
	@echo ""
	@echo "🧹 CLEANUP:"
	@echo "  clean         - Clean build artifacts"

# Install required tools for WASM development
install-tools:
	@echo "📦 Installing required tools..."
	@if ! command -v wasm-pack >/dev/null 2>&1; then \
		echo "Installing wasm-pack..."; \
		curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh; \
	else \
		echo "✅ wasm-pack already installed"; \
	fi
	@if ! command -v cargo-generate >/dev/null 2>&1; then \
		echo "Installing cargo-generate..."; \
		cargo install cargo-generate; \
	else \
		echo "✅ cargo-generate already installed"; \
	fi

# Build both native and WASM versions
build: build-native build-wasm

# Build native Rust library
build-native:
	@echo "🦀 Building native Rust library..."
	cargo build --release
	@echo "✅ Native build complete!"

# Build WASM version
build-wasm: install-tools
	@echo "🌐 Building WASM version..."
	wasm-pack build --target web --out-dir pkg --features wasm
	@echo "✅ WASM build complete! Output in pkg/"

# Build for Node.js/bundler environments
build-wasm-bundler: install-tools
	@echo "📦 Building WASM for bundlers..."
	wasm-pack build --target bundler --out-dir pkg-bundler --features wasm

# Development build (faster compilation)
build-dev:
	@echo "🚧 Building development version..."
	cargo build
	@echo "✅ Development build complete!"

# Run tests
test:
	@echo "🧪 Running tests..."
	cargo test
	@echo "✅ Tests complete!"

# Test WASM build
test-wasm: build-wasm
	@echo "🌐 Testing WASM build..."
	wasm-pack test --headless --firefox --features wasm
	@echo "✅ WASM tests complete!"

# Export ONNX model from Python
export-onnx:
	@echo "🔄 Exporting Python model to ONNX..."
	@if [ ! -f "../../artifacts/models/base.latest.pkl" ]; then \
		echo "❌ No trained model found. Run 'make train' in the root directory first."; \
		exit 1; \
	fi
	@mkdir -p ../../artifacts/models/onnx
	@echo "🐳 Running ONNX export in container..."
	cd ../.. && $(MAKE) onnx-container
	@echo "✅ ONNX export complete! Files saved to ../../artifacts/models/onnx/"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	cargo clean
	rm -rf pkg pkg-bundler build
	@echo "✅ Clean complete!"

# Generate documentation
docs:
	@echo "📚 Generating documentation..."
	cargo doc --no-deps --features wasm
	@echo "✅ Documentation generated! Open target/doc/texty_webrs/index.html"

# Run clippy for linting
lint:
	@echo "🔍 Running clippy..."
	cargo clippy --all-targets --features wasm -- -D warnings
	@echo "✅ Lint complete!"

# Format code
fmt:
	@echo "🎨 Formatting code..."
	cargo fmt
	@echo "✅ Formatting complete!"

# Check code without building
check:
	@echo "🔍 Checking code..."
	cargo check --features wasm
	@echo "✅ Check complete!"

# Benchmark (if benchmark tests are added)
bench:
	@echo "⚡ Running benchmarks..."
	cargo bench
	@echo "✅ Benchmarks complete!"

# Watch for changes and rebuild
watch:
	@if ! command -v cargo-watch >/dev/null 2>&1; then \
		echo "Installing cargo-watch..."; \
		cargo install cargo-watch; \
	fi
	@echo "👀 Watching for changes..."
	cargo watch -x "build --features wasm"

# Run development server (serves the built WASM)
run:
	@echo "🚀 Starting WebRS development server..."
	@if [ ! -d "./build" ]; then \
		echo "❌ Build directory not found. Run 'make webrs' first."; \
		exit 1; \
	fi
	@if [ ! -d "./pkg" ]; then \
		echo "❌ WASM package not found. Building now..."; \
		$(MAKE) build-wasm; \
	fi
	@echo "Starting simple HTTP server on port 8080..."
	@echo "Open http://localhost:8080 in your browser"
	@echo "Press Ctrl+C to stop"
	python3 -m http.server 8080 || python -m http.server 8080 || echo "Error: Python not found. Install Python to run the dev server."