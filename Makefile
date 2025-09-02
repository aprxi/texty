# Text Classification System Makefile with Docker/Podman Detection
.PHONY: help build train evaluate clean web web-run webrs webrs-run onnx

# Detect container runtime (Docker or Podman)
DOCKER_CHECK := $(shell command -v docker 2>/dev/null)
PODMAN_CHECK := $(shell command -v podman 2>/dev/null)
ifdef DOCKER_CHECK
RUNTIME_CMD := docker
else ifdef PODMAN_CHECK
RUNTIME_CMD := podman
else
$(error Neither Docker nor Podman is installed. Please install one of them.)
endif

# Configuration
IMAGE_NAME = texty-classifier
CONTAINER_NAME = texty-classifier

# Model name for evaluation (default: u0)
MODEL ?= u0

# Help targets
evaluate-help:
	@echo "📊 Evaluate Command Options"
	@echo "=========================="
	@echo ""
	@echo "Environment Variables:"
	@echo "  MODEL=<name>                - Dataset/model to use (default: u0)"
	@echo "  FILE=<path>                 - Single file to evaluate"
	@echo "  DEBUG=true                  - Show detailed analysis for single files"
	@echo "  VERBOSE=true                - Verbose output for dataset evaluation"
	@echo "  OUTPUT_JSON=<filename>      - Save results to JSON file"
	@echo ""
	@echo "Examples:"
	@echo "  make evaluate                           # Evaluate u0 dataset"
	@echo "  make evaluate MODEL=custom              # Use custom dataset"
	@echo "  make evaluate FILE=./test.txt           # Single file"
	@echo "  make evaluate FILE=./test.txt DEBUG=true"
	@echo "  make evaluate MODEL=u0 VERBOSE=true"
	@echo "  make evaluate MODEL=u0 OUTPUT_JSON=results.json"

train-help:
	@echo "🚂 Train Command Options"
	@echo "======================="
	@echo ""
	@echo "Environment Variables:"
	@echo "  MODEL=<name>                - Dataset to train on (default: u0)"
	@echo "  EXCLUDE=<categories>        - Comma-separated categories to exclude"
	@echo "                               Options: function,what,target,generated,patch"
	@echo "  FEATURES=<number>           - Number of TF-IDF features (default: 5000)"
	@echo "  NGRAMS=<number>             - Maximum n-gram size (default: 2)"
	@echo "  SAMPLES=<number>            - Synthetic samples per combination (default: 10)"
	@echo ""
	@echo "Examples:"
	@echo "  make train                              # Train on u0 with defaults"
	@echo "  make train MODEL=custom                 # Use custom dataset"
	@echo "  make train EXCLUDE=function,what        # Exclude categories"
	@echo "  make train FEATURES=10000               # More features"
	@echo "  make train NGRAMS=3                     # Use trigrams"
	@echo "  make train SAMPLES=50                   # More synthetic samples"
	@echo "  make train MODEL=u0 FEATURES=7500 NGRAMS=3 SAMPLES=25"

# Default target
help:
	@echo "Text Classification System - Container Commands ($(RUNTIME_CMD))"
	@echo "======================================================="
	@echo ""
	@echo "Detected runtime: $(RUNTIME_CMD)"
	@if [ -n "$(MODEL)" ]; then \
		echo "Active use case: $(MODEL)"; \
	fi
	@echo "Model: $(MODEL)"
	@echo ""
	@echo "🔨 BUILD & SETUP:"
	@echo "  build               - Build the classifier container image"
	@echo ""
	@echo "🚂 TRAINING:"
	@echo "  train               - Train base unified model (see vocabulary/config.yaml)"
	@echo "  train-help          - Show training options"
	@echo "  train-what       - Train only what model (legacy)"
	@echo "  train-target     - Train only target model (legacy)"
	@echo "  train-sentence   - Train only sentence model (legacy)"
	@echo ""
	@echo "📄 CLASSIFICATION:"
	@echo "  evaluate FILE=<path> - Classify a single text file"
	@echo "  evaluate             - Evaluate entire dataset"
	@echo "  test-samples         - Test classification on sample files"
	@echo ""
	@echo "🌐 WEB INTERFACE:"
	@echo "  web                  - Build web server container"
	@echo "  web-run              - Run web server with latest model"
	@echo ""
	@echo "🦀 RUST/WASM:"
	@echo "  webrs                - Build WebRS with ONNX artifacts (requires 'make onnx' first)"
	@echo "  webrs-run            - Run WebRS development server"
	@echo "  onnx                 - Export trained model to ONNX format (depends on train)"
	@echo ""
	@echo "  Single file parameters:"
	@echo "    DEBUG=true                  - Show detailed window analysis"
	@echo ""
	@echo "🧪 EVALUATION:"
	@echo "  evaluate            - Evaluate model on test dataset or single file"
	@echo "  evaluate-help       - Show evaluation options"
	@echo ""
	@echo "🌐 WEB INTERFACE:"
	@echo "  web                 - Build web container"
	@echo "  web-run             - Run web server (http://localhost:8000)"
	@echo ""
	@echo "🦀 RUST/WASM:"
	@echo "  webrs               - Build Rust/WASM classifier"
	@echo "  webrs-run           - Run Rust/WASM web server"
	@echo "  onnx                - Export model to ONNX format"
	@echo ""
	@echo "🧹 CLEANUP:"
	@echo "  clean               - Remove container and image"
	@echo ""
	@echo "📚 EXAMPLES:"
	@echo "  # Quick start:"
	@echo "  make build              # Build container"
	@echo "  make train              # Train model"
	@echo "  make evaluate           # Evaluate on test set"
	@echo "  make evaluate DEBUG=true # Show detailed analysis"
	@echo ""
	@echo "  # Web interface:"
	@echo "  make web                # Build web container"
	@echo "  make web-run            # Start server at http://localhost:8000"
	@echo ""
	@echo "  # Single file:"
	@echo "  make evaluate FILE=./test.txt"
	@echo ""

# Build the container image
build:
	@echo "🔨 Building text classifier image with $(RUNTIME_CMD)..."
	$(RUNTIME_CMD) build -t $(IMAGE_NAME) .
	@echo "✅ Build complete!"


# Train base unified model (recommended)
train: build
	@echo "🚂 Training base unified classifier using all training data..."
	@if [ -n "$(EXCLUDE)" ]; then \
		echo "   Excluding categories: $(EXCLUDE)"; \
	fi
	@if [ -n "$(FEATURES)" ]; then \
		echo "   Using $(FEATURES) features"; \
	fi
	@if [ -n "$(NGRAMS)" ]; then \
		echo "   Using n-grams up to $(NGRAMS)"; \
	fi
	@if [ -n "$(SAMPLES)" ]; then \
		echo "   Using $(SAMPLES) synthetic samples"; \
	fi
	@# Training data is now handled by src/trainer/run.py
	@# Ensure artifacts directory exists
	@mkdir -p ./artifacts/models
	@# Prepare exclude flag
	$(eval EXCLUDE_FLAG := $(if $(EXCLUDE),--exclude $(EXCLUDE),))
	$(eval FEATURES_FLAG := $(if $(FEATURES),--features $(FEATURES),))
	$(eval NGRAMS_FLAG := $(if $(NGRAMS),--ngrams $(NGRAMS),))
	$(eval SAMPLES_FLAG := $(if $(SAMPLES),--samples $(SAMPLES),))
ifeq ($(RUNTIME_CMD),podman)
	$(RUNTIME_CMD) run --rm \
		-v $(PWD):/app:Z \
		-v ./vocabulary:/app/vocabulary:Z \
		-v ./artifacts:/data/artifacts:Z \
		$(IMAGE_NAME) \
		python src/trainer/run.py $(EXCLUDE_FLAG) $(FEATURES_FLAG) $(NGRAMS_FLAG) $(SAMPLES_FLAG)
else
	$(RUNTIME_CMD) run --rm \
		-v $(PWD):/app \
		-v ./vocabulary:/app/vocabulary \
		-v ./artifacts:/data/artifacts \
		$(IMAGE_NAME) \
		python src/trainer/run.py $(EXCLUDE_FLAG) $(FEATURES_FLAG) $(NGRAMS_FLAG) $(SAMPLES_FLAG)
endif
	@echo "✅ Base model training complete! Model saved to ./artifacts/models/base.latest.pkl"


# Run classification on a file
run: build
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Error: Please specify a file to classify"; \
		echo "Usage: make run FILE=./data/test/example.txt"; \
		echo "       make run FILE=./data/test/example.txt DEBUG=true"; \
		exit 1; \
	fi
	@echo "📄 Classifying text from $(FILE)..."
	@# Check which models are available
	@if [ -f "./artifacts/models/base.latest.pkl" ]; then \
		echo "✅ Base model found - using unified classification"; \
		$(MAKE) run-base FILE=$(FILE) MODEL=$(MODEL) DEBUG=$(DEBUG); \
	else \
		echo "❌ Error: No model found at ./artifacts/models/base.latest.pkl"; \
		echo "Please run 'make train' first to train the model"; \
		exit 1; \
	fi


# Run classification using base unified model
run-base: build
	@if [ -z "$(FILE)" ]; then \
		echo "❌ Error: Please specify a file to classify"; \
		echo "Usage: make run FILE=./test.txt"; \
		exit 1; \
	fi
	@# Prepare flags - DEBUG is true by default for single file classification
	$(eval DEBUG_FLAG := $(if $(filter false,$(DEBUG)),,--debug))
ifeq ($(RUNTIME_CMD),podman)
	$(RUNTIME_CMD) run --rm \
		-v $(PWD):/app:Z \
		-v ./artifacts:/data/artifacts:Z \
		$(IMAGE_NAME) \
		python src/evaluator/classify.py --model /data/artifacts/models/base.latest.pkl \
		--input $(FILE) \
		$(DEBUG_FLAG)
else
	$(RUNTIME_CMD) run --rm \
		-v $(PWD):/app \
		-v ./artifacts:/data/artifacts \
		$(IMAGE_NAME) \
		python src/evaluator/classify.py --model /data/artifacts/models/base.latest.pkl \
		--input $(FILE) \
		$(DEBUG_FLAG)
endif


# Clean up
clean:
	@echo "🧹 Cleaning up..."
	-$(RUNTIME_CMD) rmi $(IMAGE_NAME)
	@echo "✅ Cleanup complete"

	@echo ""
	@echo "3. Development workflow:"
	@echo "   make shell          # Access container"
	@echo "   # Inside container:"
	@echo "   python train_model.py --help"
	@echo "   python classify.py --help"


# Evaluate model - works for both single files and full datasets
evaluate: build
	@if [ -n "$(FILE)" ]; then \
		echo "📄 Evaluating single file: $(FILE)"; \
		$(MAKE) run FILE=$(FILE) MODEL=$(MODEL) DEBUG=$(DEBUG); \
	else \
		echo "📊 Evaluating full dataset..."; \
		if [ -f "./artifacts/models/base.latest.pkl" ]; then \
			echo "✅ Base model found - using latest trained model"; \
			$(MAKE) evaluate-base MODEL=$(MODEL) VERBOSE=$(VERBOSE) OUTPUT_JSON=$(OUTPUT_JSON); \
		else \
			echo "❌ Error: No model found at ./artifacts/models/base.latest.pkl"; \
			echo "Please run 'make train' first to train the model"; \
			exit 1; \
		fi; \
	fi



# Evaluate using base unified model
evaluate-base: build
	@echo "📊 Evaluating with base unified model..."
	@if [ ! -f "./artifacts/models/base.latest.pkl" ]; then \
		echo "❌ Error: Base model not found at ./artifacts/models/base.latest.pkl"; \
		echo "Please run 'make train' first to train the model"; \
		exit 1; \
	fi
	@if [ ! -d "./tests/evaluate" ]; then \
		echo "❌ Error: Evaluation directory not found at ./tests/evaluate"; \
		exit 1; \
	fi
	@# Prepare flags
	$(eval VERBOSE_FLAG := $(if $(VERBOSE),-v,))
	$(eval DEBUG_FLAG := $(if $(DEBUG),--debug,))
	$(eval OUTPUT_FLAG := $(if $(OUTPUT_JSON),--output-json /data/results/$(OUTPUT_JSON),))
ifeq ($(RUNTIME_CMD),podman)
	$(RUNTIME_CMD) run --rm \
		-v $(PWD):/app:Z \
		-v ./artifacts:/data/artifacts:Z \
		-v ./vocabulary:/app/vocabulary:Z \
		$(IMAGE_NAME) \
		python src/evaluator/run.py --model /data/artifacts/models/base.latest.pkl \
		--eval-dir /app/tests/evaluate \
		$(VERBOSE_FLAG) $(DEBUG_FLAG) $(OUTPUT_FLAG)
else
	$(RUNTIME_CMD) run --rm \
		-v $(PWD):/app \
		-v ./artifacts:/data/artifacts \
		-v ./vocabulary:/app/vocabulary \
		$(IMAGE_NAME) \
		python src/evaluator/run.py --model /data/artifacts/models/base.latest.pkl \
		--eval-dir /app/tests/evaluate \
		$(VERBOSE_FLAG) $(DEBUG_FLAG) $(OUTPUT_FLAG)
endif


# Build web server container
web:
	@echo "🔨 Building web server container..."
	$(RUNTIME_CMD) build -t texty-web -f src/web/Dockerfile .
	@echo "✅ Web container build complete!"

# Run web server with latest trained model
web-run: web
	@echo "🌐 Starting web server with latest trained model..."
	@echo ""
	@# Check for available models in priority order and determine which to copy
	@if [ -f "./artifacts/models/base.latest.pkl" ]; then \
		echo "✅ Found latest base model - using for web server..."; \
		model_file="base.latest.pkl"; \
		source_path="./artifacts/models/base.latest.pkl"; \
	elif [ -f "$(ACTIVE_DATA_DIR)/models/base_model.pkl" ]; then \
		echo "✅ Found base unified model - using for web server..."; \
		model_file="base_model.pkl"; \
		source_path="$(ACTIVE_DATA_DIR)/models/base_model.pkl"; \
	elif [ -f "$(ACTIVE_DATA_DIR)/models/capability_model.pkl" ]; then \
		echo "✅ Found legacy capability model - using for web server..."; \
		model_file="capability_model.pkl"; \
		source_path="$(ACTIVE_DATA_DIR)/models/capability_model.pkl"; \
	else \
		echo "❌ Error: No trained models found"; \
		echo "Checked locations:"; \
		echo "  - ./artifacts/models/base.latest.pkl (recommended)"; \
		echo "  - $(ACTIVE_DATA_DIR)/models/base_model.pkl"; \
		echo "  - $(ACTIVE_DATA_DIR)/models/capability_model.pkl"; \
		echo ""; \
		echo "Please run 'make train' first to train a model"; \
		exit 1; \
	fi; \
	echo "📋 Model: $$model_file"; \
	echo "🚀 Web server will be available at: http://localhost:8000"; \
	echo "🛑 Press Ctrl+C to stop the server"; \
	echo ""; \
	$(RUNTIME_CMD) run --rm -it \
		-p 8000:8000 \
		-v "$$source_path":/app/model.pkl:ro \
		-e MODEL_PATH=/app/model.pkl \
		texty-web


# Rust/WASM targets

onnx: train
	@echo "🔄 Exporting Python model to ONNX format..."
	cd src/webrs && $(MAKE) export-onnx
	@echo "✅ ONNX export complete!"


# WebRS targets
webrs:
	@echo "🦀 Building WebRS with ONNX artifacts..."
	cd src/webrs && $(MAKE)
	@echo "✅ WebRS build complete!"

webrs-run:
	@echo "🚀 Running WebRS development server..."
	cd src/webrs && $(MAKE) run
	@echo "✅ WebRS server stopped!"


# Internal target: Export ONNX model using container
onnx-container: build
	@echo "🔄 Exporting ONNX model in container..."
	@mkdir -p artifacts/models/onnx
ifeq ($(RUNTIME_CMD),podman)
	$(RUNTIME_CMD) run --rm \
		-v $(PWD):/app:Z \
		-v ./artifacts:/data/artifacts:Z \
		$(IMAGE_NAME) \
		python src/trainer/export_onnx.py \
		--model /data/artifacts/models/base.latest.pkl \
		--output /data/artifacts/models/onnx \
		--name base_model
else
	$(RUNTIME_CMD) run --rm \
		-v $(PWD):/app \
		-v ./artifacts:/data/artifacts \
		$(IMAGE_NAME) \
		python src/trainer/export_onnx.py \
		--model /data/artifacts/models/base.latest.pkl \
		--output /data/artifacts/models/onnx \
		--name base_model
endif
