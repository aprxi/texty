#!/bin/bash
# Test the patch vocabulary improvements

echo "=========================================="
echo "Testing Vocabulary Patch"
echo "=========================================="

# Check if patch directory exists
if [ ! -d "./vocabulary/patch" ]; then
    echo "Error: ./vocabulary/patch directory not found!"
    exit 1
fi

# Copy patch vocabulary to training directory (as additional vocabulary)
echo "Setting up patch vocabulary..."

# Create a temporary training directory with patches
PATCH_DIR="./datasets/u0_patched"
echo "Creating patched dataset at $PATCH_DIR..."

# Copy original training data
cp -r ./datasets/u0 $PATCH_DIR

# Add patch vocabulary to the training data
echo "Adding high-risk patches..."
mkdir -p $PATCH_DIR/train/function/hr/patch
cat ./vocabulary/patch/hr/*.txt > $PATCH_DIR/train/function/hr/patch/combined_hr_patch.txt

echo "Adding low-risk patches..."
mkdir -p $PATCH_DIR/train/function/lr/patch
cat ./vocabulary/patch/lr/*.txt > $PATCH_DIR/train/function/lr/patch/combined_lr_patch.txt

echo ""
echo "Patch vocabulary added:"
echo "- High-risk patches: $(wc -l < $PATCH_DIR/train/function/hr/patch/combined_hr_patch.txt) lines"
echo "- Low-risk patches: $(wc -l < $PATCH_DIR/train/function/lr/patch/combined_lr_patch.txt) lines"

echo ""
echo "=========================================="
echo "Training model with patches..."
echo "=========================================="

# Train new model with patches (excluding generated data for cleaner results)
echo "Training u0_patched model..."
docker run --rm \
  -v "$(pwd)/datasets/u0_patched:/data" \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/vocabulary:/app/vocabulary" \
  texty-classifier \
  python -m src.trainer.run \
    --data-dir /data \
    --output /data/models/patched_model.pkl \
    --features 7500 \
    --ngrams 2 \
    --exclude generated

echo ""
echo "=========================================="
echo "Evaluating improvements..."
echo "=========================================="

# Run evaluation on original model
echo "Evaluating original model..."
make evaluate MODEL=u0 > ./vocabulary/patch/original_results.txt 2>&1

# Run evaluation on patched model
echo "Evaluating patched model..."
docker run --rm \
  -v "$(pwd)/datasets/u0_patched:/data" \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/vocabulary:/app/vocabulary" \
  texty-classifier \
  python -m src.evaluator.run \
    --model /data/models/patched_model.pkl \
    --eval-dir /app/vocabulary/evaluate \
    --debug > ./vocabulary/patch/patched_results.txt 2>&1

echo ""
echo "=========================================="
echo "Results Comparison"
echo "=========================================="

echo "ORIGINAL MODEL:"
grep "Overall accuracy:" ./vocabulary/patch/original_results.txt
grep "High-risk files:" ./vocabulary/patch/original_results.txt
grep "Low-risk files:" ./vocabulary/patch/original_results.txt
grep "FALSE NEGATIVES" ./vocabulary/patch/original_results.txt
grep "FALSE POSITIVES" ./vocabulary/patch/original_results.txt

echo ""
echo "PATCHED MODEL:"
grep "Overall accuracy:" ./vocabulary/patch/patched_results.txt
grep "High-risk files:" ./vocabulary/patch/patched_results.txt
grep "Low-risk files:" ./vocabulary/patch/patched_results.txt
grep "FALSE NEGATIVES" ./vocabulary/patch/patched_results.txt
grep "FALSE POSITIVES" ./vocabulary/patch/patched_results.txt

echo ""
echo "=========================================="
echo "Category-specific improvements:"
echo "=========================================="

echo "Education category (had 17 FN):"
grep -A2 "education:" ./vocabulary/patch/patched_results.txt | head -3

echo ""
echo "Personal category (had 28 FP):"
grep -A2 "personal:" ./vocabulary/patch/patched_results.txt | head -3

echo ""
echo "Research category (had 20 FP):"
grep -A2 "research:" ./vocabulary/patch/patched_results.txt | head -3

echo ""
echo "=========================================="
echo "Full results saved in:"
echo "  - ./vocabulary/patch/original_results.txt"
echo "  - ./vocabulary/patch/patched_results.txt"
echo "  - Model: ./datasets/u0_patched/models/patched_model.pkl"
echo "=========================================="

echo ""
echo "To permanently apply patches if successful:"
echo "  1. Review the results above"
echo "  2. If improved, copy vocabulary to main training:"
echo "     cp ./vocabulary/patch/hr/*.txt ./datasets/u0/train/function/hr/"
echo "     cp ./vocabulary/patch/lr/*.txt ./datasets/u0/train/function/lr/"
echo "  3. Retrain main model: make train MODEL=u0"