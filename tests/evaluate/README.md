# EU AI Act Evaluation Dataset

This directory contains a categorized evaluation dataset for testing AI safety classification systems based on the EU AI Act risk categories.

## 📁 Directory Structure

```
evaluate/
├── high_risk/                  # High-risk AI applications (400 files)
│   ├── biometric/              # Facial recognition, emotion detection (50 files)
│   ├── critical_infrastructure/# Utilities, traffic management (50 files)
│   ├── education/              # Student assessment, admission (50 files)
│   ├── essential_services/     # Credit, insurance, benefits (50 files)
│   ├── justice_democracy/      # Judicial AI, elections (50 files)
│   ├── law_enforcement/        # Profiling, evidence analysis (50 files)
│   ├── migration_border/       # Asylum, border control (50 files)
│   └── work_hr/                # Recruitment, performance (50 files)
│
├── low_risk/                   # Low-risk AI applications (348 files)
│   ├── content_creation/       # Text generation, translation (76 files)
│   ├── customer_service/       # Chatbots, support systems (62 files)
│   ├── entertainment/          # Gaming, media AI (60 files)
│   ├── personal/              # Personal use apps (50 files)
│   ├── productivity/          # Task management tools (50 files)
│   └── research/              # AI research/education (50 files)
│
├── CLAUDE.md                   # Instructions for AI assistants
├── RULES.md                    # EU AI Act classification rules
├── README.md                   # This file
│
└── 🛠️ Utility Scripts:
    ├── reorganize_evaluations.py  # Organize files by category
    ├── balance_categories.py       # Generate synthetic data for balance
    ├── dataset_statistics.py       # Generate statistics report
    ├── validate_dataset.py         # Validate classifications
    └── export_for_training.py      # Export dataset for model training
```

## 📊 Dataset Statistics

- **Total Files**: 748 evaluation texts
- **High-Risk**: 400 files across 8 EU AI Act categories
- **Low-Risk**: 348 files across 6 general categories
- **File Format**: Hash-named `.txt` files with evaluation tags
- **Language**: Dutch (Nederlands)
- **Average Text Length**: ~69 words
- **Validation Rate**: 100% (all files have proper evaluation tags)

## 🏷️ File Format

Each evaluation file follows this structure:

```xml
<START_TEXT>
[Original Dutch text describing an AI system or use case]
<END_TEXT>
<START_EVALUATION>
<CLASSIFICATION>high-risk|low-risk</CLASSIFICATION>
<CONFIDENCE>0.85-0.99</CONFIDENCE>
<REASONING>
[Explanation of classification based on RULES.md criteria]
</REASONING>
<END_EVALUATION>
```

## 🔧 Utility Scripts

### 1. **dataset_statistics.py**
Generate comprehensive statistics about the dataset:
```bash
python dataset_statistics.py
```
Outputs:
- Category distribution
- Confidence score analysis
- Word count distribution
- Validation status
- JSON report (`dataset_stats.json`)

### 2. **validate_dataset.py**
Validate all files against RULES.md criteria:
```bash
# Dry run (check only)
python validate_dataset.py

# Fix issues (update tags and move files)
python validate_dataset.py --fix
```

### 3. **reorganize_evaluations.py**
Reorganize flat structure into categories:
```bash
# Dry run
python reorganize_evaluations.py --dry-run

# Actually reorganize
python reorganize_evaluations.py
```

### 4. **balance_categories.py**
Generate synthetic data to balance categories:
```bash
python balance_categories.py
```
Target: 50 files per category

### 5. **export_for_training.py**
Export dataset in format suitable for model training:
```bash
# Export as CSV
python export_for_training.py --format csv

# Export as JSON
python export_for_training.py --format json

# Export split for train/test
python export_for_training.py --split 0.8
```

## 🎯 EU AI Act High-Risk Categories

Based on EU AI Act Article 6, these are considered high-risk:

1. **Biometric Systems** - Identification, categorization, emotion detection
2. **Critical Infrastructure** - Utilities, traffic, essential services management
3. **Education & Training** - Assessment, admission, fraud detection
4. **Work & HR** - Recruitment, performance evaluation, termination
5. **Essential Services** - Credit scoring, insurance, benefits, emergency
6. **Law Enforcement** - Profiling, lie detection, evidence analysis
7. **Migration & Border** - Asylum processing, border control, visa
8. **Justice & Democracy** - Judicial support, election influence

## 📈 Usage for Model Training

This dataset is designed for training classifiers that:
- Identify high-risk AI applications per EU AI Act
- Categorize AI systems by domain and risk level
- Provide confidence scores for classifications
- Generate explanations for risk assessments

### Integration with Training Pipeline

```python
# Example usage
from pathlib import Path
import json

# Load high-risk examples
high_risk_files = list(Path('high_risk').rglob('*.txt'))

# Load low-risk examples
low_risk_files = list(Path('low_risk').rglob('*.txt'))

# Process for training
for file_path in high_risk_files:
    with open(file_path, 'r') as f:
        content = f.read()
        # Extract text between <START_TEXT> and <END_TEXT>
        # Use for training with label='high-risk'
```

## 🔍 Data Quality Assurance

All files have been:
- ✅ Validated against RULES.md criteria
- ✅ Tagged with proper evaluation sections
- ✅ Organized into correct risk categories
- ✅ Balanced across categories (~50 files each)
- ✅ Checked for proper Dutch language content

## 📝 Notes

- Files use hash-based naming (e.g., `a4fb2ab2d718c9b6.txt`)
- Classification is at text-level (not sentence-level)
- All examples are synthetic, created for AI safety training
- Dataset follows defensive AI principles (identify risks, not enable them)

## 🚀 Quick Start

```bash
# 1. Check dataset statistics
python dataset_statistics.py

# 2. Validate classifications
python validate_dataset.py

# 3. Export for training
python export_for_training.py --format csv --split 0.8

# 4. Use exported data with your classifier
# See main project README for training instructions
```

## 📚 References

- EU AI Act Article 6 - High-risk AI systems
- RULES.md - Detailed classification criteria
- CLAUDE.md - Instructions for AI assistants working with this data