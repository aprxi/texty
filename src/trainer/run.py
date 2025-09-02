#!/usr/bin/env python3
"""Base AI risk classifier using proven TF-IDF + Naive Bayes approach with config support.

This module trains a text classification model to identify high-risk AI applications
based on vocabulary analysis. It uses a unified approach combining TF-IDF vectorization
with Multinomial Naive Bayes classification.
"""

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.helpers.stopwords_combined import STOPWORDS_LIST


# Constants
DEFAULT_MAX_FEATURES = 5000
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_MIN_DF = 1
DEFAULT_MAX_DF = 1.0
DEFAULT_ALPHA = 1.0
DEFAULT_SYNTHETIC_SAMPLES = 800000
DEFAULT_MIN_LINE_LENGTH = 10
HIGH_RISK_LABEL = 1
LOW_RISK_LABEL = 0


def load_config(config_path: Optional[Path] = None) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Optional path to configuration file. If None, searches default locations.
        
    Returns:
        Dictionary containing configuration settings.
    """
    if config_path is None:
        # Default locations to check
        possible_paths = [
            Path("./vocabulary/config.yaml"),
            Path("./config.yaml"),
            Path("/app/vocabulary/config.yaml"),  # Docker path
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        else:
            print("Warning: No config.yaml found, using defaults")
            return get_default_config()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded configuration from {config_path}")
    return config


def get_default_config() -> Dict:
    """Return default configuration if no config file found.
    
    Returns:
        Dictionary containing default configuration settings.
    """
    return {
        'vocabulary_sources': [
            {'name': 'function', 'enabled': True, 'weight': 1.0},
            {'name': 'what', 'enabled': True, 'weight': 1.0},
            {'name': 'target', 'enabled': True, 'weight': 1.0},
            {'name': 'generated', 'enabled': True, 'weight': 1.0},
        ],
        'training': {
            'tfidf': {
                'max_features': DEFAULT_MAX_FEATURES,
                'ngram_range': list(DEFAULT_NGRAM_RANGE),
                'min_df': DEFAULT_MIN_DF,
                'max_df': DEFAULT_MAX_DF,
                'lowercase': True,
                'use_stopwords': True
            },
            'naive_bayes': {
                'alpha': DEFAULT_ALPHA,
                'fit_prior': True,
                'class_prior': None
            },
            'data': {
                'min_line_length': DEFAULT_MIN_LINE_LENGTH,
                'skip_comments': True,
                'split_sentences': True
            }
        },
        'synthetic_generation': {
            'enabled': True,
            'synthetic_samples': DEFAULT_SYNTHETIC_SAMPLES,
            'random_seed': 42
        }
    }


def load_vocabulary_from_file(
    file_path: Path,
    is_high_risk: bool,
    source_weight: float,
    data_config: Dict,
    texts: List[str],
    labels: List[int],
    weights: List[float]
) -> List[str]:
    """Load vocabulary from a single file and add to training data.
    
    Args:
        file_path: Path to vocabulary file.
        is_high_risk: Whether this file contains high-risk terms.
        source_weight: Weight to apply to samples from this source.
        data_config: Data processing configuration.
        texts: List to append text samples to.
        labels: List to append labels to.
        weights: List to append weights to.
        
    Returns:
        List of vocabulary terms loaded from the file.
    """
    terms = []
    min_line_length = data_config.get('min_line_length', DEFAULT_MIN_LINE_LENGTH)
    skip_comments = data_config.get('skip_comments', True)
    
    print(f"  Reading {file_path.name} - {'HIGH' if is_high_risk else 'LOW'} risk")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Split into sentences/lines and use each as training example
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                
                # Skip based on configuration
                if not line:
                    continue
                if skip_comments and line.startswith('#'):
                    continue
                if len(line) < min_line_length:
                    continue
                
                texts.append(line)
                labels.append(HIGH_RISK_LABEL if is_high_risk else LOW_RISK_LABEL)
                weights.append(source_weight)
                terms.append(line)
                
    except Exception as e:
        print(f"    Warning: Error reading {file_path}: {e}")
        
    return terms


def generate_synthetic_data(
    config: Dict,
    vocab_path: str,
    source_weight: float
) -> Tuple[List[str], List[int]]:
    """Generate synthetic training data using the external generator script.
    
    Args:
        config: Configuration dictionary containing synthetic generation settings.
        vocab_path: Path to vocabulary directory.
        source_weight: Weight to apply to synthetic samples.
        
    Returns:
        Tuple of (texts, labels) for synthetic data.
    """
    synthetic_config = config.get('synthetic_generation', {})
    num_samples = synthetic_config.get('synthetic_samples', DEFAULT_SYNTHETIC_SAMPLES)
    samples_per_risk = num_samples // 2  # Split evenly between high/low risk
    
    syn_texts = []
    syn_labels = []
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run the generator script
        generator_path = os.path.join(os.path.dirname(__file__), 'generate.py')
        
        cmd = [
            'python', generator_path,
            '--limit', str(samples_per_risk),
            '--output-dir', temp_dir,
            '--vocab-dir', vocab_path,
            '--seed', str(synthetic_config.get('random_seed', 42))
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Generator output: {result.stdout[:500]}...")
        except subprocess.CalledProcessError as e:
            print(f"Error running generator: {e.stderr}")
            return syn_texts, syn_labels
        
        # Read generated files
        hr_file = os.path.join(temp_dir, 'hr', 'synthetic_combinations.txt')
        lr_file = os.path.join(temp_dir, 'lr', 'synthetic_combinations.txt')
        
        # Read high-risk samples
        if os.path.exists(hr_file):
            with open(hr_file, 'r', encoding='utf-8') as f:
                hr_texts = [line.strip() for line in f if line.strip()]
                syn_texts.extend(hr_texts)
                syn_labels.extend([HIGH_RISK_LABEL] * len(hr_texts))
        
        # Read low-risk samples
        if os.path.exists(lr_file):
            with open(lr_file, 'r', encoding='utf-8') as f:
                lr_texts = [line.strip() for line in f if line.strip()]
                syn_texts.extend(lr_texts)
                syn_labels.extend([LOW_RISK_LABEL] * len(lr_texts))
    
    print(f"Generated {len(syn_texts)} synthetic examples")
    print(f"  High-risk: {sum(syn_labels)}, Low-risk: {len(syn_labels) - sum(syn_labels)}")
    
    return syn_texts, syn_labels


def collect_training_data(
    base_dir: str, 
    config: Dict, 
    exclude_categories: Optional[List[str]] = None
) -> Tuple[List[str], List[int], List[float]]:
    """Collect training data based on configuration.
    
    Args:
        base_dir: Base directory containing vocabulary files.
        config: Configuration dictionary.
        exclude_categories: Optional list of categories to exclude from training.
        
    Returns:
        Tuple of (texts, labels, weights) for training.
    """
    texts: List[str] = []
    labels: List[int] = []
    weights: List[float] = []
    excluded: Set[str] = set(exclude_categories or [])
    
    # Get enabled vocabulary sources from config
    enabled_sources = [
        source for source in config.get('vocabulary_sources', [])
        if source.get('enabled', True) and source['name'] not in excluded
    ]
    
    print(f"Loading vocabulary from sources: {[s['name'] for s in enabled_sources]}")
    
    # Use base_dir as vocabulary directory
    vocab_dir = Path(base_dir)
    
    data_config = config.get('training', {}).get('data', {})
    
    # First, load vocabulary for synthetic generation
    loaded_vocabulary: Dict[str, Dict[str, List[str]]] = {}
    
    for source in enabled_sources:
        source_name = source['name']
        source_weight = source.get('weight', 1.0)
        
        # Check if this is synthetic generation
        if source.get('synthetic', False) or source_name == 'generated':
            # Handle synthetic generation separately
            continue
            
        # Look in vocabulary directory
        source_dir = vocab_dir / source_name
        
        if source_dir.exists():
            print(f"Processing {source_name} from {source_dir} (weight: {source_weight})")
            
            # Load vocabulary for synthetic generation
            if source_name in ['function', 'what', 'target']:
                if source_name not in loaded_vocabulary:
                    loaded_vocabulary[source_name] = {'hr': [], 'lr': []}
            
            # Find all txt files in hr and lr subdirectories
            for risk_level in ['hr', 'lr']:
                risk_dir = source_dir / risk_level
                if risk_dir.exists():
                    is_high_risk = (risk_level == 'hr')
                    
                    for txt_file in risk_dir.rglob("*.txt"):
                        terms = load_vocabulary_from_file(
                            txt_file, is_high_risk, source_weight, 
                            data_config, texts, labels, weights
                        )
                        
                        # Store for synthetic generation
                        if source_name in ['function', 'what', 'target']:
                            loaded_vocabulary[source_name][risk_level].extend(terms)
    
    # Generate synthetic data if enabled
    for source in enabled_sources:
        if source.get('synthetic', False) or source['name'] == 'generated':
            if config.get('synthetic_generation', {}).get('enabled', True):
                print("\nGenerating synthetic training data using generator...")
                
                # Determine vocab path based on environment
                vocab_path = '/app/vocabulary' if os.path.exists('/app/vocabulary') else str(vocab_dir)
                
                syn_texts, syn_labels = generate_synthetic_data(
                    config, vocab_path, source.get('weight', 1.0)
                )
                
                # Add synthetic data with appropriate weight
                source_weight = source.get('weight', 1.0)
                texts.extend(syn_texts)
                labels.extend(syn_labels)
                weights.extend([source_weight] * len(syn_texts))
            break
    
    print(f"\nTotal collected {len(texts)} training examples")
    print(f"High-risk: {sum(labels)}, Low-risk: {len(labels) - sum(labels)}")
    
    return texts, labels, weights


def print_feature_statistics(
    classifier: Pipeline,
    labels: List[int]
) -> None:
    """Print statistics about the trained model features.
    
    Args:
        classifier: Trained classifier pipeline.
        labels: Training labels.
    """
    print("\nModel Statistics:")
    print("=" * 50)
    
    # Get TF-IDF vectorizer and feature names
    tfidf = classifier.named_steps['tfidf']
    nb = classifier.named_steps['nb']
    feature_names = tfidf.get_feature_names_out()
    
    print(f"Total vocabulary size: {len(feature_names)}")
    print(f"Features used: {tfidf.max_features}")
    print(f"N-gram range: {tfidf.ngram_range}")
    
    # Get most important features for high-risk classification
    if hasattr(nb, 'feature_log_prob_'):
        high_risk_log_probs = nb.feature_log_prob_[1]
        low_risk_log_probs = nb.feature_log_prob_[0]
        
        # Calculate feature importance (difference in log probabilities)
        feature_importance = high_risk_log_probs - low_risk_log_probs
        
        # Get top 10 features for high-risk
        top_indices = feature_importance.argsort()[-10:][::-1]
        
        print(f"\nTop 10 HIGH-RISK features:")
        print("-" * 30)
        for i, idx in enumerate(top_indices, 1):
            feature = feature_names[idx]
            importance = feature_importance[idx]
            hr_prob = high_risk_log_probs[idx]
            lr_prob = low_risk_log_probs[idx]
            print(f"{i:2d}. {feature:<25} (diff: {importance:.3f}, HR: {hr_prob:.3f}, LR: {lr_prob:.3f})")
        
        # Get bottom 10 features (most indicative of low-risk)
        bottom_indices = feature_importance.argsort()[:10]
        
        print(f"\nTop 10 LOW-RISK features:")
        print("-" * 30)
        for i, idx in enumerate(bottom_indices, 1):
            feature = feature_names[idx]
            importance = feature_importance[idx]
            hr_prob = high_risk_log_probs[idx]
            lr_prob = low_risk_log_probs[idx]
            print(f"{i:2d}. {feature:<25} (diff: {importance:.3f}, HR: {hr_prob:.3f}, LR: {lr_prob:.3f})")
        
        print(f"\nClass distribution in training:")
        print(f"  High-risk examples: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
        print(f"  Low-risk examples: {len(labels) - sum(labels)} ({(len(labels) - sum(labels))/len(labels)*100:.1f}%)")
        
        # Show class priors
        class_priors = nb.class_log_prior_
        print(f"\nClass priors (log probabilities):")
        print(f"  Low-risk prior: {class_priors[0]:.3f}")
        print(f"  High-risk prior: {class_priors[1]:.3f}")


def test_classifier(classifier: Pipeline) -> None:
    """Test the classifier on sample texts.
    
    Args:
        classifier: Trained classifier pipeline.
    """
    test_texts = [
        "AI voor medische claims automatisch beoordelen",
        "chatbot voor klantenservice", 
        "AI system for hiring decisions",
        "movie recommendation algorithm"
    ]
    
    print("\nTesting classifier:")
    predictions = classifier.predict(test_texts)
    probabilities = classifier.predict_proba(test_texts)
    
    for text, pred, prob in zip(test_texts, predictions, probabilities):
        risk = "HIGH-RISK" if pred == HIGH_RISK_LABEL else "LOW-RISK"
        confidence = max(prob)
        print(f"{risk:10} ({confidence:.2f}) - {text}")


def save_model(
    classifier: Pipeline,
    config: Dict,
    is_container: bool
) -> None:
    """Save the trained model and configuration.
    
    Args:
        classifier: Trained classifier pipeline.
        config: Configuration used for training.
        is_container: Whether running in container environment.
    """
    # Determine output path based on environment
    if is_container:
        output_path = "/data/artifacts/models/base.latest.pkl"
    else:
        output_path = "./artifacts/models/base.latest.pkl"
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(classifier, f)
    
    print(f"\nModel saved to {output_path}")
    
    # Save configuration used for training
    config_output = output_path.replace('.pkl', '_config.yaml')
    with open(config_output, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {config_output}")


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train base AI risk classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--exclude', type=str, help='Comma-separated list of categories to exclude')
    parser.add_argument('--features', type=int, help='Override number of features (from config)')
    parser.add_argument('--ngrams', type=int, help='Override maximum n-gram range (from config)')
    parser.add_argument('--alpha', type=float, help='Override Naive Bayes alpha (from config)')
    parser.add_argument('--samples', type=int, 
                       help='Override samples per combination for synthetic generation (from config)')
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)
    
    # Parse exclude categories
    exclude_categories = []
    if args.exclude:
        exclude_categories = [cat.strip() for cat in args.exclude.split(',')]
        print(f"Excluding categories: {exclude_categories}")
    
    # Override config with command line arguments if provided
    if args.features:
        config['training']['tfidf']['max_features'] = args.features
    if args.ngrams:
        config['training']['tfidf']['ngram_range'] = [1, args.ngrams]
    if args.alpha:
        config['training']['naive_bayes']['alpha'] = args.alpha
    if args.samples:
        if 'synthetic_generation' not in config:
            config['synthetic_generation'] = {}
        config['synthetic_generation']['synthetic_samples'] = args.samples
    
    # Extract settings from config
    tfidf_config = config['training']['tfidf']
    nb_config = config['training']['naive_bayes']
    
    # Detect environment
    is_container = os.path.exists("/app/vocabulary")
    
    # Collect training data
    if is_container:
        texts, labels, weights = collect_training_data("/app/vocabulary", config, exclude_categories)
    else:
        texts, labels, weights = collect_training_data("./vocabulary", config, exclude_categories)
    
    # Validate we have training data
    if not texts:
        print("Error: No training data collected!")
        sys.exit(1)
    
    # Create base classifier
    print(f"\nTraining with configuration:")
    print(f"  Features: {tfidf_config['max_features']}")
    print(f"  N-gram range: {tfidf_config['ngram_range']}")
    print(f"  Min DF: {tfidf_config.get('min_df', DEFAULT_MIN_DF)}")
    print(f"  Max DF: {tfidf_config.get('max_df', DEFAULT_MAX_DF)}")
    print(f"  Alpha: {nb_config['alpha']}")
    
    # Create pipeline
    classifier = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=tfidf_config.get('lowercase', True),
            stop_words=STOPWORDS_LIST if tfidf_config.get('use_stopwords', True) else None,
            max_features=tfidf_config['max_features'],
            ngram_range=tuple(tfidf_config['ngram_range']),
            min_df=tfidf_config.get('min_df', DEFAULT_MIN_DF),
            max_df=tfidf_config.get('max_df', DEFAULT_MAX_DF)
        )),
        ('nb', MultinomialNB(
            alpha=nb_config['alpha'],
            fit_prior=nb_config.get('fit_prior', True),
            class_prior=nb_config.get('class_prior', None)
        ))
    ])
    
    # Train with sample weights if available
    print("\nTraining base classifier...")
    if all(w == 1.0 for w in weights):
        # No weighting needed
        classifier.fit(texts, labels)
    else:
        # Apply sample weights
        sample_weights = np.array(weights)
        print(f"Using weighted training (unique weights: {sorted(set(weights))})")
        classifier.fit(texts, labels, nb__sample_weight=sample_weights)
    
    # Print feature statistics
    print_feature_statistics(classifier, labels)
    
    # Test on sample texts
    test_classifier(classifier)
    
    # Save model and configuration
    save_model(classifier, config, is_container)


if __name__ == "__main__":
    main()