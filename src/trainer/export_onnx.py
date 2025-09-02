#!/usr/bin/env python3
"""Export trained scikit-learn model to ONNX format for use in Rust/WASM.

This module converts a trained text classification pipeline (TF-IDF + Naive Bayes)
to ONNX format, enabling deployment in production environments without Python.
"""

import argparse
import json
import pickle
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# Constants
DEFAULT_MODEL_NAME = "base_model"
ONNX_TARGET_OPSET = 12
TEST_BATCH_SIZE = 1


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert, may contain numpy types.
        
    Returns:
        Object with numpy types converted to native Python types.
    """
    if hasattr(obj, 'tolist'):  # numpy array (check this first)
        return obj.tolist()
    elif hasattr(obj, 'item') and obj.ndim == 0:  # numpy scalar (0-dimensional)
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj


def extract_vectorizer_metadata(vectorizer) -> Dict[str, Any]:
    """Extract metadata from TF-IDF vectorizer for reconstruction.
    
    Args:
        vectorizer: Scikit-learn TfidfVectorizer instance.
        
    Returns:
        Dictionary containing vectorizer configuration and vocabulary.
    """
    # Get vocabulary and feature names
    vocabulary = vectorizer.vocabulary_
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract stop words if available
    stop_words = None
    if hasattr(vectorizer, 'stop_words_') and vectorizer.stop_words_ is not None:
        stop_words = list(vectorizer.stop_words_)
    
    metadata = {
        'vocabulary': convert_numpy_types(vocabulary),
        'feature_names': feature_names.tolist(),
        'max_features': convert_numpy_types(vectorizer.max_features),
        'ngram_range': convert_numpy_types(vectorizer.ngram_range),
        'lowercase': vectorizer.lowercase,
        'stop_words': stop_words,
        'analyzer': vectorizer.analyzer,
        'token_pattern': vectorizer.token_pattern,
        'max_df': convert_numpy_types(vectorizer.max_df),
        'min_df': convert_numpy_types(vectorizer.min_df),
        'use_idf': vectorizer.use_idf,
        'smooth_idf': vectorizer.smooth_idf,
        'sublinear_tf': vectorizer.sublinear_tf,
        'norm': vectorizer.norm
    }
    
    # Add IDF weights if available
    if hasattr(vectorizer, 'idf_'):
        metadata['idf_weights'] = convert_numpy_types(vectorizer.idf_)
    
    return metadata


def test_onnx_model(
    onnx_path: Path, 
    n_features: int
) -> bool:
    """Test the exported ONNX model to ensure it works correctly.
    
    Args:
        onnx_path: Path to the ONNX model file.
        n_features: Number of features expected by the model.
        
    Returns:
        True if test successful, False otherwise.
    """
    try:
        print(f"Testing ONNX model...")
        
        # Create inference session
        sess = ort.InferenceSession(str(onnx_path))
        
        # Create test input (sparse TF-IDF representation)
        test_input = np.random.rand(TEST_BATCH_SIZE, n_features).astype(np.float32)
        
        # Make values sparse (most TF-IDF values are 0)
        test_input[test_input < 0.8] = 0.0
        
        # Get input name
        input_name = sess.get_inputs()[0].name
        
        # Run inference
        result = sess.run(None, {input_name: test_input})
        
        print(f"ONNX model test successful")
        
        # Check output shape and content
        if result and len(result) > 0:
            print(f"Output shape: {result[0].shape}")
            print(f"Output type: {result[0].dtype}")
            
            # For classification, we expect probabilities
            if len(result) > 1:
                print(f"Number of outputs: {len(result)}")
                
        return True
        
    except Exception as e:
        print(f"Error: ONNX model test failed: {e}")
        return False


def export_model_to_onnx(
    model_path: str, 
    output_dir: str, 
    model_name: str = DEFAULT_MODEL_NAME
) -> bool:
    """Export a scikit-learn pipeline to ONNX format.
    
    Args:
        model_path: Path to the trained .pkl model.
        output_dir: Directory to save ONNX model and metadata.
        model_name: Name for the output files.
        
    Returns:
        True if export successful, False otherwise.
    """
    print(f"Loading model from {model_path}...")
    
    # Load the trained model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    print(f"Model loaded successfully")
    print(f"Model type: {type(model)}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check if this is a pipeline with TF-IDF vectorizer
        if not hasattr(model, 'named_steps'):
            print("Warning: Model is not a pipeline, attempting direct conversion...")
            # Handle non-pipeline models
            initial_type = [('input', FloatTensorType([None, None]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            onnx_path = output_path / f"{model_name}.onnx"
            onnx.save_model(onnx_model, str(onnx_path))
            print(f"ONNX model saved to {onnx_path}")
            return True
            
        print(f"Pipeline steps: {list(model.named_steps.keys())}")
        
        # Extract components
        if 'tfidf' not in model.named_steps:
            print("Error: No TF-IDF vectorizer found in pipeline")
            return False
            
        vectorizer = model.named_steps['tfidf']
        classifier = model.named_steps.get('nb')  # Naive Bayes classifier
        
        if classifier is None:
            print("Error: No classifier found in pipeline")
            return False
            
        # Get vocabulary and feature names
        feature_names = vectorizer.get_feature_names_out()
        n_features = len(feature_names)
        
        print(f"Vocabulary size: {n_features}")
        print(f"Classifier type: {type(classifier).__name__}")
        
        # Extract and save vectorizer metadata
        print(f"Extracting vectorizer metadata...")
        vectorizer_metadata = extract_vectorizer_metadata(vectorizer)
        
        metadata_path = output_path / f"{model_name}_vectorizer.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(vectorizer_metadata, f, indent=2)
        print(f"Vectorizer metadata saved to {metadata_path}")
        
        # Convert classifier to ONNX
        print(f"Converting classifier to ONNX...")
        
        # Define input type for the classifier (TF-IDF output)
        initial_type = [('input', FloatTensorType([None, n_features]))]
        
        # Convert to ONNX
        onnx_model = convert_sklearn(
            classifier,
            initial_types=initial_type,
            target_opset=ONNX_TARGET_OPSET
        )
        
        # Save ONNX model
        onnx_path = output_path / f"{model_name}_classifier.onnx"
        onnx.save_model(onnx_model, str(onnx_path))
        print(f"ONNX classifier saved to {onnx_path}")
        
        # Test the ONNX model
        if not test_onnx_model(onnx_path, n_features):
            print("Warning: ONNX model test failed, but export completed")
        
        # Create complete model metadata
        print(f"Creating model metadata...")
        
        # Get classifier information
        classes = None
        class_log_priors = None
        feature_log_probs = None
        
        if hasattr(classifier, 'classes_'):
            classes = classifier.classes_.tolist()
        
        if hasattr(classifier, 'class_log_prior_'):
            class_log_priors = classifier.class_log_prior_.tolist()
            
        # Export COMPLETE feature log probabilities (not just top features)
        if hasattr(classifier, 'feature_log_prob_'):
            feature_log_probs = classifier.feature_log_prob_.tolist()
            print(f"Exporting complete feature_log_prob_ matrix: {len(feature_log_probs)} classes x {len(feature_log_probs[0])} features")
            
        # Get feature importance for summary (keeping this for backward compatibility)
        feature_importance = None
        if hasattr(classifier, 'feature_log_prob_'):
            # For Naive Bayes, calculate feature importance
            high_risk_log_probs = classifier.feature_log_prob_[1]
            low_risk_log_probs = classifier.feature_log_prob_[0]
            importance = high_risk_log_probs - low_risk_log_probs
            
            # Get top features
            top_indices = importance.argsort()[-20:][::-1]
            feature_importance = {
                'top_high_risk_features': [
                    {
                        'feature': feature_names[idx],
                        'importance': float(importance[idx])
                    }
                    for idx in top_indices
                ]
            }
        
        model_metadata = {
            'model_name': model_name,
            'model_type': 'text_classification_pipeline',
            'classifier_type': type(classifier).__name__,
            'classes': classes,
            'n_features': n_features,
            'pipeline_steps': list(model.named_steps.keys()),
            'onnx_files': {
                'classifier': f"{model_name}_classifier.onnx",
                'vectorizer_metadata': f"{model_name}_vectorizer.json"
            },
            'input_shape': [None, n_features],
            'preprocessing_required': True,
            'description': 'Text classification model with TF-IDF vectorization',
            'feature_importance': feature_importance,
            # Include complete classifier parameters for Rust implementation
            'classifier_parameters': {
                'class_log_priors': class_log_priors,
                'feature_log_probs': feature_log_probs,
                'classes': classes,
                'n_features': n_features
            }
        }
        
        metadata_full_path = output_path / f"{model_name}_metadata.json"
        with open(metadata_full_path, 'w', encoding='utf-8') as f:
            json.dump(model_metadata, f, indent=2)
        print(f"Complete model metadata saved to {metadata_full_path}")
        
        # Create a summary file
        summary = {
            'export_successful': True,
            'files_created': [
                str(onnx_path.name),
                str(metadata_path.name),
                str(metadata_full_path.name)
            ],
            'model_info': {
                'n_features': n_features,
                'classifier_type': type(classifier).__name__,
                'classes': classes
            }
        }
        
        summary_path = output_path / f"{model_name}_export_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"ONNX export completed successfully!")
        print(f"Output directory: {output_path}")
        print(f"Files created:")
        for file in summary['files_created']:
            print(f"   - {file}")
            
        return True
    
    except Exception as e:
        print(f"Error during ONNX conversion: {e}")
        traceback.print_exc()
        return False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Export scikit-learn model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model', 
        required=True, 
        help='Path to trained model (.pkl file)'
    )
    parser.add_argument(
        '--output', 
        required=True, 
        help='Output directory for ONNX model and metadata'
    )
    parser.add_argument(
        '--name', 
        default=DEFAULT_MODEL_NAME, 
        help=f'Name for output files (default: {DEFAULT_MODEL_NAME})'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1
    
    output_dir = Path(args.output)
    
    # Export model
    success = export_model_to_onnx(
        str(model_path), 
        str(output_dir), 
        args.name
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())