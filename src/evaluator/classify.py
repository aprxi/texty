#!/usr/bin/env python3
"""Classify text using the trained model."""

# Standard library imports
import argparse
import os
import pickle
import time
from pathlib import Path

# Third-party imports
import joblib
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def classify_text(model, text, threshold_count=1, threshold_percent=0.1, confidence_threshold=0.7, 
                  use_weighted_scoring=False, debug=False, context_aware=False, context_model_path=None):
    """
    Classify text using sentence-level classification and aggregation for 3-class system.
    
    Args:
        model: Trained pipeline (TF-IDF + classifier) for 3-class classification
        text: Input text to classify
        threshold_count: Number of high-risk sentences to trigger high-risk classification
        threshold_percent: Percentage of high-risk sentences to trigger high-risk classification
        confidence_threshold: Minimum confidence level to count a sentence as high-risk
        use_weighted_scoring: If True, use confidence-weighted scoring instead of binary counting
        debug: If True, show detailed sentence-by-sentence classification
    
    Returns:
        tuple: (classification, high_risk_count, total_sentences, sentence_details)
        
    Notes:
        - Model expects 3 classes: 0=low-risk, 1=neutral, 2=high-risk
        - Final classification aggregates sentence-level predictions
        - Neutral sentences don't contribute to risk thresholds
    """
    # Split text into sentences
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    
    if total_sentences == 0:
        return "neutral", 0, 0, []
    
    # Classify each sentence
    sentence_predictions = model.predict(sentences)
    sentence_probabilities = model.predict_proba(sentences)
    
    # Prepare detailed sentence information and calculate scores
    sentence_details = []
    high_risk_count = 0
    weighted_score = 0.0
    
    for i, (sentence, prediction, proba) in enumerate(zip(sentences, sentence_predictions, sentence_probabilities)):
        # Handle both binary (2-class) and 3-class systems
        if len(proba) == 2:
            # Binary system: 0=low-risk, 1=high-risk
            low_risk_prob = proba[0]
            neutral_prob = 0.0
            high_risk_prob = proba[1]
            class_names = ['LOW-RISK', 'HIGH-RISK']
        else:
            # 3-class system: 0=low-risk, 1=neutral, 2=high-risk
            low_risk_prob = proba[0] if len(proba) > 0 else 0.0
            neutral_prob = proba[1] if len(proba) > 1 else 0.0
            high_risk_prob = proba[2] if len(proba) > 2 else 0.0
            class_names = ['LOW-RISK', 'NEUTRAL', 'HIGH-RISK']
        
        predicted_class = class_names[prediction] if prediction < len(class_names) else 'UNKNOWN'
        
        # Determine effective prediction based on confidence threshold
        if confidence_threshold > 0:
            # Only count as high-risk if confidence exceeds threshold
            effective_prediction = 'HIGH-RISK' if high_risk_prob >= confidence_threshold else predicted_class
            # For display, show original prediction with a marker if overridden by threshold
            display_prediction = predicted_class
            if predicted_class == 'HIGH-RISK' and effective_prediction != 'HIGH-RISK':
                display_prediction += ' (below threshold)'
        else:
            effective_prediction = predicted_class
            display_prediction = predicted_class
        
        sentence_details.append({
            'sentence': sentence.strip(),
            'prediction': display_prediction,
            'effective_prediction': effective_prediction,
            'confidence': proba[prediction],
            'low_risk_prob': low_risk_prob,
            'neutral_prob': neutral_prob,
            'high_risk_prob': high_risk_prob
        })
        
        # Count high-risk sentences based on effective prediction
        if effective_prediction == 'HIGH-RISK':
            high_risk_count += 1
        
        # Calculate weighted score (always based on high-risk probability, regardless of threshold)
        weighted_score += high_risk_prob
    
    if debug:
        print("\n" + "="*80)
        print("DETAILED SENTENCE-BY-SENTENCE ANALYSIS (3-Class System)")
        if confidence_threshold > 0:
            print(f"Confidence threshold: {confidence_threshold:.2f}")
        print("="*80)
        
        # Get feature names from the model for analysis
        feature_names = None
        if hasattr(model, 'named_steps') and 'tfidf' in model.named_steps:
            try:
                feature_names = model.named_steps['tfidf'].get_feature_names_out()
            except AttributeError:
                pass
        
        for i, detail in enumerate(sentence_details, 1):
            # Determine classification status with clear indicators
            if detail['effective_prediction'] == 'HIGH-RISK':
                status_text = "HIGH-RISK"
                status_color = "\033[91m"  # Red
            elif detail['high_risk_prob'] > 0.3:
                status_text = "MODERATE"
                status_color = "\033[93m"  # Yellow
            elif detail['prediction'] == 'LOW-RISK':
                status_text = "LOW-RISK"
                status_color = "\033[92m"  # Green
            else:
                status_text = "NEUTRAL"
                status_color = "\033[94m"  # Blue
            
            reset_color = "\033[0m"  # Reset color
            
            print(f"\n[{i:2d}] {status_color}{status_text}{reset_color} | High-risk: {detail['high_risk_prob']:.1%} (need ≥{confidence_threshold:.0%})")
            print(f"     Text: \"{detail['sentence']}\"")
            
            # Show detailed feature analysis
            if feature_names is not None and hasattr(model, 'named_steps'):
                try:
                    # Get TF-IDF features for this sentence
                    sentence_features = model.named_steps['tfidf'].transform([detail['sentence']])
                    active_features = sentence_features.toarray()[0]
                    feature_indices = active_features.nonzero()[0]
                    
                    # Get model weights to understand feature importance
                    nb_classifier = model.named_steps['nb']
                    
                    if len(feature_indices) > 0:
                        feature_analysis = []
                        for idx in feature_indices:
                            feature_name = feature_names[idx]
                            tfidf_weight = active_features[idx]
                            
                            # Get log probabilities for this feature
                            if hasattr(nb_classifier, 'feature_log_prob_'):
                                if len(nb_classifier.feature_log_prob_) == 2:
                                    # Binary model: 0=low-risk, 1=high-risk
                                    low_risk_log_prob = nb_classifier.feature_log_prob_[0][idx]
                                    high_risk_log_prob = nb_classifier.feature_log_prob_[1][idx]
                                else:
                                    # 3-class model
                                    low_risk_log_prob = nb_classifier.feature_log_prob_[0][idx]
                                    high_risk_log_prob = nb_classifier.feature_log_prob_[2][idx] if len(nb_classifier.feature_log_prob_) > 2 else 0
                                
                                # Calculate feature contribution (higher = more high-risk)
                                feature_contribution = high_risk_log_prob - low_risk_log_prob
                                
                                feature_analysis.append({
                                    'word': feature_name,
                                    'tfidf': tfidf_weight,
                                    'contribution': feature_contribution,
                                    'hr_log_prob': high_risk_log_prob,
                                    'lr_log_prob': low_risk_log_prob
                                })
                        
                        # Sort by feature contribution (most high-risk first)
                        feature_analysis.sort(key=lambda x: x['contribution'], reverse=True)
                        
                        # Show top contributing features
                        print(f"     Feature Analysis (TF-IDF * Model Weights):")
                        for j, feat in enumerate(feature_analysis[:8], 1):
                            contribution_indicator = "+" if feat['contribution'] > 0 else "-"
                            print(f"       {j}. '{feat['word']}' | TF-IDF: {feat['tfidf']:.3f} | Contribution: {contribution_indicator}{abs(feat['contribution']):.3f}")
                        
                        # Show summary of risk vs safe words
                        risk_words = [f['word'] for f in feature_analysis if f['contribution'] > 0]
                        safe_words = [f['word'] for f in feature_analysis if f['contribution'] <= 0]
                        
                        if risk_words:
                            print(f"     Risk indicators: {', '.join(risk_words[:5])}")
                        if safe_words:
                            print(f"     Safe indicators: {', '.join(safe_words[:3])}")
                            
                except Exception as e:
                    print(f"     Feature analysis error: {e}")
            
            # Simple explanation of why this classification happened
            if detail['effective_prediction'] == 'HIGH-RISK':
                print(f"     Result: This sentence TRIGGERS high-risk classification")
            elif detail['high_risk_prob'] > 0.3:
                print(f"     Result: Potentially risky but below {confidence_threshold:.0%} threshold")
            elif detail['high_risk_prob'] > 0.1:
                print(f"     Result: Some risk signals detected ({detail['high_risk_prob']:.1%}) but low confidence")
            else:
                print(f"     Result: No significant risk signals detected")
                
        print("="*80)
        if use_weighted_scoring:
            print(f"Weighted score: {weighted_score:.3f} / {total_sentences} = {weighted_score/total_sentences:.3f}")
        print("\n")
    
    # Calculate percentages
    high_risk_percent = high_risk_count / total_sentences if total_sentences > 0 else 0
    weighted_percent = weighted_score / total_sentences if total_sentences > 0 else 0
    
    # Apply aggregation rules for 3-class system
    # Decision logic: high-risk if thresholds exceeded, otherwise neutral (conservative approach)
    if use_weighted_scoring:
        # Use weighted scoring: average confidence of high-risk across all sentences
        if weighted_score > threshold_count or weighted_percent > threshold_percent:
            classification = "high-risk"
        else:
            classification = "neutral"  # Conservative: treat as neutral unless clearly high-risk
    else:
        # Use binary counting with confidence threshold
        if high_risk_count > threshold_count or high_risk_percent > threshold_percent:
            classification = "high-risk"
        else:
            classification = "neutral"  # Conservative: treat as neutral unless clearly high-risk
    
    return classification, high_risk_count, total_sentences, sentence_details


def main():
    parser = argparse.ArgumentParser(description="Classify text using trained model")
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--input', required=True, help='Path to input text file')
    parser.add_argument('--threshold-count', type=int, default=1, 
                        help='Number of high-risk sentences to trigger high-risk classification')
    parser.add_argument('--threshold-percent', type=float, default=0.1,
                        help='Percentage of high-risk sentences to trigger high-risk classification')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                        help='Minimum confidence level to count a sentence as high-risk (0-1, 0 to disable)')
    parser.add_argument('--use-weighted-scoring', action='store_true',
                        help='Use confidence-weighted scoring instead of binary counting')
    parser.add_argument('--debug', action='store_true',
                        help='Show detailed sentence-by-sentence classification')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    
    # Load text
    print(f"Reading text from {args.input}...")
    
    # Try different path resolution strategies
    possible_paths = []
    
    if args.input.startswith('/data/') or args.input.startswith('/app/'):
        # Already a container path
        possible_paths.append(args.input)
    elif args.input.startswith('./'):
        # Relative path from current directory
        possible_paths.append(args.input.replace('./', '/app/'))
    elif args.input.startswith('/'):
        # Absolute path - could be host or container path
        possible_paths.append(args.input)  # Try as-is first
        possible_paths.append(f"/app{args.input}")  # Try prefixing with /app
    else:
        # Relative path
        possible_paths.append(f"/app/{args.input}")
    
    # Try each path until one works
    file_path = None
    content = None
    for path in possible_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                file_path = path
                break
        except FileNotFoundError:
            continue
    
    if file_path is None:
        print(f"❌ Error: Could not find file at any of these paths:")
        for path in possible_paths:
            print(f"  - {path}")
        exit(1)
    
    # Extract text between markers
    text = ""
    if '<START_TEXT>' in content and '<END_TEXT>' in content:
        start_idx = content.find('<START_TEXT>') + len('<START_TEXT>')
        end_idx = content.find('<END_TEXT>')
        text = content[start_idx:end_idx].strip()
    else:
        # Fallback: use entire content if no markers
        text = content.strip()
    
    # Extract evaluation data if present
    eval_data = {'classification': None, 'confidence': None, 'reasoning': None}
    if '<START_EVALUATION>' in content and '<END_EVALUATION>' in content:
        eval_start = content.find('<START_EVALUATION>') + len('<START_EVALUATION>')
        eval_end = content.find('<END_EVALUATION>')
        evaluation = content[eval_start:eval_end].strip()
        
        # Parse classification
        if '<CLASSIFICATION>' in evaluation and '</CLASSIFICATION>' in evaluation:
            class_start = evaluation.find('<CLASSIFICATION>') + len('<CLASSIFICATION>')
            class_end = evaluation.find('</CLASSIFICATION>')
            eval_data['classification'] = evaluation[class_start:class_end].strip()
        
        # Parse confidence
        if '<CONFIDENCE>' in evaluation and '</CONFIDENCE>' in evaluation:
            conf_start = evaluation.find('<CONFIDENCE>') + len('<CONFIDENCE>')
            conf_end = evaluation.find('</CONFIDENCE>')
            try:
                eval_data['confidence'] = float(evaluation[conf_start:conf_end].strip())
            except ValueError:
                pass
        
        # Parse reasoning
        if '<REASONING>' in evaluation and '</REASONING>' in evaluation:
            reason_start = evaluation.find('<REASONING>') + len('<REASONING>')
            reason_end = evaluation.find('</REASONING>')
            eval_data['reasoning'] = evaluation[reason_start:reason_end].strip()
    
    # Classify
    start_time = time.time()
    classification, high_risk_count, total_sentences, sentence_details = classify_text(
        model, text, args.threshold_count, args.threshold_percent, 
        args.confidence_threshold, args.use_weighted_scoring, args.debug
    )
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Calculate additional metrics for detailed output
    high_risk_percent = high_risk_count / total_sentences if total_sentences > 0 else 0
    weighted_score = sum(s['high_risk_prob'] for s in sentence_details) if sentence_details else 0
    weighted_percent = weighted_score / total_sentences if total_sentences > 0 else 0
    
    # Print results
    print("\n" + "="*60)
    print(f"File: {args.input}")
    print(f"Text length: {len(text)} characters, {len(text.split())} words")
    print(f"Final Classification: {classification.upper()}")
    print(f"High-risk sentences: {high_risk_count}/{total_sentences}")
    if args.use_weighted_scoring:
        print(f"Weighted score: {weighted_score:.3f} / {total_sentences} = {weighted_percent:.3f}")
    print(f"Inference time: {inference_time:.2f}ms")
    print(f"Thresholds: count>{args.threshold_count}, percent>{args.threshold_percent}, confidence>{args.confidence_threshold}")
    
    # Show classification reasoning
    print(f"\nClassification Logic (Two-Stage Process):")
    print(f"   Stage 1 - SENTENCE LEVEL: Each sentence needs High-risk probability >= {args.confidence_threshold}")
    print(f"      Sentences meeting threshold: {high_risk_count}/{total_sentences}")
    print(f"   Stage 2 - TEXT LEVEL: Final classification based on aggregation:")
    
    if args.use_weighted_scoring:
        print(f"      WEIGHTED MODE: Sum of all high-risk probabilities")
        print(f"        Total weighted score: {weighted_score:.3f}")
        print(f"        Average per sentence: {weighted_percent:.3f}")
        print(f"        Thresholds: weighted_score > {args.threshold_count} OR avg_score > {args.threshold_percent}")
        if weighted_score > args.threshold_count or weighted_percent > args.threshold_percent:
            print(f"        PASS - Threshold exceeded → HIGH-RISK")
        else:
            print(f"        FAIL - Threshold not met → NEUTRAL")
    else:
        print(f"      COUNT MODE: Count qualifying sentences")
        print(f"        Qualifying sentences: {high_risk_count}")
        print(f"        Percentage of text: {high_risk_percent:.3f} ({high_risk_percent*100:.1f}%)")
        print(f"        Thresholds: count > {args.threshold_count} OR percentage > {args.threshold_percent}")
        if high_risk_count > args.threshold_count or high_risk_percent > args.threshold_percent:
            print(f"        PASS - Threshold exceeded → HIGH-RISK")
        else:
            print(f"        FAIL - Threshold not met → NEUTRAL")
    
    # Show any high-risk sentences found
    if sentence_details:
        high_risk_sentences = [s for s in sentence_details if s['effective_prediction'] == 'HIGH-RISK']
        if high_risk_sentences:
            print(f"\nHigh-Risk Sentences Found ({len(high_risk_sentences)}):")
            for i, detail in enumerate(high_risk_sentences, 1):
                print(f"   [{i}] Confidence: {detail['high_risk_prob']:.3f}")
                print(f"       Text: {detail['sentence']}")
        else:
            print(f"\nNo high-risk sentences detected")
            
    # Show expected vs actual classification from evaluation data
    if eval_data['classification']:
        expected = eval_data['classification'].upper()
        print(f"\nExpected: {expected}")
        if eval_data['confidence']:
            print(f"Expected confidence: {eval_data['confidence']:.2f}")
        print(f"Predicted: {classification.upper()}")
        
        if classification.lower() != eval_data['classification'].lower():
            print(f"MISMATCH: Model disagrees with evaluation")
        
        if args.debug and eval_data['reasoning']:
            print(f"\nEvaluation reasoning:")
            print(f"{eval_data['reasoning']}")
    elif 'high_risk' in args.input or 'low_risk' in args.input:
        # Fallback to path-based detection if no evaluation data
        expected = "HIGH-RISK" if 'high_risk' in args.input else "LOW-RISK" 
        print(f"\nExpected: {expected} (from file path)")
        print(f"Predicted: {classification.upper()}")
        if classification.lower() != expected.lower():
            print(f"MISMATCH: Model disagrees with file location")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()