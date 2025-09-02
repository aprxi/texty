#!/usr/bin/env python3
"""Evaluate the classifier on all test examples for a specific use-case."""

# Standard library imports
import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

# Third-party imports
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Local imports
from classify import classify_text

# Ensure NLTK data is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def extract_text_content(file_content):
    """Extract actual text content from evaluation file format.
    
    Args:
        file_content: Full file content string
    
    Returns:
        tuple: (text_content, has_proper_format)
        - text_content: Text content between <START_TEXT> and <END_TEXT> tags or full content
        - has_proper_format: True if file has proper annotation format
    """
    try:
        start_tag = '<START_TEXT>'
        end_tag = '<END_TEXT>'
        start_idx = file_content.find(start_tag)
        end_idx = file_content.find(end_tag)
        
        if start_idx == -1 or end_idx == -1:
            # No proper tags found - return entire content and flag as improperly formatted
            return file_content.strip(), False
        
        # Extract content between tags
        text_content = file_content[start_idx + len(start_tag):end_idx].strip()
        return text_content, True
        
    except Exception:
        # Fallback: return original content if parsing fails
        return file_content.strip(), False


def extract_evaluation_content(file_content):
    """Extract evaluation section from evaluation file format.
    
    Args:
        file_content: Full file content string
    
    Returns:
        str: Evaluation content between <START_EVALUATION> and <END_EVALUATION> tags, or None if not found
    """
    try:
        start_tag = '<START_EVALUATION>'
        end_tag = '<END_EVALUATION>'
        start_idx = file_content.find(start_tag)
        end_idx = file_content.find(end_tag)
        
        if start_idx == -1 or end_idx == -1:
            return None
        
        # Extract evaluation content between tags
        evaluation_content = file_content[start_idx + len(start_tag):end_idx].strip()
        return evaluation_content
        
    except Exception:
        return None


def evaluate_dataset(model, eval_dir, confidence_threshold=0.7, use_weighted_scoring=False,
                    threshold_count=1, threshold_percent=0.1, verbose=False):
    """
    Evaluate the model on all evaluation files in a directory (supports 3-class system).
    
    Args:
        model: Trained pipeline (TF-IDF + classifier) for 3-class classification
        eval_dir: Path to evaluation directory containing high_risk/, low_risk/, and neutral/ subdirs
        confidence_threshold: Minimum confidence level to count as high-risk
        use_weighted_scoring: Use confidence-weighted scoring
        threshold_count: Number of high-risk sentences to trigger classification
        threshold_percent: Percentage of high-risk sentences to trigger classification
        verbose: Show detailed output for each file
    
    Returns:
        dict: Evaluation results including metrics and misclassified files
    """
    eval_path = Path(eval_dir)
    results = {
        'total_files': 0,
        'correct': 0,
        'incorrect': 0,
        'true_labels': [],
        'predicted_labels': [],
        'misclassified': [],
        'processing_times': [],
        'confidence_stats': defaultdict(list),
        'format_issues': [],
        'properly_formatted_files': 0,
        'improperly_formatted_files': 0
    }
    
    # Process high-risk files
    high_risk_dir = eval_path / 'high_risk'
    if high_risk_dir.exists():
        for file_path in sorted(high_risk_dir.glob('*.txt')):
            if verbose:
                print(f"Processing {file_path.name}...", end=' ')
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Extract text content from evaluation file format
                text, has_proper_format = extract_text_content(file_content)
                evaluation_content = extract_evaluation_content(file_content)
                
                # Track formatting issues
                if has_proper_format:
                    results['properly_formatted_files'] += 1
                else:
                    results['improperly_formatted_files'] += 1
                    results['format_issues'].append({
                        'file': str(file_path),
                        'category': 'high-risk',
                        'issue': 'Missing START_TEXT/END_TEXT tags'
                    })
                
                start_time = time.time()
                classification, high_risk_count, total_sentences, sentence_details = classify_text(
                    model, text, threshold_count, threshold_percent, 
                    confidence_threshold, use_weighted_scoring, debug=False
                )
                processing_time = (time.time() - start_time) * 1000
                
                results['total_files'] += 1
                results['true_labels'].append('high-risk')
                results['predicted_labels'].append(classification)
                results['processing_times'].append(processing_time)
                
                # Collect confidence statistics
                for detail in sentence_details:
                    results['confidence_stats']['high_risk_files'].append(detail['high_risk_prob'])
                
                if classification == 'high-risk':
                    results['correct'] += 1
                    if verbose:
                        print(f"✓ Correct (high-risk, {high_risk_count}/{total_sentences} sentences)")
                else:
                    results['incorrect'] += 1
                    results['misclassified'].append({
                        'file': str(file_path),
                        'true_label': 'high-risk',
                        'predicted_label': classification,
                        'high_risk_sentences': high_risk_count,
                        'total_sentences': total_sentences,
                        'confidence_details': sentence_details,
                        'evaluation_content': evaluation_content,
                        'has_proper_format': has_proper_format
                    })
                    if verbose:
                        print(f"✗ INCORRECT (predicted: {classification}, "
                              f"{high_risk_count}/{total_sentences} sentences)")
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Process low-risk files
    low_risk_dir = eval_path / 'low_risk'
    if low_risk_dir.exists():
        for file_path in sorted(low_risk_dir.glob('*.txt')):
            if verbose:
                print(f"Processing {file_path.name}...", end=' ')
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Extract text content from evaluation file format
                text, has_proper_format = extract_text_content(file_content)
                evaluation_content = extract_evaluation_content(file_content)
                
                # Track formatting issues
                if has_proper_format:
                    results['properly_formatted_files'] += 1
                else:
                    results['improperly_formatted_files'] += 1
                    results['format_issues'].append({
                        'file': str(file_path),
                        'category': 'low-risk',
                        'issue': 'Missing START_TEXT/END_TEXT tags'
                    })
                
                start_time = time.time()
                classification, high_risk_count, total_sentences, sentence_details = classify_text(
                    model, text, threshold_count, threshold_percent,
                    confidence_threshold, use_weighted_scoring, debug=False
                )
                processing_time = (time.time() - start_time) * 1000
                
                results['total_files'] += 1
                results['true_labels'].append('low-risk')
                results['predicted_labels'].append(classification)
                results['processing_times'].append(processing_time)
                
                # Collect confidence statistics
                for detail in sentence_details:
                    results['confidence_stats']['low_risk_files'].append(detail['high_risk_prob'])
                
                # Accept both low-risk and neutral as correct for low-risk files
                if classification in ['low-risk', 'neutral']:
                    results['correct'] += 1
                    if verbose:
                        print(f"✓ Correct (predicted: {classification}, {high_risk_count}/{total_sentences} sentences)")
                else:
                    results['incorrect'] += 1
                    results['misclassified'].append({
                        'file': str(file_path),
                        'true_label': 'low-risk',
                        'predicted_label': classification,
                        'high_risk_sentences': high_risk_count,
                        'total_sentences': total_sentences,
                        'confidence_details': sentence_details,
                        'evaluation_content': evaluation_content,
                        'has_proper_format': has_proper_format
                    })
                    if verbose:
                        print(f"✗ INCORRECT (predicted: {classification}, "
                              f"{high_risk_count}/{total_sentences} sentences)")
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Process neutral files
    neutral_dir = eval_path / 'neutral'
    if neutral_dir.exists():
        for file_path in sorted(neutral_dir.glob('*.txt')):
            if verbose:
                print(f"Processing {file_path.name}...", end=' ')
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                # Extract text content from evaluation file format
                text, has_proper_format = extract_text_content(file_content)
                evaluation_content = extract_evaluation_content(file_content)
                
                # Track formatting issues
                if has_proper_format:
                    results['properly_formatted_files'] += 1
                else:
                    results['improperly_formatted_files'] += 1
                    results['format_issues'].append({
                        'file': str(file_path),
                        'category': 'neutral',
                        'issue': 'Missing START_TEXT/END_TEXT tags'
                    })
                
                start_time = time.time()
                classification, high_risk_count, total_sentences, sentence_details = classify_text(
                    model, text, threshold_count, threshold_percent,
                    confidence_threshold, use_weighted_scoring, debug=False
                )
                processing_time = (time.time() - start_time) * 1000
                
                results['total_files'] += 1
                results['true_labels'].append('neutral')
                results['predicted_labels'].append(classification)
                results['processing_times'].append(processing_time)
                
                # Collect confidence statistics
                for detail in sentence_details:
                    results['confidence_stats']['neutral_files'].append(detail['high_risk_prob'])
                
                if classification == 'neutral':
                    results['correct'] += 1
                    if verbose:
                        print(f"✓ Correct (neutral, {high_risk_count}/{total_sentences} sentences)")
                else:
                    results['incorrect'] += 1
                    results['misclassified'].append({
                        'file': str(file_path),
                        'true_label': 'neutral',
                        'predicted_label': classification,
                        'high_risk_sentences': high_risk_count,
                        'total_sentences': total_sentences,
                        'confidence_details': sentence_details,
                        'evaluation_content': evaluation_content,
                        'has_proper_format': has_proper_format
                    })
                    if verbose:
                        print(f"✗ INCORRECT (predicted: {classification}, "
                              f"{high_risk_count}/{total_sentences} sentences)")
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return results


def calculate_metrics(results):
    """Calculate evaluation metrics from results (supports 3-class system)."""
    if not results['true_labels']:
        return None
    
    # Map labels to numeric values: 0=low-risk, 1=neutral, 2=high-risk
    label_to_num = {'low-risk': 0, 'neutral': 1, 'high-risk': 2}
    y_true = [label_to_num.get(label, 1) for label in results['true_labels']]  # Default to neutral if unknown
    y_pred = [label_to_num.get(label, 1) for label in results['predicted_labels']]  # Default to neutral if unknown
    
    # Calculate metrics for 3-class system
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist() if len(set(y_true)) > 1 else None
    }
    
    # Add timing statistics
    if results['processing_times']:
        metrics['avg_processing_time_ms'] = sum(results['processing_times']) / len(results['processing_times'])
        metrics['max_processing_time_ms'] = max(results['processing_times'])
        metrics['min_processing_time_ms'] = min(results['processing_times'])
    
    return metrics


def print_evaluation_report(results, metrics, confidence_threshold, use_weighted_scoring):
    """Print a comprehensive evaluation report."""
    print("\n" + "="*80)
    print("EVALUATION REPORT")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Confidence threshold: {confidence_threshold}")
    print(f"  Weighted scoring: {use_weighted_scoring}")
    
    print(f"\nDataset Summary:")
    print(f"  Total files: {results['total_files']}")
    print(f"  Properly formatted files: {results['properly_formatted_files']}")
    print(f"  Improperly formatted files: {results['improperly_formatted_files']}")
    print(f"  Correct classifications: {results['correct']} ({results['correct']/results['total_files']*100:.1f}%)")
    print(f"  Incorrect classifications: {results['incorrect']} ({results['incorrect']/results['total_files']*100:.1f}%)")
    
    if metrics:
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        
        if metrics.get('confusion_matrix'):
            cm = metrics['confusion_matrix']
            print(f"\nConfusion Matrix (3-Class):")
            print(f"                   Predicted")
            print(f"              Low-risk  Neutral  High-risk")
            if len(cm) >= 1:
                print(f"  Low-risk  :    {cm[0][0]:3d}      "
                      f"{cm[0][1] if len(cm[0]) > 1 else 0:3d}       "
                      f"{cm[0][2] if len(cm[0]) > 2 else 0:3d}")
            if len(cm) >= 2:
                print(f"  Neutral   :    {cm[1][0]:3d}      "
                      f"{cm[1][1] if len(cm[1]) > 1 else 0:3d}       "
                      f"{cm[1][2] if len(cm[1]) > 2 else 0:3d}")
            if len(cm) >= 3:
                print(f"  High-risk :    {cm[2][0]:3d}      "
                      f"{cm[2][1] if len(cm[2]) > 1 else 0:3d}       "
                      f"{cm[2][2] if len(cm[2]) > 2 else 0:3d}")
        
        print(f"\nProcessing Time Statistics:")
        print(f"  Average: {metrics['avg_processing_time_ms']:.1f}ms")
        print(f"  Min: {metrics['min_processing_time_ms']:.1f}ms")
        print(f"  Max: {metrics['max_processing_time_ms']:.1f}ms")
    
    # Confidence statistics
    if results['confidence_stats']:
        print(f"\nConfidence Statistics:")
        if results['confidence_stats']['high_risk_files']:
            hr_confs = results['confidence_stats']['high_risk_files']
            print(f"  High-risk files - Avg high-risk confidence: {sum(hr_confs)/len(hr_confs):.3f}")
        if results['confidence_stats']['low_risk_files']:
            lr_confs = results['confidence_stats']['low_risk_files']
            print(f"  Low-risk files - Avg high-risk confidence: {sum(lr_confs)/len(lr_confs):.3f}")
        if results['confidence_stats'].get('neutral_files'):
            nr_confs = results['confidence_stats']['neutral_files']
            print(f"  Neutral files - Avg high-risk confidence: {sum(nr_confs)/len(nr_confs):.3f}")
    
    if results['misclassified']:
        print(f"\nMisclassified Files ({len(results['misclassified'])} total):")
        for item in results['misclassified']:
            print(f"\n  File: {Path(item['file']).name}")
            print(f"  Expected: {item['true_label']}, Got: {item['predicted_label']}")
            print(f"  High-risk sentences: {item['high_risk_sentences']}/{item['total_sentences']}")
            
            # Show formatting status
            if not item.get('has_proper_format', True):
                print(f"  ⚠️  File format issue: Missing START_TEXT/END_TEXT tags")
            
            # Show human evaluation if available
            if item.get('evaluation_content'):
                print(f"  Human Evaluation:")
                # Indent each line of the evaluation
                for line in item['evaluation_content'].split('\n'):
                    if line.strip():
                        print(f"    {line}")
            elif item.get('has_proper_format', True):
                print(f"  ⚠️  No evaluation content found in properly formatted file")
            
            # Show top 3 highest confidence sentences
            sorted_details = sorted(item['confidence_details'], 
                                  key=lambda x: x['high_risk_prob'], reverse=True)[:3]
            print(f"  Top confidence sentences:")
            for i, detail in enumerate(sorted_details, 1):
                neutral_info = f", N: {detail.get('neutral_prob', 0):.3f}" if 'neutral_prob' in detail else ""
                print(f"    {i}. (HR: {detail['high_risk_prob']:.3f}{neutral_info}) {detail['sentence'][:80]}...")
    
    # Show formatting issues summary
    if results['format_issues']:
        print(f"\n⚠️  FORMAT ISSUES ({len(results['format_issues'])} files need proper annotation):")
        by_category = defaultdict(int)
        for issue in results['format_issues']:
            by_category[issue['category']] += 1
        
        for category, count in by_category.items():
            print(f"  {category}: {count} files missing START_TEXT/END_TEXT tags")
        
        print(f"\n  Note: Files without proper tags may include evaluation content as part")
        print(f"        of the text being classified, affecting model performance.")
        
        if len(results['format_issues']) <= 10:
            print(f"\n  Files with format issues:")
            for issue in results['format_issues']:
                print(f"    {Path(issue['file']).name} ({issue['category']})")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier on evaluation dataset")
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--eval-dir', required=True, help='Path to evaluation directory')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                        help='Minimum confidence level to count as high-risk')
    parser.add_argument('--use-weighted-scoring', action='store_true',
                        help='Use confidence-weighted scoring')
    parser.add_argument('--threshold-count', type=int, default=1,
                        help='Number of high-risk sentences to trigger classification')
    parser.add_argument('--threshold-percent', type=float, default=0.1,
                        help='Percentage of high-risk sentences to trigger classification')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed output for each file')
    parser.add_argument('--output-json', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    
    # Run evaluation
    print(f"Evaluating on {args.eval_dir}...")
    results = evaluate_dataset(
        model, args.eval_dir, 
        confidence_threshold=args.confidence_threshold,
        use_weighted_scoring=args.use_weighted_scoring,
        threshold_count=args.threshold_count,
        threshold_percent=args.threshold_percent,
        verbose=args.verbose
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print report
    print_evaluation_report(results, metrics, args.confidence_threshold, args.use_weighted_scoring)
    
    # Save JSON output if requested
    if args.output_json:
        output_data = {
            'configuration': {
                'confidence_threshold': args.confidence_threshold,
                'use_weighted_scoring': args.use_weighted_scoring,
                'threshold_count': args.threshold_count,
                'threshold_percent': args.threshold_percent
            },
            'results': {
                'total_files': results['total_files'],
                'correct': results['correct'],
                'incorrect': results['incorrect'],
                'misclassified_files': [
                    {
                        'file': Path(m['file']).name,
                        'true_label': m['true_label'],
                        'predicted_label': m['predicted_label'],
                        'high_risk_sentences': m['high_risk_sentences'],
                        'total_sentences': m['total_sentences']
                    }
                    for m in results['misclassified']
                ]
            },
            'metrics': metrics
        }
        
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()