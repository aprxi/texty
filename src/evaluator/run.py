#!/usr/bin/env python3
"""Evaluation using the base classifier"""

# Standard library imports
import argparse
import json
import pickle
import time
from collections import defaultdict, Counter
from pathlib import Path

# Third-party imports
from tqdm import tqdm


def load_base_model(model_path):
    """Load the base trained model from disk."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def parse_evaluation_file(file_path):
    """Parse a file with XML-style evaluation format."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract text between markers
    text = ""
    if '<START_TEXT>' in content and '<END_TEXT>' in content:
        start_idx = content.find('<START_TEXT>') + len('<START_TEXT>')
        end_idx = content.find('<END_TEXT>')
        text = content[start_idx:end_idx].strip()
    else:
        # Fallback: use entire content if no markers
        text = content.strip()
    
    # Extract evaluation data
    eval_data = {
        'classification': None,
        'confidence': None,
        'reasoning': None
    }
    
    if '<START_EVALUATION>' in content and '<END_EVALUATION>' in content:
        eval_start = content.find('<START_EVALUATION>') + len('<START_EVALUATION>')
        eval_end = content.find('<END_EVALUATION>')
        evaluation = content[eval_start:eval_end].strip()
        
        # Parse XML-style tags
        if '<CLASSIFICATION>' in evaluation and '</CLASSIFICATION>' in evaluation:
            class_start = evaluation.find('<CLASSIFICATION>') + len('<CLASSIFICATION>')
            class_end = evaluation.find('</CLASSIFICATION>')
            eval_data['classification'] = evaluation[class_start:class_end].strip()
        
        if '<CONFIDENCE>' in evaluation and '</CONFIDENCE>' in evaluation:
            conf_start = evaluation.find('<CONFIDENCE>') + len('<CONFIDENCE>')
            conf_end = evaluation.find('</CONFIDENCE>')
            try:
                eval_data['confidence'] = float(evaluation[conf_start:conf_end].strip())
            except:
                pass
        
        if '<REASONING>' in evaluation and '</REASONING>' in evaluation:
            reason_start = evaluation.find('<REASONING>') + len('<REASONING>')
            reason_end = evaluation.find('</REASONING>')
            eval_data['reasoning'] = evaluation[reason_start:reason_end].strip()
    
    return text, eval_data


def classify_single_file(file_path, model, debug=False):
    """Classify a single text file and print results."""
    start_time = time.time()
    
    # Read and parse the file
    result = parse_evaluation_file(file_path)
    text = result['text']
    
    if not text.strip():
        print("❌ Error: File contains no text")
        return
    
    print(f"📄 Text length: {len(text)} characters, {len(text.split())} words")
    
    # Classify the text
    classification, confidence, detailed_results = model.classify_text(text)
    
    inference_time = time.time() - start_time
    
    # Print results
    print(f"🎯 Classification: {classification.upper()}")
    print(f"📊 Confidence: {confidence:.3f}")
    print(f"⚡ Inference time: {inference_time*1000:.2f}ms")
    
    if debug and detailed_results:
        print(f"\n🔍 Detailed Analysis:")
        for i, detail in enumerate(detailed_results):
            if detail['classification'] == 'high_risk':
                print(f"  [{i+1}] HIGH-RISK (conf: {detail['confidence']:.3f}): {detail['text'][:100]}...")


def evaluate_dataset(eval_dir, model, verbose=False, debug=False):
    """Evaluate all texts in a directory using the base model."""
    all_results = []
    
    # Process all text files
    eval_path = Path(eval_dir)
    text_files = list(eval_path.rglob('*.txt'))
    
    print(f"Found {len(text_files)} files to evaluate")
    
    # Track timing
    start_time = time.time()
    files_processed = 0
    
    # Add progress bar
    for file_path in tqdm(text_files, desc="Evaluating files", unit="file"):
        # Parse the file
        text, eval_data = parse_evaluation_file(file_path)
        
        # Determine expected label from evaluation data or path
        if eval_data['classification']:
            expected_label = 'high_risk' if eval_data['classification'] == 'high-risk' else 'low_risk'
        elif 'high_risk' in str(file_path):
            expected_label = 'high_risk'
        elif 'low_risk' in str(file_path):
            expected_label = 'low_risk'
        else:
            expected_label = 'unknown'
        
        # Evaluate with base model
        probabilities = model.predict_proba([text])[0]
        high_risk_probability = probabilities[1]
        
        # Use more sensitive threshold due to class imbalance in training data
        prediction = 1 if high_risk_probability >= 0.5 else 0
        predicted_label = 'high_risk' if prediction == 1 else 'low_risk'
        confidence = max(probabilities)
        high_risk_probability = probabilities[1]
        
        result_entry = {
            'file': str(file_path.relative_to(eval_path)),
            'expected': expected_label,
            'predicted': predicted_label,
            'correct': expected_label == predicted_label,
            'confidence': confidence,
            'high_risk_probability': high_risk_probability,
            'eval_data': eval_data,
            'text_length': len(text),
            'text': text  # Add full text for debug analysis
        }
        
        all_results.append(result_entry)
        
        # Update progress tracking
        files_processed += 1
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"File: {result_entry['file']}")
            print(f"Expected: {expected_label}, Predicted: {predicted_label}")
            print(f"Confidence: {confidence:.3f}, HR Probability: {high_risk_probability:.3f}")
            if eval_data['confidence'] is not None:
                print(f"Evaluation confidence: {eval_data['confidence']}")
            print(f"Text: {text[:200]}...")
    
    # Final timing report
    total_time = time.time() - start_time
    print(f"\nEvaluation complete: {len(text_files)} files in {total_time:.1f}s "
          f"({len(text_files)/total_time:.1f} files/sec)")
    
    return all_results


def print_evaluation_summary(results):
    """Print summary statistics for evaluation results."""
    # Overall accuracy
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    # Count files with evaluations
    files_with_eval = sum(1 for r in results if r['eval_data']['classification'] is not None)
    
    # Average confidence for files with evaluations
    conf_scores = [r['eval_data']['confidence'] for r in results if r['eval_data']['confidence'] is not None]
    avg_eval_confidence = sum(conf_scores) / len(conf_scores) if conf_scores else 0
    
    print(f"\n{'='*80}")
    print("PER-TEXT BREAKDOWN (first 20 files)")
    print(f"{'='*80}")
    print(f"{'File':<25} {'Exp':<3} {'Pred':<4} {'Conf':<5} {'HR_Prob':<6} {'Correct':<7}")
    print(f"{'-'*25} {'-'*3} {'-'*4} {'-'*5} {'-'*6} {'-'*7}")
    
    # Show first 10 high-risk and first 10 low-risk files
    high_risk_results = [r for r in results if r['expected'] == 'high_risk'][:10]
    low_risk_results = [r for r in results if r['expected'] == 'low_risk'][:10]
    combined_results = high_risk_results + low_risk_results
    
    for result in combined_results:
        file_short = result['file'][:24] + ('…' if len(result['file']) > 24 else '')
        expected_short = result['expected'][0].upper()  # H or L
        predicted_short = result['predicted'][0].upper()  # H or L
        correct_mark = "✓" if result['correct'] else "✗"
        
        print(f"{file_short:<25} {expected_short:<3} {predicted_short:<4} "
              f"{result['confidence']:.3f} {result['high_risk_probability']:.3f}  "
              f"{correct_mark:<7}")
    
    print(f"\nShowing 10 high-risk + 10 low-risk files (out of {len(results)} total)")
    
    print(f"\n{'='*80}")
    print("OVERALL EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total files evaluated: {total}")
    print(f"Files with XML evaluations: {files_with_eval}")
    print(f"Average evaluation confidence: {avg_eval_confidence:.3f}")
    print(f"\nCorrect predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Confusion matrix
    print(f"\n{'='*80}")
    print("CONFUSION MATRIX")
    print(f"{'='*80}")
    
    cm = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    for result in results:
        if result['expected'] == 'high_risk' and result['predicted'] == 'high_risk':
            cm['tp'] += 1
        elif result['expected'] == 'low_risk' and result['predicted'] == 'high_risk':
            cm['fp'] += 1
        elif result['expected'] == 'low_risk' and result['predicted'] == 'low_risk':
            cm['tn'] += 1
        elif result['expected'] == 'high_risk' and result['predicted'] == 'low_risk':
            cm['fn'] += 1
    
    print("                 Predicted")
    print("                 Low-risk  High-risk")
    print(f"Actual Low-risk  {cm['tn']:8d}  {cm['fp']:9d}")
    print(f"       High-risk {cm['fn']:8d}  {cm['tp']:9d}")
    
    # Calculate metrics
    precision = cm['tp'] / (cm['tp'] + cm['fp']) if (cm['tp'] + cm['fp']) > 0 else 0
    recall = cm['tp'] / (cm['tp'] + cm['fn']) if (cm['tp'] + cm['fn']) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")


def print_debug_analysis(results, model):
    """Print detailed debug analysis showing misclassifications and feature patterns"""
    print(f"\n{'='*100}")
    print("DEBUG: DETAILED FEATURE AND FAILURE ANALYSIS")
    print(f"{'='*100}")
    
    # Get model feature info
    tfidf = model.named_steps['tfidf']
    feature_names = tfidf.get_feature_names_out()
    nb_classifier = model.named_steps['nb']
    
    # Calculate global feature importance
    if hasattr(nb_classifier, 'feature_log_prob_'):
        if len(nb_classifier.feature_log_prob_) == 2:
            # Binary model: 0=low-risk, 1=high-risk
            low_risk_log_probs = nb_classifier.feature_log_prob_[0]
            high_risk_log_probs = nb_classifier.feature_log_prob_[1]
        else:
            # 3-class model
            low_risk_log_probs = nb_classifier.feature_log_prob_[0]
            high_risk_log_probs = (nb_classifier.feature_log_prob_[2] 
                                   if len(nb_classifier.feature_log_prob_) > 2 
                                   else nb_classifier.feature_log_prob_[1])
        
        # Calculate feature contributions (higher = more high-risk)
        feature_contributions = high_risk_log_probs - low_risk_log_probs
        
        # Show top 30 positive and negative features
        print(f"\nTOP 30 HIGH-RISK FEATURES (strongest risk indicators):")
        print("-" * 80)
        top_positive_indices = feature_contributions.argsort()[-30:][::-1]
        for i, idx in enumerate(top_positive_indices, 1):
            feature = feature_names[idx]
            contribution = feature_contributions[idx]
            hr_prob = high_risk_log_probs[idx]
            lr_prob = low_risk_log_probs[idx]
            print(f"{i:2d}. {feature:<25} | Contribution: +{contribution:.3f} | HR: {hr_prob:.3f} | LR: {lr_prob:.3f}")
        
        print(f"\nTOP 30 LOW-RISK FEATURES (strongest safety indicators):")
        print("-" * 80)
        top_negative_indices = feature_contributions.argsort()[:30]
        for i, idx in enumerate(top_negative_indices, 1):
            feature = feature_names[idx]
            contribution = feature_contributions[idx]
            hr_prob = high_risk_log_probs[idx]
            lr_prob = low_risk_log_probs[idx]
            print(f"{i:2d}. {feature:<25} | Contribution: {contribution:.3f} | HR: {hr_prob:.3f} | LR: {lr_prob:.3f}")
    
    # Separate into correct and incorrect
    false_positives = []  # Low-risk wrongly classified as high-risk
    false_negatives = []  # High-risk wrongly classified as low-risk
    
    for result in results:
        if not result['correct']:
            if result['expected'] == 'low_risk' and result['predicted'] == 'high_risk':
                false_positives.append(result)
            elif result['expected'] == 'high_risk' and result['predicted'] == 'low_risk':
                false_negatives.append(result)
    
    print(f"\n{'='*100}")
    print(f"MISCLASSIFICATION ANALYSIS:")
    print(f"{'='*100}")
    print(f"FALSE NEGATIVES (High-risk missed): {len(false_negatives)}")
    print(f"FALSE POSITIVES (Low-risk over-flagged): {len(false_positives)}")
    
    # Analyze problematic features in misclassifications
    problematic_features = {}  # feature -> {'fn_count': X, 'fp_count': Y, 'examples': [...]}
    
    # Analyze false negatives
    if false_negatives:
        print(f"\nFALSE NEGATIVES ANALYSIS ({len(false_negatives)} cases):")
        print("-" * 80)
        for i, result in enumerate(false_negatives[:10], 1):  # Show first 10
            # Extract filename and category from path
            filepath = Path(result['file'])
            category = filepath.parent.name if filepath.parent.name != 'high_risk' else filepath.parent.parent.name
            filename = filepath.name  # with extension
            
            print(f"\n{i}. File: {filename}")
            print(f"   Category: {category}")
            print(f"   Confidence: {result['high_risk_probability']:.1%}")
            
            # Show text preview
            text_preview = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
            text_preview = text_preview.replace('\n', ' ')  # Single line
            print(f"   Text: {text_preview}")
            
            # Analyze features in this text
            text_features = tfidf.transform([result['text']])
            active_features = text_features.toarray()[0]
            feature_indices = active_features.nonzero()[0]
            
            if len(feature_indices) > 0:
                # Find which features should have triggered high-risk but didn't
                feature_analysis = []
                for idx in feature_indices:
                    feature_name = feature_names[idx]
                    tfidf_weight = active_features[idx]
                    contribution = feature_contributions[idx]
                    
                    feature_analysis.append({
                        'word': feature_name,
                        'tfidf': tfidf_weight,
                        'contribution': contribution
                    })
                    
                    # Track problematic features
                    if feature_name not in problematic_features:
                        problematic_features[feature_name] = {'fn_count': 0, 'fp_count': 0, 'examples': []}
                    problematic_features[feature_name]['fn_count'] += 1
                    problematic_features[feature_name]['examples'].append(f"FN: {result['file']}")
                
                # Sort by contribution and show top features
                feature_analysis.sort(key=lambda x: x['contribution'], reverse=True)
                
                # Show positive and negative contributors separately
                positive_features = [f for f in feature_analysis if f['contribution'] > 0][:5]
                negative_features = [f for f in feature_analysis if f['contribution'] < 0][:3]
                
                if positive_features:
                    print(f"   High-risk indicators:")
                    for f in positive_features:
                        print(f"      + {f['word']:15} (score: {f['contribution']:+6.2f})")
                
                if negative_features:
                    print(f"   Low-risk indicators:")
                    for f in negative_features:
                        print(f"      - {f['word']:15} (score: {f['contribution']:+6.2f})")
    
    # Analyze false positives
    if false_positives:
        print(f"\nFALSE POSITIVES ANALYSIS ({len(false_positives)} cases):")
        print("-" * 80)
        for i, result in enumerate(false_positives[:10], 1):  # Show first 10
            # Extract filename and category from path
            filepath = Path(result['file'])
            category = filepath.parent.name if filepath.parent.name != 'low_risk' else filepath.parent.parent.name
            filename = filepath.name  # with extension
            
            print(f"\n{i}. File: {filename}")
            print(f"   Category: {category}")
            print(f"   Confidence: {result['high_risk_probability']:.1%}")
            
            # Show text preview
            text_preview = result['text'][:150] + "..." if len(result['text']) > 150 else result['text']
            text_preview = text_preview.replace('\n', ' ')  # Single line
            print(f"   Text: {text_preview}")
            
            # Analyze features that triggered false positive
            text_features = tfidf.transform([result['text']])
            active_features = text_features.toarray()[0]
            feature_indices = active_features.nonzero()[0]
            
            if len(feature_indices) > 0:
                feature_analysis = []
                for idx in feature_indices:
                    feature_name = feature_names[idx]
                    tfidf_weight = active_features[idx]
                    contribution = feature_contributions[idx]
                    
                    feature_analysis.append({
                        'word': feature_name,
                        'tfidf': tfidf_weight,
                        'contribution': contribution
                    })
                    
                    # Track problematic features
                    if feature_name not in problematic_features:
                        problematic_features[feature_name] = {'fn_count': 0, 'fp_count': 0, 'examples': []}
                    if contribution > 0:  # Only count positive contributors to false positives
                        problematic_features[feature_name]['fp_count'] += 1
                        problematic_features[feature_name]['examples'].append(f"FP: {result['file']}")
                
                # Sort by contribution and show features that caused false positive
                feature_analysis.sort(key=lambda x: x['contribution'], reverse=True)
                
                # Show what triggered the false positive
                risk_features = [f for f in feature_analysis if f['contribution'] > 0][:5]
                safe_features = [f for f in feature_analysis if f['contribution'] < 0][:3]
                
                if risk_features:
                    print(f"   Triggered by (false indicators):")
                    for f in risk_features:
                        print(f"      ! {f['word']:15} (score: {f['contribution']:+6.2f})")
                
                if safe_features:
                    print(f"   Correctly identified as low-risk:")
                    for f in safe_features:
                        print(f"      ✓ {f['word']:15} (score: {f['contribution']:+6.2f})")
    
    # Show most problematic features across all misclassifications
    if problematic_features:
        print(f"\nMOST PROBLEMATIC FEATURES:")
        print("-" * 80)
        # Sort by total problem count
        sorted_problems = sorted(problematic_features.items(), 
                               key=lambda x: x[1]['fn_count'] + x[1]['fp_count'], 
                               reverse=True)
        
        print(f"{'Feature':<20} {'FN':<5} {'FP':<5} {'Total':<8} Impact")
        print("-" * 80)
        
        for feature, stats in sorted_problems[:15]:  # Show top 15 problematic features
            total_problems = stats['fn_count'] + stats['fp_count']
            if total_problems > 1:  # Only show features that appear in multiple misclassifications
                # Determine impact type
                if stats['fn_count'] > 0 and stats['fp_count'] > 0:
                    impact = "Mixed (causes both FN and FP)"
                elif stats['fn_count'] > 0:
                    impact = "Missing high-risk signal"
                else:
                    impact = "False high-risk trigger"
                
                print(f"{feature:<20} {stats['fn_count']:<5} {stats['fp_count']:<5} {total_problems:<8} {impact}")
    
    # Show accuracy breakdown
    hr_correct = sum(1 for r in results if r['expected'] == 'high_risk' and r['correct'])
    hr_total = sum(1 for r in results if r['expected'] == 'high_risk')
    lr_correct = sum(1 for r in results if r['expected'] == 'low_risk' and r['correct'])
    lr_total = sum(1 for r in results if r['expected'] == 'low_risk')
    
    hr_accuracy = (hr_correct / hr_total * 100) if hr_total > 0 else 0
    lr_accuracy = (lr_correct / lr_total * 100) if lr_total > 0 else 0
    overall_accuracy = ((hr_correct + lr_correct) / (hr_total + lr_total) * 100) if (hr_total + lr_total) > 0 else 0
    
    print(f"\nACCURACY BREAKDOWN:")
    print("-" * 40)
    print(f"High-risk files: {hr_correct}/{hr_total} correct ({hr_accuracy:.1f}%)")
    print(f"Low-risk files:  {lr_correct}/{lr_total} correct ({lr_accuracy:.1f}%)")  
    print(f"Overall accuracy: {overall_accuracy:.1f}% ({hr_correct + lr_correct}/{hr_total + lr_total})")
    
    # Generate actionable recommendations
    generate_actionable_recommendations(false_negatives, false_positives, problematic_features)


def generate_actionable_recommendations(false_negatives, false_positives, problematic_features):
    """Generate actionable recommendations for improving the training data."""
    
    print(f"\n{'='*100}")
    print(f"ACTIONABLE TRAINING DATA RECOMMENDATIONS:")
    print(f"{'='*100}")
    
    # 1. Analyze category-specific issues
    fn_categories = Counter([Path(r['file']).parent.name for r in false_negatives])
    fp_categories = Counter([Path(r['file']).parent.name for r in false_positives])
    
    print(f"\n1. CATEGORY-SPECIFIC ISSUES:")
    print("-" * 80)
    
    # Find categories with high false negative rates
    if fn_categories:
        print(f"\nHigh-risk categories being missed (False Negatives):")
        for category, count in fn_categories.most_common(5):
            print(f"  • {category}: {count} misses")
            print(f"    ACTION: Add more {category} vocabulary to datasets/u0/train/*/hr/{category}/")
            print(f"    FOCUS: Look for domain-specific terms that distinguish {category} from general use")
    
    # Find categories with high false positive rates  
    if fp_categories:
        print(f"\nLow-risk categories being over-flagged (False Positives):")
        for category, count in fp_categories.most_common(5):
            print(f"  • {category}: {count} over-flags")
            print(f"    ACTION: Add more {category} vocabulary to datasets/u0/train/*/lr/{category}/")
            print(f"    FOCUS: Add terms that clearly indicate non-critical {category} use cases")
    
    # 2. Vocabulary improvements based on problematic features
    print(f"\n2. VOCABULARY ADJUSTMENTS NEEDED:")
    print("-" * 80)
    
    # Features causing false negatives (need high-risk context)
    fn_features = [f for f, stats in problematic_features.items() 
                   if stats['fn_count'] > 0 and stats['fp_count'] == 0][:5]
    if fn_features:
        print(f"\nTerms incorrectly treated as low-risk (add to high-risk vocabulary):")
        for feature in fn_features:
            stats = problematic_features[feature]
            print(f"  • '{feature}' (missed in {stats['fn_count']} high-risk cases)")
            print(f"    ACTION: Add phrases containing '{feature}' to datasets/u0/train/function/hr/")
            print(f"    EXAMPLE: '{feature} voor automatische besluitvorming', '{feature} detectie systeem'")
    
    # Features causing false positives (need low-risk context)
    fp_features = [f for f, stats in problematic_features.items() 
                   if stats['fp_count'] > 0 and stats['fn_count'] == 0][:5]
    if fp_features:
        print(f"\nTerms incorrectly treated as high-risk (add to low-risk vocabulary):")
        for feature in fp_features:
            stats = problematic_features[feature]
            print(f"  • '{feature}' (false trigger in {stats['fp_count']} low-risk cases)")
            print(f"    ACTION: Add phrases containing '{feature}' to datasets/u0/train/function/lr/")
            print(f"    EXAMPLE: '{feature} voor onderzoek', '{feature} in gaming context'")
    
    # Ambiguous features (need context clarification)
    mixed_features = [f for f, stats in problematic_features.items() 
                      if stats['fn_count'] > 0 and stats['fp_count'] > 0][:5]
    if mixed_features:
        print(f"\nAmbiguous terms needing context (add to both with clear context):")
        for feature in mixed_features:
            stats = problematic_features[feature]
            print(f"  • '{feature}' (FN: {stats['fn_count']}, FP: {stats['fp_count']})")
            print(f"    HIGH-RISK: Add '{feature} + kritieke infrastructuur/biometrie/rechtspraak'")
            print(f"    LOW-RISK: Add '{feature} + entertainment/persoonlijk gebruik/onderzoek'")
    
    # 3. Missing vocabulary patterns
    print(f"\n3. VOCABULARY GAPS TO FILL:")
    print("-" * 80)
    
    # Analyze text patterns in false negatives to find missing high-risk signals
    if false_negatives:
        print(f"\nCommon patterns in missed high-risk texts:")
        # Extract key phrases from FN texts
        common_phrases = []
        for fn in false_negatives[:10]:
            text_lower = fn['text'].lower()
            if 'asiel' in text_lower or 'ind' in text_lower:
                common_phrases.append('migration/asylum terms')
            if 'rechtbank' in text_lower or 'rechter' in text_lower:
                common_phrases.append('justice/legal terms')
            if 'politie' in text_lower or 'opspo' in text_lower:
                common_phrases.append('law enforcement terms')
            if 'school' in text_lower or 'student' in text_lower or 'onderwijs' in text_lower:
                common_phrases.append('education terms')
        
        phrase_counts = Counter(common_phrases)
        for phrase, count in phrase_counts.most_common(3):
            print(f"  • Missing {phrase} (found in {count} missed cases)")
            print(f"    ACTION: Review and expand vocabulary in relevant category")
    
    # 4. Training strategy recommendations
    print(f"\n4. TRAINING STRATEGY ADJUSTMENTS:")
    print("-" * 80)
    
    total_fn = len(false_negatives)
    total_fp = len(false_positives)
    
    if total_fp > total_fn * 1.5:
        print(f"\n⚠️  Model is OVER-SENSITIVE (too many false positives: {total_fp} vs {total_fn} false negatives)")
        print(f"   ACTIONS:")
        print(f"   1. Increase low-risk training data, especially in categories: "
              f"{', '.join(list(fp_categories.keys())[:3])}")
        print(f"   2. Consider excluding noisy categories: make train EXCLUDE=generated")
        print(f"   3. Add more context-specific low-risk examples")
    elif total_fn > total_fp * 1.5:
        print(f"\n⚠️  Model is UNDER-SENSITIVE (missing too many high-risk: {total_fn} vs {total_fp} false positives)")
        print(f"   ACTIONS:")
        print(f"   1. Increase high-risk training data, especially in categories: "
              f"{', '.join(list(fn_categories.keys())[:3])}")
        print(f"   2. Review high-risk vocabulary for completeness")
        print(f"   3. Consider lowering confidence threshold (currently 0.7)")
    else:
        print(f"\n✓ Model balance is reasonable (FN: {total_fn}, FP: {total_fp})")
        print(f"   Fine-tune by addressing specific category issues listed above")
    
    # 5. Quick wins
    print(f"\n5. QUICK WINS (immediate actions):")
    print("-" * 80)
    print(f"1. Create new vocabulary files for top problematic categories:")
    for cat in list(fn_categories.keys())[:2]:
        print(f"   echo 'high-risk {cat} terms' > datasets/u0/train/function/hr/{cat}_specific.txt")
    print(f"\n2. Review and expand existing vocabulary:")
    print(f"   ls datasets/u0/train/*/hr/ | head -5  # Check current high-risk vocab")
    print(f"\n3. Generate more synthetic examples for problem categories:")
    print(f"   cd src/generator && make generate CATEGORIES={','.join(list(fn_categories.keys())[:3])}")
    print(f"\n4. Retrain with adjusted parameters:")
    print(f"   make train MODEL=u0 FEATURES=7500  # Increase feature count")
    print(f"   make train MODEL=u0 EXCLUDE=generated  # Exclude noisy synthetic data")


def main():
    parser = argparse.ArgumentParser(description="Evaluate texts using base model")
    parser.add_argument('--model', required=True, help='Path to base model')
    parser.add_argument('--eval-dir', help='Directory containing texts to evaluate')
    parser.add_argument('--input', help='Single input text file to classify')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Show detailed results for each file')
    parser.add_argument('--debug', action='store_true',
                        help='Show detailed failure analysis including misclassified texts')
    parser.add_argument('--output-json', help='Save detailed results to JSON file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.eval_dir and not args.input:
        parser.error("Either --eval-dir or --input must be specified")
    if args.eval_dir and args.input:
        parser.error("Cannot specify both --eval-dir and --input")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_base_model(args.model)
    print("Model loaded successfully")
    
    if args.input:
        # Single file classification
        print(f"\nClassifying single file: {args.input}")
        classify_single_file(args.input, model, debug=args.debug)
    else:
        # Evaluate dataset
        print(f"\nEvaluating texts in: {args.eval_dir}")
        results = evaluate_dataset(args.eval_dir, model, verbose=args.verbose, debug=args.debug)
        
        # Print summary
        print_evaluation_summary(results)
        
        # Print debug information if requested
        if args.debug:
            print_debug_analysis(results, model)
        
        # Save detailed results if requested
        if args.output_json:
            json_results = []
            for result in results:
                json_result = {
                    'file': result['file'],
                    'expected': result['expected'],
                    'predicted': result['predicted'],
                    'correct': result['correct'],
                    'confidence': result['confidence'],
                    'high_risk_probability': result['high_risk_probability'],
                    'text_length': result['text_length'],
                    'evaluation': {
                        'classification': result['eval_data']['classification'],
                        'confidence': result['eval_data']['confidence'],
                        'reasoning': result['eval_data']['reasoning']
                    }
                }
                json_results.append(json_result)
            
            with open(args.output_json, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"\nDetailed results saved to: {args.output_json}")


if __name__ == "__main__":
    main()