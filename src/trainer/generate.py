#!/usr/bin/env python3
"""Generate synthetic training data by combining vocabulary elements.

This script creates synthetic sentences by combining elements from:
- function: what the AI system does
- target: who is affected  
- what: what data is processed

High-risk combinations require at least 2 high-risk elements.
Low-risk combinations use only low-risk elements.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Constants
DEFAULT_SAMPLES_PER_RISK = 10000
DEFAULT_VOCAB_DIR = Path('./vocabulary')
DEFAULT_OUTPUT_DIR = Path('./generated')
DEFAULT_RANDOM_SEED = 42
DEFAULT_MAX_TERMS_PER_CATEGORY = 50
HEADER_END_MARKER = 'END HEADER RULES'
VOCABULARY_CATEGORIES = ['function', 'target', 'what']
RISK_LEVELS = ['hr', 'lr']


class VocabularyLoader:
    """Load vocabulary terms from text files."""
    
    def __init__(self, vocab_dir: Path):
        """Initialize vocabulary loader.
        
        Args:
            vocab_dir: Path to vocabulary directory.
        """
        self.vocab_dir = vocab_dir
        self.vocabulary = self._load_all_vocabulary()
    
    def _load_all_vocabulary(self) -> Dict[str, Dict[str, List[str]]]:
        """Load all vocabulary files organized by category and risk level.
        
        Returns:
            Dictionary mapping category -> risk_level -> list of terms.
        """
        vocab: Dict[str, Dict[str, List[str]]] = {
            category: {risk: [] for risk in RISK_LEVELS}
            for category in VOCABULARY_CATEGORIES
        }
        
        for category in VOCABULARY_CATEGORIES:
            for risk in RISK_LEVELS:
                path = self.vocab_dir / category / risk
                if path.exists():
                    vocab[category][risk] = self._load_category(path)
                    print(f"Loaded {len(vocab[category][risk])} {risk} terms for {category}")
                else:
                    print(f"Warning: Path not found: {path}")
        
        return vocab
    
    def _load_category(self, category_path: Path) -> List[str]:
        """Load all vocabulary terms from a category directory.
        
        Args:
            category_path: Path to category directory containing text files.
            
        Returns:
            List of unique vocabulary terms.
        """
        terms = []
        
        for txt_file in category_path.rglob('*.txt'):
            file_terms = self._parse_vocabulary_file(txt_file)
            terms.extend(file_terms)
        
        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms
    
    def _parse_vocabulary_file(self, file_path: Path) -> List[str]:
        """Parse a single vocabulary file and extract terms.
        
        Args:
            file_path: Path to vocabulary file.
            
        Returns:
            List of terms from the file.
        """
        terms = []
        header_ended = False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Look for end of header marker
                    if HEADER_END_MARKER in line:
                        header_ended = True
                        continue
                    
                    # Skip header section
                    if not header_ended:
                        continue
                    
                    # Skip comment lines in content section
                    if line.startswith('#'):
                        continue
                    
                    # Extract term (clean line that's not a comment)
                    # Some terms might have English in parentheses
                    term = line.split('(')[0].strip()
                    
                    if term:
                        terms.append(term)
        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        return terms
    
    def get_vocabulary_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about loaded vocabulary.
        
        Returns:
            Dictionary with counts per category and risk level.
        """
        stats = {}
        for category in VOCABULARY_CATEGORIES:
            stats[category] = {
                risk: len(self.vocabulary[category][risk])
                for risk in RISK_LEVELS
            }
        return stats


class SentenceTemplate:
    """Represents a sentence template for combining vocabulary elements."""
    
    def __init__(self, template: str):
        """Initialize sentence template.
        
        Args:
            template: Template string with placeholders.
        """
        self.template = template
    
    def format(self, function: str, target: str, what: str) -> str:
        """Format template with vocabulary elements.
        
        Args:
            function: Function vocabulary term.
            target: Target vocabulary term.
            what: What vocabulary term.
            
        Returns:
            Formatted sentence with first letter capitalized.
        """
        try:
            sentence = self.template.format(
                function=function,
                target=target,
                what=what
            )
            # Capitalize first letter
            return sentence[0].upper() + sentence[1:] if sentence else ""
        except Exception:
            return ""


class SentenceGenerator:
    """Generate synthetic sentences from vocabulary combinations."""
    
    # Default sentence templates
    DEFAULT_TEMPLATES = [
        "{function} voor {target} met {what}",
        "{function} analyseert {what} van {target}",
        "{target} gebruikt {function} voor {what}",
        "{what} wordt verwerkt door {function} voor {target}",
        "{function} verwerkt {what} om {target} te helpen",
        "{target} profiteert van {function} die {what} gebruikt",
        "{function} optimaliseert {what} voor {target}",
        "{what} wordt geanalyseerd met {function} voor {target}",
    ]
    
    def __init__(self, vocabulary: Dict[str, Dict[str, List[str]]]):
        """Initialize sentence generator.
        
        Args:
            vocabulary: Dictionary of loaded vocabulary.
        """
        self.vocabulary = vocabulary
        self.templates = [SentenceTemplate(t) for t in self.DEFAULT_TEMPLATES]
    
    def generate_combinations(self, risk_level: str, limit: int) -> List[str]:
        """Generate sentences for a specific risk level.
        
        Args:
            risk_level: 'hr' for high-risk or 'lr' for low-risk.
            limit: Maximum number of sentences to generate.
            
        Returns:
            List of generated sentences.
        """
        if risk_level not in RISK_LEVELS:
            raise ValueError(f"Invalid risk level: {risk_level}")
        
        sentences = []
        
        if risk_level == 'hr':
            # High-risk: need at least 2 high-risk elements
            vocab_sets = self._get_high_risk_vocab_sets()
        else:
            # Low-risk: all elements must be low-risk
            vocab_sets = self._get_low_risk_vocab_sets()
        
        if not vocab_sets:
            print(f"Warning: No vocabulary sets available for {risk_level}")
            return sentences
        
        # Calculate how many sentences per template we need
        sentences_per_template = max(1, limit // len(self.templates))
        
        # Generate sentences using round-robin through vocabulary
        for template_idx, template in enumerate(self.templates):
            if len(sentences) >= limit:
                break
                
            # Generate sentences for this template
            for i in range(sentences_per_template):
                if len(sentences) >= limit:
                    break
                
                # Create balanced combination using modulo to cycle through terms
                combo = self._get_balanced_combination(vocab_sets, i, template_idx)
                if combo and all(combo):  # Ensure all elements are non-empty
                    sentence = template.format(*combo)
                    if sentence:
                        sentences.append(sentence)
        
        return sentences[:limit]  # Ensure we don't exceed limit
    
    def _get_balanced_combination(
        self, 
        vocab_sets: List[Dict[str, List[str]]], 
        iteration: int, 
        template_idx: int
    ) -> Optional[Tuple[str, str, str]]:
        """Get a balanced combination by cycling through vocabulary.
        
        Args:
            vocab_sets: List of vocabulary sets to use.
            iteration: Current iteration number.
            template_idx: Index of current template.
            
        Returns:
            Tuple of (function, target, what) or None if no valid combination.
        """
        if not vocab_sets:
            return None
            
        # Select which vocabulary set to use (rotate through them)
        vocab_set_idx = (iteration // 100) % len(vocab_sets)  # Change set every 100 iterations
        vocab_set = vocab_sets[vocab_set_idx]
        
        # Get terms from each category, cycling through them
        function_terms = vocab_set.get('function', [])
        target_terms = vocab_set.get('target', [])
        what_terms = vocab_set.get('what', [])
        
        if not (function_terms and target_terms and what_terms):
            return None
        
        # Use different prime numbers for each category to create variety
        # This ensures good distribution across all combinations
        function_idx = (iteration * 7 + template_idx * 3) % len(function_terms)
        target_idx = (iteration * 11 + template_idx * 5) % len(target_terms)
        what_idx = (iteration * 13 + template_idx * 7) % len(what_terms)
        
        return (
            function_terms[function_idx],
            target_terms[target_idx],
            what_terms[what_idx]
        )
    
    def _get_high_risk_vocab_sets(self) -> List[Dict[str, List[str]]]:
        """Get vocabulary sets for high-risk combinations.
        
        Returns:
            List of vocabulary sets that produce high-risk combinations.
        """
        vocab_sets = []
        
        # All high-risk (3 HR)
        if all(self.vocabulary[cat]['hr'] for cat in VOCABULARY_CATEGORIES):
            vocab_sets.append({
                'function': self.vocabulary['function']['hr'],
                'target': self.vocabulary['target']['hr'],
                'what': self.vocabulary['what']['hr']
            })
        
        # 2 high-risk combinations
        # Function + Target high-risk, What low-risk
        if (self.vocabulary['function']['hr'] and 
            self.vocabulary['target']['hr'] and 
            self.vocabulary['what']['lr']):
            vocab_sets.append({
                'function': self.vocabulary['function']['hr'],
                'target': self.vocabulary['target']['hr'],
                'what': self.vocabulary['what']['lr']
            })
        
        # Function + What high-risk, Target low-risk
        if (self.vocabulary['function']['hr'] and 
            self.vocabulary['target']['lr'] and 
            self.vocabulary['what']['hr']):
            vocab_sets.append({
                'function': self.vocabulary['function']['hr'],
                'target': self.vocabulary['target']['lr'],
                'what': self.vocabulary['what']['hr']
            })
        
        # Target + What high-risk, Function low-risk
        if (self.vocabulary['function']['lr'] and 
            self.vocabulary['target']['hr'] and 
            self.vocabulary['what']['hr']):
            vocab_sets.append({
                'function': self.vocabulary['function']['lr'],
                'target': self.vocabulary['target']['hr'],
                'what': self.vocabulary['what']['hr']
            })
        
        return vocab_sets
    
    def _get_low_risk_vocab_sets(self) -> List[Dict[str, List[str]]]:
        """Get vocabulary sets for low-risk combinations.
        
        Returns:
            List of vocabulary sets that produce low-risk combinations.
        """
        # Low-risk requires all elements to be low-risk
        if all(self.vocabulary[cat]['lr'] for cat in VOCABULARY_CATEGORIES):
            return [{
                'function': self.vocabulary['function']['lr'],
                'target': self.vocabulary['target']['lr'],
                'what': self.vocabulary['what']['lr']
            }]
        return []


def validate_vocabulary(vocabulary: Dict[str, Dict[str, List[str]]]) -> bool:
    """Validate that vocabulary has required terms for generation.
    
    Args:
        vocabulary: Loaded vocabulary dictionary.
        
    Returns:
        True if vocabulary is valid for generation.
    """
    # Check we have at least some terms in each category
    for category in VOCABULARY_CATEGORIES:
        if not any(vocabulary[category][risk] for risk in RISK_LEVELS):
            print(f"Error: No vocabulary loaded for category '{category}'")
            return False
    
    # Check we can generate at least some combinations
    has_hr_combination = False
    has_lr_combination = True  # All LR terms needed
    
    # Check for high-risk combinations (need at least 2 HR categories)
    hr_count = sum(1 for cat in VOCABULARY_CATEGORIES if vocabulary[cat]['hr'])
    if hr_count >= 2:
        has_hr_combination = True
    
    # Check for low-risk combination (all must have LR terms)
    for category in VOCABULARY_CATEGORIES:
        if not vocabulary[category]['lr']:
            has_lr_combination = False
            break
    
    if not has_hr_combination:
        print("Warning: Cannot generate high-risk combinations (need at least 2 HR categories)")
    if not has_lr_combination:
        print("Warning: Cannot generate low-risk combinations (all categories need LR terms)")
    
    return has_hr_combination or has_lr_combination


def main():
    """Main function to generate synthetic training data."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic training data from vocabulary combinations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=DEFAULT_SAMPLES_PER_RISK,
        help=f'Maximum number of samples to generate per risk level (default: {DEFAULT_SAMPLES_PER_RISK})'
    )
    parser.add_argument(
        '--vocab-dir',
        type=Path,
        default=DEFAULT_VOCAB_DIR,
        help=f'Path to vocabulary directory (default: {DEFAULT_VOCAB_DIR})'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for generated data (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f'Random seed for deterministic generation (default: {DEFAULT_RANDOM_SEED})'
    )
    
    args = parser.parse_args()
    
    # Set random seed for deterministic output
    random.seed(args.seed)
    
    # Load vocabulary
    print(f"Loading vocabulary from {args.vocab_dir}")
    loader = VocabularyLoader(args.vocab_dir)
    
    # Print vocabulary statistics
    stats = loader.get_vocabulary_stats()
    total_terms = sum(sum(counts.values()) for counts in stats.values())
    
    print("\nVocabulary Statistics:")
    print("-" * 40)
    for category, counts in stats.items():
        hr_count = counts['hr']
        lr_count = counts['lr']
        print(f"{category}: {hr_count} high-risk, {lr_count} low-risk terms")
    
    if total_terms == 0:
        print("\nError: No vocabulary terms loaded!")
        print("Please check that vocabulary files exist and are properly formatted.")
        return 1
    
    # Validate vocabulary
    if not validate_vocabulary(loader.vocabulary):
        print("\nError: Vocabulary validation failed!")
        return 1
    
    # Create output directories
    hr_dir = args.output_dir / 'hr'
    lr_dir = args.output_dir / 'lr'
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = SentenceGenerator(loader.vocabulary)
    
    # Generate high-risk sentences
    print(f"\nGenerating up to {args.limit} high-risk sentences...")
    hr_sentences = generator.generate_combinations('hr', args.limit)
    
    if hr_sentences:
        # Write high-risk sentences
        hr_file = hr_dir / 'synthetic_combinations.txt'
        with open(hr_file, 'w', encoding='utf-8') as f:
            for sentence in hr_sentences:
                f.write(sentence + '\n')
        print(f"Generated {len(hr_sentences)} high-risk sentences -> {hr_file}")
    else:
        print("Warning: No high-risk sentences generated")
    
    # Generate low-risk sentences
    print(f"\nGenerating up to {args.limit} low-risk sentences...")
    lr_sentences = generator.generate_combinations('lr', args.limit)
    
    if lr_sentences:
        # Write low-risk sentences
        lr_file = lr_dir / 'synthetic_combinations.txt'
        with open(lr_file, 'w', encoding='utf-8') as f:
            for sentence in lr_sentences:
                f.write(sentence + '\n')
        print(f"Generated {len(lr_sentences)} low-risk sentences -> {lr_file}")
    else:
        print("Warning: No low-risk sentences generated")
    
    # Print summary
    print("\nGeneration complete!")
    print(f"Total sentences generated: {len(hr_sentences) + len(lr_sentences)}")
    
    return 0


if __name__ == '__main__':
    exit(main())