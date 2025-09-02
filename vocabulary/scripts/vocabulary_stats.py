#!/usr/bin/env python3
"""
Vocabulary Statistics Analyzer
Analyzes vocabulary files for distribution, formatting, and quality metrics.
"""

import os
import re
from pathlib import Path
from collections import defaultdict, Counter
import argparse


class VocabularyStats:
    def __init__(self, vocab_root):
        self.vocab_root = Path(vocab_root)
        self.categories = ['function', 'target', 'what', 'sentence']
        self.stats = defaultdict(lambda: defaultdict(dict))
        self.formatting_issues = []
        
    def is_header_line(self, line):
        """Check if line is part of header (comment or empty)"""
        line = line.strip()
        return line.startswith('#') or line == '' or line == '# END HEADER RULES'
    
    def is_valid_header(self, lines):
        """Check if file has proper header format"""
        if len(lines) < 10:
            return False
            
        # Check for required header elements
        header_text = '\n'.join(lines[:30])  # Check first 30 lines
        required_elements = [
            'RISK',
            'PATH:',
            'DIMENSION:',
            'HEADER RULES',
            'VALIDATION RULES:',
            'FORMAT:',
            'END HEADER RULES'
        ]
        
        return all(element in header_text for element in required_elements)
    
    def count_words_in_term(self, term):
        """Count words in a vocabulary term"""
        # Remove common patterns that shouldn't count as separate words
        term = re.sub(r'\([^)]+\)', '', term)  # Remove parenthetical content
        term = re.sub(r'\s+', ' ', term.strip())  # Normalize whitespace
        
        if not term:
            return 0
            
        # Split on spaces and count non-empty parts
        words = [w for w in term.split() if w]
        return len(words)
    
    def analyze_file(self, file_path):
        """Analyze a single vocabulary file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.rstrip('\n\r') for line in f.readlines()]
        except Exception as e:
            return {
                'error': str(e),
                'valid_header': False,
                'term_count': 0,
                'avg_word_length': 0,
                'long_terms': []
            }
        
        # Check header validity
        valid_header = self.is_valid_header(lines)
        
        # Extract vocabulary terms (non-header lines)
        vocab_terms = []
        past_header = False
        
        for line in lines:
            original_line = line
            line = line.strip()
            if line == '# END HEADER RULES':
                past_header = True
                continue
            if past_header and line and not line.startswith('#') and line:
                # Skip section headers like "# Basis filter functies"
                vocab_terms.append(line)
        
        # Calculate word length statistics
        word_counts = [self.count_words_in_term(term) for term in vocab_terms if term]
        avg_word_length = sum(word_counts) / len(word_counts) if word_counts else 0
        
        # Determine category and max allowed word length
        rel_path = file_path.relative_to(self.vocab_root)
        category = rel_path.parts[0]
        max_words = 5 if category == 'function' else 3
        
        # Find terms that are too long
        long_terms = []
        for i, term in enumerate(vocab_terms):
            word_count = word_counts[i] if i < len(word_counts) else 0
            if word_count > max_words:
                long_terms.append(f"{term} ({word_count} words)")
        
        return {
            'valid_header': valid_header,
            'term_count': len(vocab_terms),
            'avg_word_length': avg_word_length,
            'long_terms': long_terms,
            'category': category,
            'risk_level': 'hr' if '/hr/' in str(file_path) else 'lr'
        }
    
    def analyze_all_files(self):
        """Analyze all vocabulary files"""
        total_files = 0
        
        for category in self.categories:
            category_path = self.vocab_root / category
            if not category_path.exists():
                continue
                
            # Initialize category stats
            self.stats[category]['hr'] = {'files': 0, 'terms': 0, 'word_lengths': [], 'issues': [], 'all_terms': set(), 'term_files': defaultdict(list)}
            self.stats[category]['lr'] = {'files': 0, 'terms': 0, 'word_lengths': [], 'issues': [], 'all_terms': set(), 'term_files': defaultdict(list)}
            
            # Find all .txt files
            for txt_file in category_path.rglob('*.txt'):
                total_files += 1
                analysis = self.analyze_file(txt_file)
                
                if 'error' in analysis:
                    self.formatting_issues.append({
                        'file': str(txt_file.relative_to(self.vocab_root)),
                        'issue': f"Read error: {analysis['error']}"
                    })
                    continue
                
                risk_level = analysis['risk_level']
                
                # Update statistics
                self.stats[category][risk_level]['files'] += 1
                self.stats[category][risk_level]['terms'] += analysis['term_count']
                if analysis['avg_word_length'] > 0:
                    self.stats[category][risk_level]['word_lengths'].append(analysis['avg_word_length'])
                
                # Collect all terms for duplicate checking
                rel_file_path = str(txt_file.relative_to(self.vocab_root))
                with open(txt_file, 'r', encoding='utf-8') as f:
                    past_header = False
                    for line in f:
                        line = line.strip()
                        if line == '# END HEADER RULES':
                            past_header = True
                            continue
                        if past_header and line and not line.startswith('#'):
                            # Store lowercase for comparison
                            term_lower = line.lower()
                            self.stats[category][risk_level]['all_terms'].add(term_lower)
                            self.stats[category][risk_level]['term_files'][term_lower].append(rel_file_path)
                
                # Check for formatting issues
                rel_path = str(txt_file.relative_to(self.vocab_root))
                
                if not analysis['valid_header']:
                    self.formatting_issues.append({
                        'file': rel_path,
                        'issue': "Missing or invalid header"
                    })
                
                if analysis['long_terms']:
                    max_words = 5 if category == 'function' else 3
                    self.formatting_issues.append({
                        'file': rel_path,
                        'issue': f"Terms exceed {max_words} words: {'; '.join(analysis['long_terms'])}"
                    })
        
        return total_files
    
    def print_distribution_stats(self):
        """Print high/low risk distribution statistics"""
        print("📊 VOCABULARY DISTRIBUTION STATISTICS")
        print("=" * 50)
        
        total_files = 0
        total_terms = 0
        
        for category in self.categories:
            if category not in self.stats:
                continue
                
            hr_files = self.stats[category]['hr']['files']
            lr_files = self.stats[category]['lr']['files']
            hr_terms = self.stats[category]['hr']['terms']
            lr_terms = self.stats[category]['lr']['terms']
            
            cat_files = hr_files + lr_files
            cat_terms = hr_terms + lr_terms
            
            total_files += cat_files
            total_terms += cat_terms
            
            print(f"\n{category.upper()}")
            print(f"  Files: {cat_files:3d} total ({hr_files:2d} HR, {lr_files:2d} LR)")
            print(f"  Terms: {cat_terms:4d} total ({hr_terms:3d} HR, {lr_terms:3d} LR)")
            
            if cat_files > 0:
                hr_pct = (hr_files / cat_files) * 100
                print(f"  Ratio: {hr_pct:.1f}% HR, {100-hr_pct:.1f}% LR")
        
        print(f"\nTOTAL ACROSS ALL CATEGORIES:")
        print(f"  Files: {total_files}")
        print(f"  Terms: {total_terms}")
    
    def print_word_length_stats(self):
        """Print word length statistics"""
        print("\n\n📏 WORD LENGTH ANALYSIS")
        print("=" * 50)
        
        for category in self.categories:
            if category not in self.stats:
                continue
                
            hr_lengths = self.stats[category]['hr']['word_lengths']
            lr_lengths = self.stats[category]['lr']['word_lengths']
            
            print(f"\n{category.upper()}")
            
            if hr_lengths:
                hr_avg = sum(hr_lengths) / len(hr_lengths)
                print(f"  HR avg words/term: {hr_avg:.2f}")
            else:
                print(f"  HR avg words/term: N/A")
                
            if lr_lengths:
                lr_avg = sum(lr_lengths) / len(lr_lengths)
                print(f"  LR avg words/term: {lr_avg:.2f}")
            else:
                print(f"  LR avg words/term: N/A")
            
            # Combined average
            all_lengths = hr_lengths + lr_lengths
            if all_lengths:
                combined_avg = sum(all_lengths) / len(all_lengths)
                print(f"  Combined average:  {combined_avg:.2f}")
                
                max_allowed = 5 if category == 'function' else 3
                print(f"  Max allowed:      {max_allowed}")
    
    def check_duplicate_terms(self):
        """Check for terms that appear in both HR and LR"""
        duplicate_issues = []
        
        for category in self.categories:
            if category not in self.stats:
                continue
                
            hr_terms = self.stats[category]['hr'].get('all_terms', set())
            lr_terms = self.stats[category]['lr'].get('all_terms', set())
            hr_term_files = self.stats[category]['hr'].get('term_files', defaultdict(list))
            lr_term_files = self.stats[category]['lr'].get('term_files', defaultdict(list))
            
            # Find overlapping terms
            duplicates = hr_terms.intersection(lr_terms)
            
            if duplicates:
                # Collect detailed information for each duplicate term
                duplicate_details = []
                for term in sorted(duplicates):
                    hr_files = hr_term_files.get(term, [])
                    lr_files = lr_term_files.get(term, [])
                    duplicate_details.append({
                        'term': term,
                        'hr_files': hr_files,
                        'lr_files': lr_files
                    })
                
                duplicate_issues.append({
                    'category': category,
                    'count': len(duplicates),
                    'details': duplicate_details
                })
        
        return duplicate_issues
    
    def print_formatting_issues(self):
        """Print formatting validation results"""
        print("\n\n🔍 FORMATTING VALIDATION")
        print("=" * 50)
        
        # Check for HR/LR duplicates first
        duplicate_issues = self.check_duplicate_terms()
        if duplicate_issues:
            print("\n❌ CRITICAL: Terms appearing in BOTH HR and LR:")
            for issue in duplicate_issues:
                print(f"\n{issue['category'].upper()}: {issue['count']} duplicate terms")
                print("Duplicate terms with file locations:")
                for detail in issue['details']:
                    print(f"  - '{detail['term']}'")
                    print(f"    HR files: {', '.join(detail['hr_files'])}")
                    print(f"    LR files: {', '.join(detail['lr_files'])}")
                    print()
        
        if not self.formatting_issues and not duplicate_issues:
            print("✅ No formatting issues found!")
            return
        
        # Group issues by type
        issue_types = defaultdict(list)
        for issue in self.formatting_issues:
            if "header" in issue['issue'].lower():
                issue_types['Header Issues'].append(issue)
            elif "words" in issue['issue'].lower():
                issue_types['Word Length Issues'].append(issue)
            else:
                issue_types['Other Issues'].append(issue)
        
        for issue_type, issues in issue_types.items():
            print(f"\n{issue_type} ({len(issues)} files):")
            for issue in issues:  # Show all issues
                print(f"  ❌ {issue['file']}")
                print(f"     {issue['issue']}")
    
    def run_analysis(self, show_issues=True):
        """Run complete analysis and print results"""
        print("Analyzing vocabulary files...")
        total_files = self.analyze_all_files()
        
        if total_files == 0:
            print("No vocabulary files found!")
            return
        
        print(f"Analyzed {total_files} files\n")
        
        self.print_distribution_stats()
        self.print_word_length_stats()
        
        if show_issues:
            self.print_formatting_issues()


def main():
    parser = argparse.ArgumentParser(description='Analyze vocabulary statistics')
    parser.add_argument('--vocab-root', default='.', 
                       help='Root directory of vocabulary files (default: current directory)')
    parser.add_argument('--no-issues', action='store_true',
                       help='Skip formatting issues report')
    parser.add_argument('--category', choices=['context', 'function', 'target', 'what'],
                       help='Analyze only specific category')
    parser.add_argument('--summary-only', action='store_true',
                       help='Show only summary statistics')
    
    args = parser.parse_args()
    
    analyzer = VocabularyStats(args.vocab_root)
    
    # Filter categories if specified
    if args.category:
        analyzer.categories = [args.category]
    
    if args.summary_only:
        total_files = analyzer.analyze_all_files()
        if total_files > 0:
            print(f"Total files: {total_files}")
            analyzer.print_distribution_stats()
    else:
        analyzer.run_analysis(show_issues=not args.no_issues)


if __name__ == '__main__':
    main()
