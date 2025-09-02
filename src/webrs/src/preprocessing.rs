//! Text preprocessing module for Rust text classification
//! 
//! This module handles:
//! - Sentence tokenization
//! - Text normalization
//! - TF-IDF vectorization (compatible with scikit-learn)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;
// use unicode_segmentation::UnicodeSegmentation; // Currently unused
use crate::ClassifierError;

/// Configuration for the TF-IDF vectorizer (loaded from Python export)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorizerConfig {
    pub vocabulary: HashMap<String, usize>,
    pub feature_names: Vec<String>,
    pub max_features: Option<usize>,
    pub ngram_range: (usize, usize),
    pub lowercase: bool,
    pub stop_words: Option<Vec<String>>,
    pub analyzer: String,
    pub token_pattern: String,
    pub max_df: f64,
    pub min_df: f64,
    pub use_idf: bool,
    pub smooth_idf: bool,
    pub sublinear_tf: bool,
    pub norm: Option<String>,
    pub idf_weights: Vec<f32>,
}

/// Text preprocessor that mimics scikit-learn's TfidfVectorizer
/// Implements TF-IDF transformation using exported model parameters
pub struct TextPreprocessor {
    config: VectorizerConfig,
    token_regex: Regex,
    stop_words_set: Option<std::collections::HashSet<String>>,
}

impl TextPreprocessor {
    pub fn new(config: VectorizerConfig) -> Result<Self, ClassifierError> {
        // Compile token regex
        let token_regex = Regex::new(&config.token_pattern)
            .map_err(|e| ClassifierError::Preprocessing(format!("Invalid token pattern: {}", e)))?;
        
        // Convert stop words to HashSet for faster lookup
        let stop_words_set = config.stop_words.as_ref().map(|words| {
            words.iter().cloned().collect()
        });
        
        Ok(Self {
            config,
            token_regex,
            stop_words_set,
        })
    }
    
    /// Tokenize text into words using the same logic as scikit-learn
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        // Normalize and lowercase if configured
        let processed_text = if self.config.lowercase {
            text.to_lowercase()
        } else {
            text.to_string()
        };
        
        // Extract tokens using regex
        let mut tokens = Vec::new();
        for mat in self.token_regex.find_iter(&processed_text) {
            let token = mat.as_str().to_string();
            
            // Filter stop words
            if let Some(ref stop_words) = self.stop_words_set {
                if stop_words.contains(&token) {
                    continue;
                }
            }
            
            tokens.push(token);
        }
        
        tokens
    }
    
    /// Generate n-grams from tokens
    pub fn generate_ngrams(&self, tokens: &[String]) -> Vec<String> {
        let mut ngrams = Vec::new();
        let (min_n, max_n) = self.config.ngram_range;
        
        for n in min_n..=max_n {
            for window in tokens.windows(n) {
                let ngram = window.join(" ");
                ngrams.push(ngram);
            }
        }
        
        ngrams
    }
    
    /// Convert text to TF-IDF features
    pub fn vectorize(&self, text: &str) -> Result<Vec<f32>, ClassifierError> {
        // Tokenize text
        let tokens = self.tokenize(text);
        
        #[cfg(feature = "wasm")]
        web_sys::console::log_1(&format!("Text: '{}'", text).into());
        #[cfg(feature = "wasm")]
        web_sys::console::log_1(&format!("Tokens: {:?}", tokens).into());
        
        // Generate n-grams
        let ngrams = self.generate_ngrams(&tokens);
        
        #[cfg(feature = "wasm")]
        web_sys::console::log_1(&format!("N-grams ({}): {:?}", ngrams.len(), &ngrams[..ngrams.len().min(10)]).into());
        
        // Count term frequencies
        let mut term_counts: HashMap<String, usize> = HashMap::new();
        for ngram in &ngrams {
            *term_counts.entry(ngram.clone()).or_insert(0) += 1;
        }
        
        // Create feature vector
        let mut features = vec![0.0f32; self.config.feature_names.len()];
        
        let mut found_terms = Vec::new();
        let mut missing_terms = Vec::new();
        
        for (term, count) in term_counts {
            if let Some(&feature_idx) = self.config.vocabulary.get(&term) {
                if feature_idx < features.len() {
                    let tf = count as f32;
                    
                    // Apply sublinear TF if configured
                    let final_tf = if self.config.sublinear_tf {
                        1.0 + tf.ln()
                    } else {
                        tf
                    };
                    
                    // Apply IDF if configured and available
                    let final_value = if self.config.use_idf && feature_idx < self.config.idf_weights.len() {
                        final_tf * self.config.idf_weights[feature_idx]
                    } else {
                        final_tf
                    };
                    
                    features[feature_idx] = final_value;
                    found_terms.push((term.clone(), feature_idx, final_value));
                }
            } else {
                missing_terms.push(term);
            }
        }
        
        #[cfg(feature = "wasm")]
        web_sys::console::log_1(&format!("Found in vocab: {:?}", found_terms).into());
        #[cfg(feature = "wasm")]
        if !missing_terms.is_empty() {
            web_sys::console::log_1(&format!("Missing from vocab: {:?}", missing_terms).into());
        }
        
        // Apply normalization if configured
        if let Some(ref norm) = self.config.norm {
            match norm.as_str() {
                "l1" => {
                    let sum: f32 = features.iter().sum();
                    if sum > 0.0 {
                        for feature in &mut features {
                            *feature /= sum;
                        }
                    }
                },
                "l2" => {
                    let sum_sq: f32 = features.iter().map(|x| x * x).sum();
                    let norm_val = sum_sq.sqrt();
                    if norm_val > 0.0 {
                        for feature in &mut features {
                            *feature /= norm_val;
                        }
                    }
                },
                _ => {} // No normalization or unknown norm
            }
        }
        
        Ok(features)
    }
}

/// Tokenize text into sentences
pub fn tokenize_sentences(text: &str) -> Result<Vec<String>, ClassifierError> {
    // Simple sentence splitting - in production, you might want to use a more sophisticated approach
    let sentence_endings = Regex::new(r"[.!?]+\s+")
        .map_err(|e| ClassifierError::Preprocessing(e.to_string()))?;
    
    let sentences: Vec<String> = sentence_endings
        .split(text)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    
    if sentences.is_empty() {
        // If no sentence endings found, treat the whole text as one sentence
        Ok(vec![text.trim().to_string()])
    } else {
        Ok(sentences)
    }
}

/// Vectorize a single sentence using the vectorizer configuration
pub fn vectorize_sentence(
    sentence: &str,
    config: &VectorizerConfig,
) -> Result<Vec<f32>, ClassifierError> {
    let preprocessor = TextPreprocessor::new(config.clone())?;
    preprocessor.vectorize(sentence)
}

/// Normalize text (Unicode normalization, whitespace cleanup)
pub fn normalize_text(text: &str) -> String {
    // Unicode normalization
    let normalized: String = text.nfc().collect();
    
    // Normalize whitespace
    let whitespace_regex = Regex::new(r"\s+").unwrap();
    let cleaned = whitespace_regex.replace_all(&normalized, " ");
    
    cleaned.trim().to_string()
}

/// Clean text by removing unwanted characters
pub fn clean_text(text: &str) -> String {
    // Remove or replace problematic characters
    let mut cleaned = text.to_string();
    
    // Remove control characters except newlines and tabs
    cleaned = cleaned.chars()
        .filter(|&c| !c.is_control() || c == '\n' || c == '\t')
        .collect();
    
    // Normalize quotes and dashes
    cleaned = cleaned
        .replace(['\u{201C}', '\u{201D}'], "\"")  // Smart quotes to regular quotes
        .replace(['\u{2014}', '\u{2013}'], "-")   // Em/en dashes to hyphens  
        .replace('\u{2019}', "'")         // Smart apostrophes
        .replace('\u{2026}', "...");      // Ellipsis
    
    normalize_text(&cleaned)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sentence_tokenization() {
        let text = "This is sentence one. This is sentence two! And this is sentence three?";
        let sentences = tokenize_sentences(text).unwrap();
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "This is sentence one");
        assert_eq!(sentences[1], "This is sentence two");
        assert_eq!(sentences[2], "And this is sentence three?");
    }
    
    #[test]
    fn test_text_cleaning() {
        let dirty_text = "This  has   multiple\t\tspaces\nand \u{201C}smart\u{201D} quotes\u{2014}plus em-dashes.";
        let cleaned = clean_text(dirty_text);
        assert!(!cleaned.contains('\u{201C}'));
        assert!(!cleaned.contains('\u{2014}'));
    }
    
    #[test]
    fn test_normalization() {
        let text = "  Multiple    spaces   and   tabs\t\t  ";
        let normalized = normalize_text(text);
        assert_eq!(normalized, "Multiple spaces and tabs");
    }
}