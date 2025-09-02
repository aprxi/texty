//! Texty WebRS - Rust/WASM Text Classification Library
//! 
//! This library provides text classification functionality that can run in:
//! - Native Rust applications
//! - WebAssembly (WASM) in browsers
//! - Server-side applications
//! 
//! It uses ONNX models exported from the Python scikit-learn pipeline.

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod classifier;
pub mod preprocessing;
pub mod utils;
pub mod onnx_parser;

#[cfg(feature = "wasm")]
use serde_wasm_bindgen;

/// Configuration for the text classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
    pub confidence_threshold: f32,
    pub count_threshold: usize,
    pub percentage_threshold: f32,
    pub use_weighted_scoring: bool,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.7,
            count_threshold: 1,
            percentage_threshold: 0.1,
            use_weighted_scoring: false,
        }
    }
}

/// Result of text classification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen(getter_with_clone)]
pub struct ClassificationResult {
    pub classification: String,
    pub confidence: f32,
    pub high_risk_sentences: usize,
    pub total_sentences: usize,
    pub percentage_high_risk: f32,
    pub processing_time_ms: f64,
    pub sentence_details: Vec<SentenceResult>,
}

/// Result for individual sentence classification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen(getter_with_clone)]
pub struct SentenceResult {
    pub sentence: String,
    pub prediction: String,
    pub effective_prediction: String,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
    pub low_risk_prob: f32,
    pub high_risk_prob: f32,
    pub active_features_count: usize,
    pub top_features: Vec<FeatureContribution>,
}

/// Feature contribution details for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen(getter_with_clone)]
pub struct FeatureContribution {
    pub word: String,
    pub tfidf_weight: f32,
    pub contribution: f32,
    pub feature_index: usize,
}

/// Error types for the classifier
#[derive(Debug, thiserror::Error)]
pub enum ClassifierError {
    #[error("Model loading error: {0}")]
    ModelLoad(String),
    
    #[error("Preprocessing error: {0}")]
    Preprocessing(String),
    
    #[error("Inference error: {0}")]
    Inference(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
}

/// Main text classifier struct (pure Rust implementation)
pub struct TextClassifier {
    model_loaded: bool,
    nb_model: Option<onnx_parser::NaiveBayesModel>,
    vectorizer_config: Option<preprocessing::VectorizerConfig>,
    config: ClassifierConfig,
    class_names: Vec<String>,
}

impl TextClassifier {
    /// Create a new text classifier
    pub fn new() -> Self {
        Self {
            model_loaded: false,
            nb_model: None,
            vectorizer_config: None,
            config: ClassifierConfig::default(),
            class_names: vec!["low-risk".to_string(), "high-risk".to_string()],
        }
    }
    
    /// Load model from ONNX bytes and vectorizer configuration
    pub fn load_model(
        &mut self,
        onnx_bytes: &[u8],
        vectorizer_config: preprocessing::VectorizerConfig,
    ) -> Result<(), ClassifierError> {
        // Parse ONNX model and load vectorizer configuration
        println!("[PURE_RUST] Loading ONNX model of {} bytes", onnx_bytes.len());
        
        let (nb_model, vec_config) = onnx_parser::load_model_from_artifacts(
            onnx_bytes, 
            &serde_json::to_string(&vectorizer_config)
                .map_err(|e| ClassifierError::ModelLoad(format!("Failed to serialize config: {}", e)))?
        )?;
        
        self.nb_model = Some(nb_model);
        self.vectorizer_config = Some(vec_config);
        self.model_loaded = true;
        self.vectorizer_config = Some(vectorizer_config);
        
        Ok(())
    }
    
    /// Load model with complete metadata containing real classifier parameters
    pub fn load_model_with_metadata(
        &mut self,
        onnx_bytes: &[u8],
        vectorizer_json: &str,
        metadata_json: &str,
    ) -> Result<(), ClassifierError> {
        println!("[PURE_RUST] Loading ONNX model with metadata: {} bytes", onnx_bytes.len());
        
        let (nb_model, vec_config) = onnx_parser::load_model_from_artifacts_with_metadata(
            onnx_bytes, 
            vectorizer_json,
            metadata_json
        )?;
        
        self.nb_model = Some(nb_model);
        self.vectorizer_config = Some(vec_config);
        self.model_loaded = true;
        
        Ok(())
    }
    
    /// Classify text and return detailed results
    pub fn classify_text(&self, text: &str) -> Result<ClassificationResult, ClassifierError> {
        // WASM-compatible timing (use 0.0 as placeholder)
        let _start_time = 0.0;
        
        // Check if model is loaded
        if !self.model_loaded {
            return Err(ClassifierError::Configuration("Model not loaded".to_string()));
        }
        
        let vectorizer_config = self.vectorizer_config.as_ref()
            .ok_or_else(|| ClassifierError::Configuration("Vectorizer config not loaded".to_string()))?;
        
        // Preprocess text into sentences
        let sentences = preprocessing::tokenize_sentences(text)?;
        let total_sentences = sentences.len();
        
        if total_sentences == 0 {
            return Ok(ClassificationResult {
                classification: "neutral".to_string(),
                confidence: 0.0,
                high_risk_sentences: 0,
                total_sentences: 0,
                percentage_high_risk: 0.0,
                processing_time_ms: 0.0, // Timing not available in WASM stub
                sentence_details: vec![],
            });
        }
        
        // Process each sentence
        let mut sentence_results = Vec::new();
        let mut high_risk_count = 0;
        
        for sentence in &sentences {
            // Vectorize sentence
            let features = preprocessing::vectorize_sentence(sentence, vectorizer_config)?;
            
            // Debug: Log feature vector info
            let non_zero_features: Vec<(usize, f32)> = features.iter()
                .enumerate()
                .filter(|(_, &val)| val > 0.0)
                .map(|(idx, &val)| (idx, val))
                .collect();
            
            #[cfg(feature = "wasm")]
            console_log!("=== SENTENCE DEBUG ===");
            #[cfg(feature = "wasm")]
            console_log!("Sentence: '{}'", sentence);
            #[cfg(feature = "wasm")]
            console_log!("Non-zero features: {} out of {}", non_zero_features.len(), features.len());
            #[cfg(feature = "wasm")]
            if !non_zero_features.is_empty() {
                console_log!("First 10 features: {:?}", &non_zero_features[..non_zero_features.len().min(10)]);
            } else {
                console_log!("ERROR: NO FEATURES DETECTED!");
            }
            
            // Run inference
            let probabilities = self.run_inference(&features)?;
            
            #[cfg(feature = "wasm")]
            console_log!("Probabilities: [{:.6}, {:.6}]", probabilities[0], probabilities[1]);
            
            // Determine prediction
            let prediction_idx = probabilities.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            let prediction = self.class_names.get(prediction_idx)
                .unwrap_or(&"unknown".to_string())
                .clone();
            
            let confidence = probabilities[prediction_idx];
            let low_risk_prob = probabilities.get(0).copied().unwrap_or(0.0);
            let high_risk_prob = probabilities.get(1).copied().unwrap_or(0.0);
            
            // Calculate feature contributions for detailed analysis
            let top_features = self.calculate_feature_contributions(&features, &non_zero_features, vectorizer_config)?;
            
            // Check if sentence meets high-risk criteria
            let effective_prediction = if prediction_idx == 1 && high_risk_prob >= self.config.confidence_threshold {
                high_risk_count += 1;
                "high-risk".to_string()
            } else {
                prediction.clone()
            };
            
            #[cfg(feature = "wasm")]
            {
                let status = if effective_prediction == "high-risk" { "HIGH-RISK" }
                else if high_risk_prob > 0.3 { "MODERATE" }
                else { "LOW-RISK" };
                
                console_log!("=== SENTENCE ANALYSIS ===");
                console_log!("Text: \"{}\"", sentence);
                console_log!("Status: {} | High-risk: {:.1}% | Confidence threshold: {:.0}%", 
                    status, high_risk_prob * 100.0, self.config.confidence_threshold * 100.0);
                console_log!("Probabilities: Low-risk={:.3}, High-risk={:.3}", low_risk_prob, high_risk_prob);
                console_log!("Active features: {}", non_zero_features.len());
                
                if !top_features.is_empty() {
                    console_log!("Top feature contributions:");
                    for (i, feat) in top_features.iter().take(5).enumerate() {
                        let sign = if feat.contribution > 0.0 { "+" } else { "-" };
                        console_log!("  {}. '{}' | TF-IDF: {:.3} | Contribution: {}{:.3}", 
                            i + 1, feat.word, feat.tfidf_weight, sign, feat.contribution.abs());
                    }
                }
                
                if effective_prediction == "high-risk" {
                    console_log!("Result: This sentence TRIGGERS high-risk classification");
                } else if high_risk_prob > 0.3 {
                    console_log!("Result: Potentially risky but below {:.0}% threshold", self.config.confidence_threshold * 100.0);
                } else {
                    console_log!("Result: No significant risk signals detected");
                }
                console_log!("=====================================");
            }
            
            sentence_results.push(SentenceResult {
                sentence: sentence.clone(),
                prediction: prediction.clone(),
                effective_prediction,
                confidence,
                probabilities: probabilities.clone(),
                low_risk_prob,
                high_risk_prob,
                active_features_count: non_zero_features.len(),
                top_features,
            });
        }
        
        // Determine final classification
        let percentage_high_risk = if total_sentences > 0 {
            (high_risk_count as f32 / total_sentences as f32) * 100.0
        } else {
            0.0
        };
        
        let final_classification = if high_risk_count >= self.config.count_threshold ||
            (percentage_high_risk / 100.0) >= self.config.percentage_threshold {
            "high-risk"
        } else {
            "neutral"
        };
        
        let avg_confidence = if !sentence_results.is_empty() {
            sentence_results.iter()
                .map(|s| s.confidence)
                .sum::<f32>() / sentence_results.len() as f32
        } else {
            0.0
        };
        
        Ok(ClassificationResult {
            classification: final_classification.to_string(),
            confidence: avg_confidence,
            high_risk_sentences: high_risk_count,
            total_sentences,
            percentage_high_risk,
            processing_time_ms: 0.0, // Timing not available in WASM stub
            sentence_details: sentence_results,
        })
    }
    
    /// Run inference using real Naive Bayes implementation
    fn run_inference(&self, features: &[f32]) -> Result<Vec<f32>, ClassifierError> {
        if !self.model_loaded {
            return Err(ClassifierError::Inference("Model not loaded".to_string()));
        }
        
        let nb_model = self.nb_model.as_ref()
            .ok_or_else(|| ClassifierError::Inference("Naive Bayes model not loaded".to_string()))?;
        
        // Use the real Naive Bayes model for prediction
        nb_model.predict_proba(features)
    }
    
    /// Update classifier configuration
    pub fn set_config(&mut self, config: ClassifierConfig) {
        self.config = config;
    }
    
    /// Calculate feature contributions for detailed analysis (similar to Python version)
    fn calculate_feature_contributions(
        &self,
        features: &[f32],
        non_zero_features: &[(usize, f32)],
        vectorizer_config: &preprocessing::VectorizerConfig,
    ) -> Result<Vec<FeatureContribution>, ClassifierError> {
        let mut contributions = Vec::new();
        
        let nb_model = self.nb_model.as_ref()
            .ok_or_else(|| ClassifierError::Inference("Naive Bayes model not loaded".to_string()))?;
        
        // Create reverse vocabulary mapping for feature index -> word
        let mut index_to_word = std::collections::HashMap::new();
        for (word, &index) in &vectorizer_config.vocabulary {
            index_to_word.insert(index, word.clone());
        }
        
        for &(feature_idx, tfidf_weight) in non_zero_features {
            if feature_idx < nb_model.feature_log_probs[0].len() && feature_idx < nb_model.feature_log_probs[1].len() {
                // Get log probabilities for this feature
                let low_risk_log_prob = nb_model.feature_log_probs[0][feature_idx];
                let high_risk_log_prob = nb_model.feature_log_probs[1][feature_idx];
                
                // Calculate contribution (higher = more high-risk)
                let contribution = high_risk_log_prob - low_risk_log_prob;
                
                // Get word for this feature index
                let word = index_to_word.get(&feature_idx)
                    .map(|s| s.clone())
                    .unwrap_or_else(|| format!("feature_{}", feature_idx));
                
                contributions.push(FeatureContribution {
                    word,
                    tfidf_weight,
                    contribution,
                    feature_index: feature_idx,
                });
            }
        }
        
        // Sort by contribution (most high-risk first)
        contributions.sort_by(|a, b| b.contribution.partial_cmp(&a.contribution).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(contributions)
    }
}

// WASM bindings
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmTextClassifier {
    inner: TextClassifier,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmTextClassifier {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        utils::set_panic_hook();
        Self {
            inner: TextClassifier::new(),
        }
    }
    
    #[wasm_bindgen]
    pub fn load_model_from_bytes(
        &mut self,
        onnx_bytes: &[u8],
        vectorizer_config_json: &str,
    ) -> Result<(), JsValue> {
        let vectorizer_config: preprocessing::VectorizerConfig = 
            serde_json::from_str(vectorizer_config_json)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        self.inner.load_model(onnx_bytes, vectorizer_config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn load_model_with_metadata(
        &mut self,
        onnx_bytes: &[u8],
        vectorizer_config_json: &str,
        metadata_json: &str,
    ) -> Result<(), JsValue> {
        self.inner.load_model_with_metadata(onnx_bytes, vectorizer_config_json, metadata_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        Ok(())
    }
    
    #[wasm_bindgen]
    pub fn classify(&self, text: &str) -> Result<JsValue, JsValue> {
        let result = self.inner.classify_text(text)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        Ok(serde_wasm_bindgen::to_value(&result)?)
    }
    
    #[wasm_bindgen]
    pub fn set_config(&mut self, config_json: &str) -> Result<(), JsValue> {
        let config: ClassifierConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        self.inner.set_config(config);
        Ok(())
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[cfg(feature = "wasm")]
macro_rules! console_log {
    ($($t:tt)*) => (web_sys::console::log_1(&format_args!($($t)*).to_string().into()))
}

#[cfg(feature = "wasm")]
pub(crate) use console_log;