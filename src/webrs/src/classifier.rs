//! Core classification logic and model management

use crate::{ClassifierError, ClassifierConfig, ClassificationResult, SentenceResult};
use crate::preprocessing::{VectorizerConfig, tokenize_sentences, vectorize_sentence};
// ONNX inference implemented as pure Rust for WASM compatibility

/// High-level classifier interface (stub implementation)
pub struct Classifier {
    // session: Session, // Temporarily disabled due to ort API compatibility
    model_loaded: bool,
    // TODO: Add pure Rust model weights when implementing real inference
    weights: Vec<f32>,
    vectorizer_config: VectorizerConfig,
    config: ClassifierConfig,
    class_names: Vec<String>,
}

impl Classifier {
    /// Create a new classifier from ONNX model bytes and vectorizer config
    pub fn from_bytes(
        onnx_bytes: &[u8],
        vectorizer_config: VectorizerConfig,
        config: ClassifierConfig,
    ) -> Result<Self, ClassifierError> {
        // TODO: Implement proper ONNX session loading once ort API is stable
        // let session = Session::from_memory(onnx_bytes)
        //     .map_err(|e| ClassifierError::ModelLoad(e.to_string()))?;
        
        println!("[STUB] Loading ONNX model of {} bytes", onnx_bytes.len());
        
        // TODO: Parse ONNX model and extract weights
        let weights = vec![0.0f32; 1000]; // Placeholder weights
        
        Ok(Self {
            model_loaded: true,
            weights,
            vectorizer_config,
            config,
            class_names: vec![
                "low-risk".to_string(),
                "neutral".to_string(),
                "high-risk".to_string()
            ],
        })
    }
    
    /// Load classifier from file paths
    #[cfg(not(feature = "wasm"))]
    pub fn from_files(
        onnx_path: &str,
        vectorizer_config_path: &str,
        config: ClassifierConfig,
    ) -> Result<Self, ClassifierError> {
        use std::fs;
        
        // Load ONNX model
        let onnx_bytes = fs::read(onnx_path)
            .map_err(|e| ClassifierError::ModelLoad(format!("Failed to read ONNX file: {}", e)))?;
        
        // Load vectorizer config
        let config_json = fs::read_to_string(vectorizer_config_path)
            .map_err(|e| ClassifierError::ModelLoad(format!("Failed to read vectorizer config: {}", e)))?;
        
        let vectorizer_config: VectorizerConfig = serde_json::from_str(&config_json)
            .map_err(|e| ClassifierError::ModelLoad(format!("Failed to parse vectorizer config: {}", e)))?;
        
        Self::from_bytes(&onnx_bytes, vectorizer_config, config)
    }
    
    /// Classify a single text
    pub fn classify(&self, text: &str) -> Result<ClassificationResult, ClassifierError> {
        // WASM-compatible timing (use 0.0 as placeholder)
        let _start_time = 0.0;
        
        // Tokenize into sentences
        let sentences = tokenize_sentences(text)?;
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
        let mut total_confidence = 0.0;
        
        for sentence in &sentences {
            let result = self.classify_sentence(sentence)?;
            
            // Check if this sentence contributes to high-risk classification
            if result.prediction == "high-risk" && 
               result.probabilities.get(2).unwrap_or(&0.0) >= &self.config.confidence_threshold {
                high_risk_count += 1;
            }
            
            total_confidence += result.confidence;
            sentence_results.push(result);
        }
        
        // Calculate final metrics
        let percentage_high_risk = if total_sentences > 0 {
            (high_risk_count as f32 / total_sentences as f32) * 100.0
        } else {
            0.0
        };
        
        let avg_confidence = if total_sentences > 0 {
            total_confidence / total_sentences as f32
        } else {
            0.0
        };
        
        // Determine final classification based on thresholds
        let final_classification = self.aggregate_classification(high_risk_count, percentage_high_risk);
        
        Ok(ClassificationResult {
            classification: final_classification,
            confidence: avg_confidence,
            high_risk_sentences: high_risk_count,
            total_sentences,
            percentage_high_risk,
            processing_time_ms: 0.0, // Timing not available in WASM stub
            sentence_details: sentence_results,
        })
    }
    
    /// Classify a single sentence
    fn classify_sentence(&self, sentence: &str) -> Result<SentenceResult, ClassifierError> {
        // Vectorize the sentence
        let features = vectorize_sentence(sentence, &self.vectorizer_config)?;
        
        // Run inference
        let probabilities = self.run_inference(&features)?;
        
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
        
        Ok(SentenceResult {
            sentence: sentence.to_string(),
            prediction: prediction.clone(),
            effective_prediction: prediction,
            confidence,
            probabilities: probabilities.clone(),
            low_risk_prob: probabilities.get(0).copied().unwrap_or(0.0),
            high_risk_prob: probabilities.get(1).copied().unwrap_or(0.0),
            active_features_count: 0, // Not calculated in stub classifier
            top_features: vec![], // Empty for stub classifier
        })
    }
    
    /// Run ONNX inference on feature vector (stub implementation)
    fn run_inference(&self, features: &[f32]) -> Result<Vec<f32>, ClassifierError> {
        // TODO: Implement proper ONNX inference once ort API is stable
        // This is a stub implementation for demonstration
        
        if !self.model_loaded {
            return Err(ClassifierError::Inference("Model not loaded".to_string()));
        }
        
        // Stub: Return mock probabilities based on feature analysis
        let feature_sum: f32 = features.iter().sum();
        let feature_avg = if !features.is_empty() {
            feature_sum / features.len() as f32
        } else {
            0.0
        };
        
        // Mock classification: higher feature average suggests higher risk
        let high_risk_prob = if feature_avg > 0.1 {
            0.8 // High confidence high-risk
        } else if feature_avg > 0.05 {
            0.6 // Medium confidence high-risk  
        } else {
            0.2 // Low-risk
        };
        
        let low_risk_prob = 1.0 - high_risk_prob;
        
        Ok(vec![low_risk_prob, high_risk_prob])
    }
    
    /// Apply aggregation rules to determine final classification
    fn aggregate_classification(&self, high_risk_count: usize, percentage_high_risk: f32) -> String {
        if self.config.use_weighted_scoring {
            // TODO: Implement weighted scoring logic
            "neutral".to_string()
        } else {
            // Use count and percentage thresholds
            if high_risk_count > self.config.count_threshold ||
               (percentage_high_risk / 100.0) > self.config.percentage_threshold {
                "high-risk".to_string()
            } else {
                "neutral".to_string()
            }
        }
    }
    
    /// Update classifier configuration
    pub fn set_config(&mut self, config: ClassifierConfig) {
        self.config = config;
    }
    
    /// Get current configuration
    pub fn get_config(&self) -> &ClassifierConfig {
        &self.config
    }
    
    /// Get model information (stub implementation)
    pub fn get_model_info(&self) -> ModelInfo {
        ModelInfo {
            input_shape: vec![None, Some(self.vectorizer_config.feature_names.len() as u32)],
            output_shapes: vec![vec![None, Some(2u32)]], // Binary classification
            vocabulary_size: self.vectorizer_config.vocabulary.len(),
            feature_count: self.vectorizer_config.feature_names.len(),
            ngram_range: self.vectorizer_config.ngram_range,
        }
    }
}

/// Information about the loaded model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub input_shape: Vec<Option<u32>>,
    pub output_shapes: Vec<Vec<Option<u32>>>,
    pub vocabulary_size: usize,
    pub feature_count: usize,
    pub ngram_range: (usize, usize),
}

/// Builder pattern for creating classifiers with custom configurations
pub struct ClassifierBuilder {
    config: ClassifierConfig,
}

impl ClassifierBuilder {
    pub fn new() -> Self {
        Self {
            config: ClassifierConfig::default(),
        }
    }
    
    pub fn confidence_threshold(mut self, threshold: f32) -> Self {
        self.config.confidence_threshold = threshold;
        self
    }
    
    pub fn count_threshold(mut self, threshold: usize) -> Self {
        self.config.count_threshold = threshold;
        self
    }
    
    pub fn percentage_threshold(mut self, threshold: f32) -> Self {
        self.config.percentage_threshold = threshold;
        self
    }
    
    pub fn use_weighted_scoring(mut self, use_weighted: bool) -> Self {
        self.config.use_weighted_scoring = use_weighted;
        self
    }
    
    pub fn build_from_bytes(
        self,
        onnx_bytes: &[u8],
        vectorizer_config: VectorizerConfig,
    ) -> Result<Classifier, ClassifierError> {
        Classifier::from_bytes(onnx_bytes, vectorizer_config, self.config)
    }
    
    #[cfg(not(feature = "wasm"))]
    pub fn build_from_files(
        self,
        onnx_path: &str,
        vectorizer_config_path: &str,
    ) -> Result<Classifier, ClassifierError> {
        Classifier::from_files(onnx_path, vectorizer_config_path, self.config)
    }
}

impl Default for ClassifierBuilder {
    fn default() -> Self {
        Self::new()
    }
}