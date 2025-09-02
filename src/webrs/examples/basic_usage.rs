//! Basic usage example for the Rust text classifier

use texty_webrs::{TextClassifier, ClassifierConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦀 Texty WebRS - Basic Usage Example");
    println!("====================================");
    
    // Example usage (once ONNX model is available)
    let config = ClassifierConfig {
        confidence_threshold: 0.7,
        count_threshold: 1,
        percentage_threshold: 10.0,
        use_weighted_scoring: false,
    };
    
    println!("📋 Configuration: {:#?}", config);
    
    // This would work once we have the ONNX model exported:
    /*
    let mut classifier = TextClassifier::new();
    
    // Load ONNX model and vectorizer config
    let onnx_bytes = std::fs::read("./models/base_model_classifier.onnx")?;
    let vectorizer_config_json = std::fs::read_to_string("./models/base_model_vectorizer.json")?;
    let vectorizer_config = serde_json::from_str(&vectorizer_config_json)?;
    
    classifier.load_model(&onnx_bytes, vectorizer_config)?;
    classifier.set_config(config);
    
    // Test text
    let test_text = "We use facial recognition technology to automatically identify employees and track their productivity throughout the workday.";
    
    println!("\n📝 Test Text: {}", test_text);
    
    let result = classifier.classify_text(test_text)?;
    
    println!("\n📊 Classification Results:");
    println!("   Classification: {}", result.classification);
    println!("   Confidence: {:.3}", result.confidence);
    println!("   High-risk sentences: {}/{}", result.high_risk_sentences, result.total_sentences);
    println!("   Risk percentage: {:.1}%", result.percentage_high_risk);
    println!("   Processing time: {:.2}ms", result.processing_time_ms);
    
    if !result.sentence_details.is_empty() {
        println!("\n🔍 Sentence Details:");
        for (i, sentence) in result.sentence_details.iter().enumerate() {
            println!("   [{}] {} - {} ({:.3})", 
                i + 1, 
                sentence.prediction,
                sentence.sentence,
                sentence.confidence
            );
        }
    }
    */
    
    println!("\n💡 To run this example:");
    println!("   1. First export the Python model: make webrs-export");
    println!("   2. Then rebuild: make webrs-build");
    println!("   3. Run: cargo run --example basic_usage");
    
    Ok(())
}