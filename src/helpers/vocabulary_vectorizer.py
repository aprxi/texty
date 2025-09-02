#!/usr/bin/env python3
"""Shared VocabularyVectorizer for vocabulary-based models."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer


class VocabularyVectorizer(BaseEstimator, TransformerMixin):
    """Custom vectorizer that treats vocabulary terms as complete phrase features."""
    
    def __init__(self):
        self.vocabulary_ = set()
        self.dict_vectorizer = DictVectorizer(sparse=True)
    
    def fit(self, X, y=None):
        # Build vocabulary from all training texts (each text is a vocabulary term)
        self.vocabulary_ = set(text.lower().strip() for text in X if text.strip())
        
        # Create feature dictionaries for fitting the DictVectorizer
        feature_dicts = []
        for text in X:
            features = {}
            text_lower = text.lower().strip()
            if text_lower in self.vocabulary_:
                features[f"vocab_term_{text_lower}"] = 1.0
            feature_dicts.append(features)
        
        self.dict_vectorizer.fit(feature_dicts)
        return self
    
    def transform(self, X):
        # Transform texts to feature vectors based on vocabulary matches
        feature_dicts = []
        for text in X:
            features = {}
            text_lower = text.lower().strip()
            
            # Exact match for vocabulary term
            if text_lower in self.vocabulary_:
                features[f"vocab_term_{text_lower}"] = 1.0
            
            # Also check if any vocabulary term is contained in the text
            for vocab_term in self.vocabulary_:
                if vocab_term in text_lower and vocab_term != text_lower:
                    features[f"contains_{vocab_term}"] = 1.0
            
            feature_dicts.append(features)
        
        return self.dict_vectorizer.transform(feature_dicts)
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation. Compatible with sklearn interface."""
        return self.dict_vectorizer.get_feature_names_out(input_features)