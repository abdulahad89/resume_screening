import numpy as np
import requests
import streamlit as st
import re
import hashlib
import json
import os
import time
from typing import Dict, List, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HuggingFaceEmbeddingManager:
    """Embedding manager using Hugging Face Inference API"""
    
    def __init__(self):
        from config import (
            HUGGINGFACE_API_URL, HUGGINGFACE_TOKEN, EMBEDDING_MODEL,
            HF_TIMEOUT, MAX_RETRIES, CACHE_EMBEDDINGS, SKILL_CATEGORIES
        )
        
        self.api_url = HUGGINGFACE_API_URL
        self.token = HUGGINGFACE_TOKEN
        self.model_name = EMBEDDING_MODEL
        self.timeout = HF_TIMEOUT
        self.max_retries = MAX_RETRIES
        self.cache_enabled = CACHE_EMBEDDINGS
        self.skill_categories = SKILL_CATEGORIES
        
        # Cache directory for embeddings
        self.cache_dir = "./data/embeddings_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # API headers
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        # Test API connection on initialization (with better error handling)
        self.api_status = self._test_api_connection()
    
    def _test_api_connection(self) -> Dict[str, Any]:
        """Test Hugging Face API connection and model availability"""
        status = {
            "connected": False,
            "model_available": False,
            "error": None,
            "token_valid": False
        }
        
        if not self.token:
            status["error"] = "No Hugging Face token provided"
            return status
        
        if len(self.token) < 10:
            status["error"] = "Invalid token format"
            return status
        
        try:
            # Use a simpler API endpoint for testing
            test_url = f"{self.api_url}/models/{self.model_name}"
            
            # Make a simple GET request first (lighter than POST)
            response = requests.get(test_url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                status["connected"] = True
                status["token_valid"] = True
                status["model_available"] = True
            elif response.status_code == 401:
                status["error"] = "Invalid Hugging Face token"
            elif response.status_code == 403:
                status["error"] = "Token doesn't have required permissions"
            elif response.status_code == 404:
                status["error"] = f"Model {self.model_name} not found"
            elif response.status_code == 503:
                # Model is loading - this is actually OK
                status["connected"] = True
                status["token_valid"] = True
                status["model_available"] = True
                status["error"] = "Model is loading (this is normal)"
            else:
                status["error"] = f"API returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            status["error"] = "No internet connection or API unreachable"
        except requests.exceptions.Timeout:
            status["error"] = "API request timed out"
        except Exception as e:
            status["error"] = f"Connection test failed: {str(e)}"
        
        return status
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get current API status"""
        return self.api_status
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from cache"""
        if not self.cache_enabled:
            return None
        
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
        if os.path.exists(cache_file):
            try:
                return np.load(cache_file)
            except:
                # Remove corrupted cache file
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to cache"""
        if not self.cache_enabled:
            return
        
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
            np.save(cache_file, embedding)
        except:
            pass  # Ignore cache save errors
    
        def _call_hf_api(self, text: str, retry_count: int = 0) -> Optional[np.ndarray]:
        """Call Hugging Face Inference API with correct format"""
        
        # Skip if we know the API isn't working
        if not self.api_status.get("token_valid", False):
            raise Exception("Invalid or missing Hugging Face token")
        
        try:
            url = f"{self.api_url}/models/{self.model_name}"
            
            # FIXED: Use correct format for sentence-transformers models
            # Send as array of sentences, not single input
            payload = {
                "inputs": [text],  # Array format required for sentence-transformers
                "options": {
                    "wait_for_model": True,
                    "use_cache": True
                }
            }
            
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                # Parse response
                try:
                    result = response.json()
                    
                    # Handle sentence-transformers response format
                    if isinstance(result, list) and len(result) > 0:
                        # Result should be array of embeddings
                        if isinstance(result[0], list):
                            # Take first embedding (since we sent one sentence)
                            embedding = np.array(result[0])
                        else:
                            embedding = np.array(result)
                        return embedding
                    else:
                        st.error(f"Unexpected API response format: {type(result)}")
                        return None
                        
                except Exception as e:
                    st.error(f"Error parsing API response: {e}")
                    return None
            
            elif response.status_code == 503 and retry_count < self.max_retries:
                # Model is loading, retry after delay
                wait_time = min(10 + retry_count * 5, 30)
                st.info(f"Model loading... retrying in {wait_time}s")
                time.sleep(wait_time)
                return self._call_hf_api(text, retry_count + 1)
            
            elif response.status_code == 429 and retry_count < self.max_retries:
                # Rate limit, retry with backoff
                wait_time = min(5 + retry_count * 2, 15)
                time.sleep(wait_time)
                return self._call_hf_api(text, retry_count + 1)
            
            else:
                error_msg = f"HF API Error {response.status_code}"
                try:
                    error_detail = response.json()
                    if 'error' in error_detail:
                        error_msg += f": {error_detail['error']}"
                except:
                    error_msg += f": {response.text[:200]}"
                
                if retry_count == 0:  # Only log on first attempt
                    st.error(error_msg)
                return None
                
        except requests.exceptions.Timeout:
            if retry_count < self.max_retries:
                return self._call_hf_api(text, retry_count + 1)
            else:
                raise Exception("API timeout after retries")
        except requests.exceptions.ConnectionError:
            raise Exception("No internet connection")
        except Exception as e:
            if retry_count < self.max_retries:
                time.sleep(2)
                return self._call_hf_api(text, retry_count + 1)
            else:
                raise e
    
    def preprocess_text_advanced(self, text: str) -> str:
        """Advanced text preprocessing for better embeddings"""
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize technology terms
        tech_normalizations = {
            r'\bjavascript\b': 'JavaScript',
            r'\bnodejs\b|node\.js': 'Node.js',
            r'\breactjs\b': 'React',
            r'\bangularjs\b': 'Angular',
            r'\bml\b': 'machine learning',
            r'\bai\b': 'artificial intelligence',
            r'\baws\b': 'Amazon Web Services',
            r'\bgcp\b': 'Google Cloud Platform',
        }
        
        text_lower = text.lower()
        for pattern, replacement in tech_normalizations.items():
            text_lower = re.sub(pattern, replacement.lower(), text_lower, flags=re.IGNORECASE)
        
        # Keep important short terms
        important_terms = {'r', 'c', 'go', 'ai', 'ml', 'bi', 'ui', 'ux', 'qa'}
        words = text_lower.split()
        words = [w for w in words if len(w) >= 2 or w in important_terms]
        
        return ' '.join(words)
    
    def extract_enhanced_keywords(self, text: str) -> Dict[str, List[str]]:
        """Extract keywords using comprehensive skill categories"""
        text_lower = text.lower()
        found_keywords = {}
        
        for category, keywords in self.skill_categories.items():
            found = []
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    found.append(keyword)
            if found:
                found_keywords[category] = found
        
        return found_keywords
    
    def encode_text(self, text: str) -> np.ndarray:
        """Generate embedding for text using HF API"""
        
        # Preprocess text
        clean_text = self.preprocess_text_advanced(text)
        
        # Truncate if too long (HF has input limits)
        if len(clean_text) > 500:
            clean_text = clean_text[:500]
        
        # Try cache first
        cache_key = self._get_cache_key(clean_text)
        cached_embedding = self._load_from_cache(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        # Call HF API
        try:
            with st.spinner("🤖 Getting embedding from Hugging Face..."):
                embedding = self._call_hf_api(clean_text)
            
            if embedding is not None:
                # Ensure proper shape and normalize
                if embedding.ndim == 1:
                    embedding = embedding / np.linalg.norm(embedding)
                else:
                    embedding = embedding.flatten()
                    embedding = embedding / np.linalg.norm(embedding)
                
                # Save to cache
                self._save_to_cache(cache_key, embedding)
                
                return embedding
            else:
                # Fallback to simple text features
                st.warning("Using fallback embedding due to API error")
                return self._create_fallback_embedding(clean_text)
                
        except Exception as e:
            st.error(f"Embedding API error: {str(e)}")
            return self._create_fallback_embedding(clean_text)
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts (one by one due to HF API limitations)"""
        embeddings = []
        
        for i, text in enumerate(texts):
            if i > 0 and i % 5 == 0:  # Rate limiting
                time.sleep(1)
            
            embedding = self.encode_text(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _create_fallback_embedding(self, text: str) -> np.ndarray:
        """Create simple fallback embedding when API fails"""
        # Create basic text features
        features = []
        
        # Text statistics
        features.append(len(text.split()))  # Word count
        features.append(len(set(text.lower().split())))  # Unique words
        features.append(text.count('.'))  # Sentences
        
        # Technology keywords
        tech_words = ['python', 'java', 'javascript', 'react', 'sql', 'aws', 'docker']
        for word in tech_words:
            features.append(1 if word in text.lower() else 0)
        
        # Experience indicators
        exp_words = ['years', 'experience', 'senior', 'lead', 'manager']
        for word in exp_words:
            features.append(1 if word in text.lower() else 0)
        
        # Pad to standard embedding size (384 for compatibility)
        while len(features) < 384:
            features.append(0.0)
        
        embedding = np.array(features[:384], dtype=np.float32)
        return embedding / (np.linalg.norm(embedding) + 1e-8)  # Normalize with epsilon
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using HF embeddings"""
        try:
            # Get embeddings
            embedding1 = self.encode_text(text1)
            embedding2 = self.encode_text(text2)
            
            # Ensure same shape
            if embedding1.shape != embedding2.shape:
                min_len = min(len(embedding1), len(embedding2))
                embedding1 = embedding1[:min_len]
                embedding2 = embedding2[:min_len]
            
            # Calculate cosine similarity
            similarity = float(cosine_similarity([embedding1], [embedding2])[0][0])
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
        except Exception as e:
            st.error(f"Similarity calculation error: {e}")
            return 0.0
    
    def calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF similarity"""
        try:
            clean_text1 = self.preprocess_text_advanced(text1)
            clean_text2 = self.preprocess_text_advanced(text2)
            
            tfidf_matrix = self.tfidf.fit_transform([clean_text1, clean_text2])
            similarity = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
            return max(0.0, similarity)
            
        except Exception as e:
            return 0.0
    
    def calculate_keyword_similarity(self, jd_keywords: Dict, resume_keywords: Dict) -> Dict:
        """Calculate keyword similarity by category"""
        category_scores = {}
        overall_scores = []
        
        for category in jd_keywords:
            jd_cat_keywords = set(jd_keywords[category])
            resume_cat_keywords = set(resume_keywords.get(category, []))
            
            if jd_cat_keywords:
                overlap = jd_cat_keywords.intersection(resume_cat_keywords)
                missing = jd_cat_keywords - resume_cat_keywords
                category_score = len(overlap) / len(jd_cat_keywords)
                
                category_scores[category] = {
                    'score': category_score,
                    'matched': list(overlap),
                    'missing': list(missing)
                }
                overall_scores.append(category_score)
        
        overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0
        
        return {
            'overall_score': overall_score,
            'category_breakdown': category_scores
        }
    
    def hybrid_similarity_analysis(self, jd_text: str, resume_text: str) -> Dict:
        """Comprehensive similarity analysis using HF embeddings"""
        
        # Extract keywords
        jd_keywords = self.extract_enhanced_keywords(jd_text)
        resume_keywords = self.extract_enhanced_keywords(resume_text)
        
        # Calculate different similarity metrics
        semantic_score = self.calculate_semantic_similarity(jd_text, resume_text)
        tfidf_score = self.calculate_tfidf_similarity(jd_text, resume_text)
        keyword_analysis = self.calculate_keyword_similarity(jd_keywords, resume_keywords)
        
        # Combined score
        from config import SCORING_WEIGHTS
        combined_score = (
            semantic_score * SCORING_WEIGHTS['semantic'] +
            keyword_analysis['overall_score'] * SCORING_WEIGHTS['keyword_categories'] +
            tfidf_score * SCORING_WEIGHTS['tfidf']
        )
        
        return {
            'combined': combined_score,
            'semantic': semantic_score,
            'keyword_categories': keyword_analysis['overall_score'],
            'tfidf': tfidf_score,
            'category_breakdown': keyword_analysis['category_breakdown'],
            'jd_keywords': jd_keywords,
            'resume_keywords': resume_keywords,
            'model_used': self.model_name,
            'api_provider': 'Hugging Face Inference API'
        }
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'name': self.model_name.split('/')[-1],
            'full_name': self.model_name,
            'provider': 'Hugging Face',
            'type': 'Remote Inference API',
            'status': self.api_status,
            'cache_enabled': self.cache_enabled
        }
    
    def clear_cache(self):
        """Clear embedding cache"""
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
            st.success("Embedding cache cleared")
        except Exception as e:
            st.error(f"Failed to clear cache: {e}")
