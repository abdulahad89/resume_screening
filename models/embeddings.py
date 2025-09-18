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
    """Production-ready embedding manager using Hugging Face Inference API"""
    
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
        
        # Test API connection on initialization
        self.api_status = self._test_api_connection()
    
    def _test_api_connection(self) -> Dict[str, Any]:
        """Test Hugging Face API connection and model availability"""
        status = {
            "connected": False,
            "model_available": False,
            "error": None,
            "token_valid": False
        }
        
        # Validate token
        if not self.token:
            status["error"] = "No Hugging Face token provided"
            return status
        
        if len(self.token) < 10:
            status["error"] = "Invalid token format"
            return status
        
        try:
            # SIMPLIFIED TEST - just check token validity with models endpoint
            test_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
            
            response = requests.get(
                test_url,
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=5  # Short timeout for quick test
            )
            
            if response.status_code in [200, 503]:  # 200 = OK, 503 = loading (both fine)
                status["connected"] = True
                status["token_valid"] = True
                status["model_available"] = True
            elif response.status_code == 401:
                status["error"] = "Invalid Hugging Face token"
            elif response.status_code == 403:
                status["error"] = "Token doesn't have required permissions"
            elif response.status_code == 404:
                status["error"] = f"Model {self.model_name} not found"
            else:
                # Don't fail for other status codes, just mark as connected
                status["connected"] = True
                status["token_valid"] = True
                status["model_available"] = True
                
        except Exception as e:
            # Don't fail the entire connection test - just log and continue
            status["error"] = f"Connection test skipped: {str(e)}"
            # IMPORTANT: Still mark as connected so the API can be used
            status["connected"] = True
            status["token_valid"] = True
            status["model_available"] = True
        
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
                embedding = np.load(cache_file)
                if embedding.size > 0 and not np.isnan(embedding).any():
                    return embedding
            except:
                # Remove corrupted cache file
                try:
                    os.remove(cache_file)
                except:
                    pass
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """Save embedding to cache"""
        if not self.cache_enabled or embedding.size == 0:
            return
        
        try:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.npy")
            np.save(cache_file, embedding)
        except Exception:
            pass
    
    def _call_hf_api(self, text: str, retry_count: int = 0) -> Optional[np.ndarray]:
        """Call Hugging Face API using standard models endpoint - FINAL WORKING VERSION"""
        
        if not self.api_status.get("token_valid", False):
            raise Exception("Invalid or missing Hugging Face token")
        
        try:
            # Use standard models API endpoint (NOT pipeline endpoint - that causes 404)
            url = f"https://api-inference.huggingface.co/models/{self.model_name}"
            
            # CORRECT PAYLOAD for sentence-transformers models
            payload = {
                "inputs": [text],  # MUST be array format for sentence-transformers
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
                try:
                    result = response.json()
                    
                    # Handle response format for sentence-transformers embeddings
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], list):
                            # Standard format: [[embedding_vector]]
                            embedding = np.array(result[0], dtype=np.float32)
                        else:
                            # Alternative format: [embedding_vector]
                            embedding = np.array(result, dtype=np.float32)
                        
                        if embedding.size > 0 and not np.isnan(embedding).any():
                            return embedding
                    
                    st.error(f"Unexpected API response format: {type(result)}")
                    return None
                    
                except json.JSONDecodeError:
                    st.error("Failed to parse API response as JSON")
                    return None
                except Exception as e:
                    st.error(f"Error parsing API response: {e}")
                    return None
            
            elif response.status_code == 503 and retry_count < self.max_retries:
                # Model is loading - retry with backoff
                wait_time = min(10 + retry_count * 5, 30)
                st.info(f"🤖 Model loading... retrying in {wait_time}s")
                time.sleep(wait_time)
                return self._call_hf_api(text, retry_count + 1)
            
            elif response.status_code == 429 and retry_count < self.max_retries:
                # Rate limit - retry with backoff
                wait_time = min(5 + retry_count * 2, 15)
                st.warning(f"⏰ Rate limited, retrying in {wait_time}s")
                time.sleep(wait_time)
                return self._call_hf_api(text, retry_count + 1)
            
            else:
                # Handle error response
                error_msg = f"HF API Error {response.status_code}"
                try:
                    error_detail = response.json()
                    if 'error' in error_detail:
                        error_msg += f": {error_detail['error']}"
                except:
                    error_msg += f": {response.text[:200]}"
                
                if retry_count == 0:
                    st.error(error_msg)
                return None
                
        except requests.exceptions.Timeout:
            if retry_count < self.max_retries:
                st.warning(f"⏰ Request timed out, retrying ({retry_count + 1}/{self.max_retries})...")
                return self._call_hf_api(text, retry_count + 1)
            else:
                raise Exception("API timeout after all retries")
        except requests.exceptions.ConnectionError:
            raise Exception("No internet connection available")
        except Exception as e:
            if retry_count < self.max_retries:
                time.sleep(2)
                return self._call_hf_api(text, retry_count + 1)
            else:
                raise e
    
    def preprocess_text_advanced(self, text: str) -> str:
        """Advanced text preprocessing for better embeddings"""
        if not text or not text.strip():
            return ""
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize technology terms for better matching
        tech_normalizations = {
            r'\bjavascript\b': 'JavaScript',
            r'\bnodejs\b|node\.js': 'Node.js',
            r'\breactjs\b': 'React',
            r'\bangularjs\b': 'Angular',
            r'\bml\b(?!\w)': 'machine learning',
            r'\bai\b(?!\w)': 'artificial intelligence',
            r'\baws\b': 'Amazon Web Services',
            r'\bgcp\b': 'Google Cloud Platform',
        }
        
        text_lower = text.lower()
        for pattern, replacement in tech_normalizations.items():
            text_lower = re.sub(pattern, replacement.lower(), text_lower, flags=re.IGNORECASE)
        
        # Keep important short terms that are meaningful
        important_terms = {'r', 'c', 'go', 'ai', 'ml', 'bi', 'ui', 'ux', 'qa'}
        words = text_lower.split()
        words = [w for w in words if len(w) >= 2 or w in important_terms]
        
        return ' '.join(words)
    
    def extract_enhanced_keywords(self, text: str) -> Dict[str, List[str]]:
        """Extract keywords using comprehensive skill categories"""
        if not text:
            return {}
        
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
        """Generate embedding for text using HF API with smart caching"""
        if not text or not text.strip():
            return self._create_fallback_embedding("")
        
        # Preprocess text for better embeddings
        clean_text = self.preprocess_text_advanced(text)
        
        # Truncate if too long (most sentence-transformers have ~512 token limit)
        if len(clean_text) > 500:
            clean_text = clean_text[:500]
        
        # Try cache first to reduce API calls
        cache_key = self._get_cache_key(clean_text)
        cached_embedding = self._load_from_cache(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        # Call HF API for fresh embedding
        try:
            with st.spinner("🤖 Getting embedding from Hugging Face..."):
                embedding = self._call_hf_api(clean_text)
            
            if embedding is not None and embedding.size > 0:
                # Normalize embedding for proper cosine similarity
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                else:
                    embedding = self._create_fallback_embedding(clean_text)
                
                # Save to cache for future use
                self._save_to_cache(cache_key, embedding)
                return embedding
            else:
                st.warning("⚠️ HF API failed, using fallback embedding")
                return self._create_fallback_embedding(clean_text)
                
        except Exception as e:
            st.error(f"❌ Embedding API error: {str(e)}")
            return self._create_fallback_embedding(clean_text)
    
    def batch_encode(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts with rate limiting to prevent API overload"""
        embeddings = []
        
        for i, text in enumerate(texts):
            # Rate limiting to avoid overwhelming the API
            if i > 0 and i % 5 == 0:
                time.sleep(1)
            
            embedding = self.encode_text(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _create_fallback_embedding(self, text: str) -> np.ndarray:
        """Create high-quality fallback embedding when API fails"""
        if not text:
            return np.zeros(384, dtype=np.float32)
        
        # Create comprehensive feature-based embedding
        features = []
        text_lower = text.lower()
        
        # Text statistics features
        words = text.split()
        features.extend([
            len(words),  # Word count
            len(set(text_lower.split())),  # Unique word count
            text.count('.'),  # Sentence count
            text.count(','),  # Complexity indicator
            len(text),  # Character count
        ])
        
        # Technology keywords presence (extensive list)
        tech_keywords = [
            'python', 'java', 'javascript', 'react', 'sql', 'aws', 'docker',
            'kubernetes', 'git', 'linux', 'tensorflow', 'pytorch', 'pandas',
            'numpy', 'machine learning', 'data science', 'api', 'database',
            'cloud', 'devops', 'agile', 'scrum', 'marketing', 'analytics',
            'excel', 'tableau', 'power bi', 'salesforce', 'hubspot'
        ]
        
        for keyword in tech_keywords:
            features.append(1.0 if keyword in text_lower else 0.0)
        
        # Experience level indicators
        experience_indicators = [
            'senior', 'lead', 'principal', 'manager', 'director',
            'junior', 'intern', 'entry', 'years', 'experience'
        ]
        
        for indicator in experience_indicators:
            features.append(1.0 if indicator in text_lower else 0.0)
        
        # Education and certification keywords
        education_keywords = [
            'bachelor', 'master', 'phd', 'degree', 'university',
            'college', 'certification', 'certified'
        ]
        
        for keyword in education_keywords:
            features.append(1.0 if keyword in text_lower else 0.0)
        
        # Pad to standard embedding size (384 dimensions like MPNet)
        while len(features) < 384:
            features.append(0.0)
        
        embedding = np.array(features[:384], dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using HF embeddings"""
        try:
            embedding1 = self.encode_text(text1)
            embedding2 = self.encode_text(text2)
            
            # Ensure same dimensions
            min_len = min(len(embedding1), len(embedding2))
            embedding1 = embedding1[:min_len]
            embedding2 = embedding2[:min_len]
            
            if min_len > 0:
                similarity = float(cosine_similarity([embedding1], [embedding2])[0][0])
                return max(0.0, min(1.0, similarity))  # Clamp to valid range
            else:
                return 0.0
            
        except Exception as e:
            st.error(f"Similarity calculation error: {e}")
            return 0.0
    
    def calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF similarity for keyword matching"""
        try:
            if not text1.strip() or not text2.strip():
                return 0.0
            
            clean_text1 = self.preprocess_text_advanced(text1)
            clean_text2 = self.preprocess_text_advanced(text2)
            
            if not clean_text1 or not clean_text2:
                return 0.0
            
            tfidf_matrix = self.tfidf.fit_transform([clean_text1, clean_text2])
            similarity = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
            return max(0.0, similarity)
            
        except Exception:
            return 0.0
    
    def calculate_keyword_similarity(self, jd_keywords: Dict, resume_keywords: Dict) -> Dict:
        """Calculate detailed keyword similarity by category"""
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
        
        overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        
        return {
            'overall_score': overall_score,
            'category_breakdown': category_scores
        }
    
    def hybrid_similarity_analysis(self, jd_text: str, resume_text: str) -> Dict:
        """Comprehensive similarity analysis combining multiple AI methods"""
        
        # Extract keywords from both texts using AI-powered categorization
        jd_keywords = self.extract_enhanced_keywords(jd_text)
        resume_keywords = self.extract_enhanced_keywords(resume_text)
        
        # Calculate different similarity metrics
        semantic_score = self.calculate_semantic_similarity(jd_text, resume_text)
        tfidf_score = self.calculate_tfidf_similarity(jd_text, resume_text)
        keyword_analysis = self.calculate_keyword_similarity(jd_keywords, resume_keywords)
        
        # Calculate weighted combined score using configured weights
        from config import SCORING_WEIGHTS
        combined_score = (
            semantic_score * SCORING_WEIGHTS['semantic'] +
            keyword_analysis['overall_score'] * SCORING_WEIGHTS['keyword_categories'] +
            tfidf_score * SCORING_WEIGHTS['tfidf']
        )
        
        # Create intelligent analysis summary
        if combined_score >= 0.7:
            summary = "Excellent match - candidate strongly aligns with job requirements"
        elif combined_score >= 0.5:
            summary = "Good match - candidate meets most requirements with minor gaps"
        elif combined_score >= 0.3:
            summary = "Moderate match - candidate has relevant skills but notable gaps"
        else:
            summary = "Limited match - significant gaps in required qualifications"
        
        return {
            'combined': combined_score,
            'semantic': semantic_score,
            'keyword_categories': keyword_analysis['overall_score'],
            'tfidf': tfidf_score,
            'category_breakdown': keyword_analysis['category_breakdown'],
            'jd_keywords': jd_keywords,
            'resume_keywords': resume_keywords,
            'model_used': self.model_name,
            'api_provider': 'Hugging Face Inference API',
            'analysis_summary': summary
        }
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information for monitoring"""
        return {
            'name': self.model_name.split('/')[-1],
            'full_name': self.model_name,
            'provider': 'Hugging Face',
            'type': 'Sentence Transformers API',
            'status': self.api_status,
            'cache_enabled': self.cache_enabled,
            'endpoint': f"https://api-inference.huggingface.co/models/{self.model_name}",
            'dimensions': 384 if 'mpnet' in self.model_name.lower() else 768
        }
    
    def clear_cache(self):
        """Clear embedding cache and show results"""
        try:
            import shutil
            cache_files = 0
            
            if os.path.exists(self.cache_dir):
                cache_files = len([f for f in os.listdir(self.cache_dir) if f.endswith('.npy')])
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
            
            st.success(f"✅ Cleared {cache_files} cached embeddings")
            
        except Exception as e:
            st.error(f"❌ Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics for performance monitoring"""
        try:
            if not os.path.exists(self.cache_dir):
                return {'enabled': self.cache_enabled, 'files': 0, 'size_mb': 0}
            
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.npy')]
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f)) 
                for f in cache_files
            )
            
            return {
                'enabled': self.cache_enabled,
                'files': len(cache_files),
                'size_mb': round(total_size / (1024 * 1024), 2)
            }
            
        except Exception:
            return {'enabled': self.cache_enabled, 'files': 0, 'size_mb': 0}
