import os
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import UPLOAD_FOLDER, MAX_FILE_SIZE
import re

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file and return path"""
    try:
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"File too large: {uploaded_file.size / (1024*1024):.1f}MB (max: {MAX_FILE_SIZE/(1024*1024):.1f}MB)")
            return None
        
        # Create upload directory
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Save file
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    
    except Exception as e:
        st.error(f"Failed to save file: {e}")
        return None

def calculate_enhanced_similarity(jd_text: str, resume_text: str, embedding_manager) -> dict:
    """Calculate enhanced similarity using HuggingFace API or fallback methods"""
    try:
        # Check if HF API is available
        api_status = embedding_manager.get_api_status()
        
        if api_status["model_available"]:
            # Use HF API hybrid similarity analysis
            similarity_results = embedding_manager.hybrid_similarity_analysis(jd_text, resume_text)
            return similarity_results
        else:
            # Use fallback methods
            return _fallback_similarity_analysis(jd_text, resume_text, embedding_manager)
            
    except Exception as e:
        st.error(f"Enhanced similarity calculation error: {e}")
        return _fallback_similarity_analysis(jd_text, resume_text, embedding_manager)

def _fallback_similarity_analysis(jd_text: str, resume_text: str, embedding_manager) -> dict:
    """Fallback similarity analysis when HF API is unavailable"""
    try:
        # Extract keywords
        jd_keywords = embedding_manager.extract_enhanced_keywords(jd_text)
        resume_keywords = embedding_manager.extract_enhanced_keywords(resume_text)
        
        # Calculate keyword similarity
        keyword_analysis = embedding_manager.calculate_keyword_similarity(jd_keywords, resume_keywords)
        
        # Calculate TF-IDF similarity
        tfidf_score = embedding_manager.calculate_tfidf_similarity(jd_text, resume_text)
        
        # Combined score (no semantic similarity available in fallback)
        combined_score = keyword_analysis['overall_score'] * 0.7 + tfidf_score * 0.3
        
        return {
            'combined': combined_score,
            'semantic': 0.0,  # Not available in fallback
            'keyword_categories': keyword_analysis['overall_score'],
            'tfidf': tfidf_score,
            'category_breakdown': keyword_analysis['category_breakdown'],
            'jd_keywords': jd_keywords,
            'resume_keywords': resume_keywords,
            'model_used': 'fallback',
            'api_provider': 'Fallback Analysis (No HF Token)',
            'analysis_summary': 'Using keyword and TF-IDF analysis only'
        }
        
    except Exception as e:
        return {
            'combined': 0.0,
            'semantic': 0.0,
            'keyword_categories': 0.0,
            'tfidf': 0.0,
            'category_breakdown': {},
            'jd_keywords': {},
            'resume_keywords': {},
            'model_used': 'error',
            'api_provider': 'Error',
            'analysis_summary': f'Error in similarity calculation: {str(e)}'
        }

def calculate_similarity_score(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings (backward compatibility)"""
    try:
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        return float(max(0.0, similarity))
    except Exception as e:
        st.error(f"Similarity calculation error: {e}")
        return 0.0

def get_scoring_color(score: float) -> str:
    """Get color emoji for score display"""
    if score >= 0.7:
        return "🟢"  # Green for high scores
    elif score >= 0.5:
        return "🟡"  # Yellow for medium scores  
    elif score >= 0.3:
        return "🟠"  # Orange for low-medium scores
    else:
        return "🔴"  # Red for low scores

def format_score_display(score: float, include_color: bool = True) -> str:
    """Format score for display with color and percentage"""
    color = get_scoring_color(score) if include_color else ""
    return f"{color} {score:.1%}".strip()

def create_detailed_score_explanation(similarity_results: dict, include_hf_info: bool = True) -> str:
    """Create detailed explanation of the scoring methodology"""
    
    if not similarity_results or 'combined' not in similarity_results:
        return "❌ No scoring data available"
    
    provider = similarity_results.get('api_provider', 'Unknown')
    model_used = similarity_results.get('model_used', 'Unknown')
    
    explanation = f"""
**🎯 Overall Relevance Score: {similarity_results['combined']:.1%}**

**📊 Analysis Method**: {provider}
**🤖 Model**: {model_used}

**Score Breakdown:**
"""
    
    # Different explanations based on provider
    if 'Hugging Face' in provider:
        explanation += f"""
• **Semantic Similarity**: {similarity_results.get('semantic', 0):.1%} 
  ↳ *Advanced AI understanding of content meaning via Hugging Face*
• **Skill Categories**: {similarity_results.get('keyword_categories', 0):.1%}
  ↳ *Technical and professional skill alignment*  
• **Keyword Matching**: {similarity_results.get('tfidf', 0):.1%}
  ↳ *Important term frequency and relevance*
"""
    else:
        explanation += f"""
• **Skill Categories**: {similarity_results.get('keyword_categories', 0):.1%}
  ↳ *Technical and professional skill alignment*  
• **Keyword Matching**: {similarity_results.get('tfidf', 0):.1%}
  ↳ *Important term frequency and relevance*
• **Semantic Analysis**: Not available (requires HF token)
"""
    
    # Add category breakdown if available
    if 'category_breakdown' in similarity_results and similarity_results['category_breakdown']:
        explanation += "\n\n**🔍 Skills Analysis by Category:**\n"
        breakdown = similarity_results['category_breakdown']
        
        for category, details in list(breakdown.items())[:5]:  # Show top 5 categories
            if details.get('score', 0) > 0:
                category_name = category.replace('_', ' ').title()
                matched = details.get('matched', [])
                missing = details.get('missing', [])
                
                explanation += f"\n• **{category_name}**: {details.get('score', 0):.1%}"
                if matched:
                    explanation += f"\n  ✅ Found: {', '.join(matched[:3])}"
                if missing:
                    explanation += f"\n  ❌ Missing: {', '.join(missing[:3])}"
    
    # Add setup recommendation if using fallback
    if 'Fallback' in provider or 'fallback' in model_used.lower():
        explanation += f"""

💡 **Improve Accuracy**: Get free Hugging Face token for semantic analysis:
1. Visit huggingface.co and create free account
2. Go to Settings → Access Tokens  
3. Create new token and add to .env file
4. Restart app for better AI-powered matching
"""
    
    return explanation

def get_hf_setup_instructions() -> dict:
    """Get Hugging Face setup instructions"""
    return {
        "title": "🔑 Set up Hugging Face API for Better Accuracy",
        "benefits": [
            "🎯 Much more accurate semantic matching",
            "🧠 Advanced AI understanding of job requirements", 
            "📈 Better candidate relevance scores",
            "🆓 Free tier available with generous limits"
        ],
        "steps": [
            "1. Go to [huggingface.co](https://huggingface.co) and create a free account",
            "2. Navigate to Settings → Access Tokens", 
            "3. Click 'New token' and create a token with 'Read' permissions",
            "4. Create a `.env` file in your project directory",
            "5. Add this line: `HUGGINGFACE_TOKEN=your_token_here`",
            "6. Restart the Streamlit app"
        ],
        "fallback_note": "The app works without a token using keyword analysis, but HF provides much better semantic understanding."
    }

def extract_key_resume_highlights(text: str, skills: dict) -> dict:
    """Extract key highlights from resume for quick review"""
    
    highlights = {
        'years_experience': 0,
        'education_level': 'Not specified',
        'top_skills': [],
        'certifications': [],
        'key_achievements': []
    }
    
    text_lower = text.lower()
    
    # Extract years of experience
    experience_patterns = [
        r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
        r'experience.*?(\d+)\+?\s*years?'
    ]
    
    years = []
    for pattern in experience_patterns:
        matches = re.findall(pattern, text_lower)
        years.extend([int(match) for match in matches if match.isdigit()])
    
    if years:
        highlights['years_experience'] = max(years)
    
    # Education level
    education_keywords = {
        'phd': ['phd', 'doctorate', 'doctoral'],
        'masters': ['master', 'msc', 'ma', 'mba', 'ms'], 
        'bachelors': ['bachelor', 'bsc', 'ba', 'bs'],
        'associates': ['associate', 'diploma']
    }
    
    for level, keywords in education_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            highlights['education_level'] = level.title()
            break
    
    # Top skills (from most populated categories)
    skill_counts = [(category, len(skill_list)) for category, skill_list in skills.items()]
    skill_counts.sort(key=lambda x: x[1], reverse=True)
    
    for category, count in skill_counts[:3]:
        if count > 0:
            category_name = category.replace('_', ' ').title()
            highlights['top_skills'].append(f"{category_name} ({count} skills)")
    
    # Certifications
    cert_patterns = [
        r'certified\s+([^,\n.]+)',
        r'(aws|azure|google|microsoft)\s+certified'
    ]
    
    for pattern in cert_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        highlights['certifications'].extend(matches[:3])
    
    # Key achievements (lines with percentages or improvements)
    achievement_patterns = [
        r'[^.]*\d+%[^.]*',
        r'[^.]*increased?[^.]*\d+[^.]*',
        r'[^.]*improved?[^.]*\d+[^.]*'
    ]
    
    achievements = []
    for pattern in achievement_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        achievements.extend([match.strip() for match in matches if len(match.strip()) < 200])
    
    highlights['key_achievements'] = achievements[:3]
    
    return highlights

def create_hf_status_display(api_status: dict) -> str:
    """Create formatted display of HF API status"""
    
    if api_status.get("model_available"):
        return """
✅ **Hugging Face API**: Connected and ready
🤖 **Model Status**: Available for inference  
🎯 **Quality**: High-accuracy semantic analysis enabled
"""
    elif api_status.get("token_valid") == False:
        return """
❌ **Hugging Face API**: Invalid or missing token
🔧 **Action Needed**: Set up your free HF token
📊 **Current Mode**: Fallback analysis (reduced accuracy)

**Quick Setup:**
1. Get free token at huggingface.co
2. Add to .env file: `HUGGINGFACE_TOKEN=your_token`
3. Restart app
"""
    elif "loading" in api_status.get("error", "").lower():
        return """
⏳ **Hugging Face API**: Model is loading
⏰ **Status**: Please wait a moment and try again
🔄 **Retry**: Model will be ready shortly
"""
    else:
        error = api_status.get("error", "Unknown error")
        return f"""
⚠️ **Hugging Face API**: Connection issue
❌ **Error**: {error}
📊 **Current Mode**: Fallback analysis
"""

def validate_file_type(filename: str) -> bool:
    """Check if file type is supported"""
    supported_extensions = ['.pdf', '.docx', '.doc', '.txt']
    file_extension = os.path.splitext(filename)[1].lower()
    return file_extension in supported_extensions

def format_skills_display(skills_dict: dict) -> str:
    """Format skills dictionary for display"""
    if not skills_dict:
        return "No specific skills detected"
    
    formatted = []
    for category, skills in skills_dict.items():
        if skills:
            category_name = category.replace('_', ' ').title()
            formatted.append(f"**{category_name}**: {', '.join(skills[:5])}")
    
    return '\n'.join(formatted) if formatted else "No specific skills detected"

def create_api_usage_summary(embedding_manager) -> dict:
    """Create summary of API usage and caching"""
    
    api_status = embedding_manager.get_api_status()
    cache_enabled = getattr(embedding_manager, 'cache_enabled', False)
    
    summary = {
        "provider": "Hugging Face Inference API",
        "model": getattr(embedding_manager, 'model_name', 'Unknown'),
        "status": "✅ Connected" if api_status.get("model_available") else "❌ Unavailable",
        "caching": "✅ Enabled" if cache_enabled else "❌ Disabled",
        "benefits": [
            "No local model storage (saves GB of space)",
            "Always latest model version",
            "Scalable cloud infrastructure",
            "Reduced local memory usage",
            "Faster app startup"
        ]
    }
    
    if not api_status.get("model_available"):
        summary["setup_needed"] = True
        summary["setup_url"] = "https://huggingface.co"
    
    return summary

def get_performance_tips() -> list:
    """Get performance optimization tips for HF deployment"""
    return [
        "🚀 **Enable Caching**: Embeddings are cached to reduce API calls",
        "⚡ **Batch Processing**: Upload resumes in smaller batches for faster processing",
        "🔄 **Model Warm-up**: First API call may be slower while model loads",
        "💾 **Local Cache**: Repeated analyses use cached embeddings",
        "📊 **Rate Limits**: Free tier has generous limits for most use cases",
        "🔧 **Fallback Mode**: App works without HF token using keyword analysis"
    ]