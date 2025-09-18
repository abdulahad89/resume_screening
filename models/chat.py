import requests
import streamlit as st
import time
import re
from typing import Dict, Any, Optional

class HuggingFaceChatBot:
    """Enhanced chatbot using Hugging Face Inference API with local fallback"""
    
    def __init__(self):
        from config import (
            HUGGINGFACE_API_URL, HUGGINGFACE_TOKEN, CHAT_MODEL,
            OLLAMA_URL, OLLAMA_MODEL, HF_TIMEOUT
        )
        
        self.hf_api_url = HUGGINGFACE_API_URL
        self.hf_token = HUGGINGFACE_TOKEN
        self.hf_model = CHAT_MODEL
        self.ollama_url = OLLAMA_URL
        self.ollama_model = OLLAMA_MODEL
        self.timeout = HF_TIMEOUT
        
        # API headers for HF
        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        
        # System prompt for HR assistant
        self.system_prompt = """You are an expert HR assistant helping with resume screening and candidate evaluation.

Your role is to analyze resumes and job descriptions to help recruiters make informed hiring decisions.

Instructions:
1. Answer questions based ONLY on the provided context (job description and candidate information)
2. Provide specific evidence from the documents when making assessments
3. Be objective and professional in your evaluations
4. If you don't have enough information, clearly state this limitation
5. For candidate comparisons, highlight key differentiators with specific examples
6. Keep responses concise and focused (under 250 words)
7. Use bullet points for clarity when listing multiple items
8. Always reference specific candidates by name when making recommendations

Remember: Base all assessments on factual information from the provided context only."""
        
        # Check available chat methods
        self.chat_status = self._check_chat_methods()
    
    def _check_chat_methods(self) -> Dict[str, Any]:
        """Check which chat methods are available"""
        status = {
            "hf_available": False,
            "ollama_available": False,
            "fallback_only": True,
            "primary_method": "fallback",
            "hf_error": None,
            "ollama_error": None
        }
        
        # Check Hugging Face
        if self.hf_token:
            try:
                test_url = f"{self.hf_api_url}/models/{self.hf_model}"
                response = requests.post(
                    test_url,
                    headers=self.headers,
                    json={"inputs": "test"},
                    timeout=5
                )
                
                if response.status_code in [200, 503]:  # 503 means loading
                    status["hf_available"] = True
                    status["primary_method"] = "huggingface"
                    status["fallback_only"] = False
                elif response.status_code == 401:
                    status["hf_error"] = "Invalid HF token"
                else:
                    status["hf_error"] = f"HF API error: {response.status_code}"
                    
            except Exception as e:
                status["hf_error"] = f"HF connection failed: {str(e)}"
        else:
            status["hf_error"] = "No HF token provided"
        
        # Check Ollama
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if self.ollama_model in model_names:
                    status["ollama_available"] = True
                    if not status["hf_available"]:
                        status["primary_method"] = "ollama"
                        status["fallback_only"] = False
                else:
                    status["ollama_error"] = f"{self.ollama_model} not found"
            else:
                status["ollama_error"] = "Ollama not responding"
                
        except Exception as e:
            status["ollama_error"] = f"Ollama connection failed: {str(e)}"
        
        return status
    
    def get_chat_status(self) -> Dict[str, Any]:
        """Get current chat system status"""
        return self.chat_status
    
    def _call_hf_chat(self, prompt: str) -> Optional[str]:
        """Call Hugging Face chat model"""
        try:
            url = f"{self.hf_api_url}/models/{self.hf_model}"
            
            # Format prompt for chat model
            formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
            
            payload = {
                "inputs": formatted_prompt,
                "parameters": {
                    "max_new_tokens": 300,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                },
                "options": {
                    "wait_for_model": True,
                    "use_cache": False
                }
            }
            
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                elif isinstance(result, dict):
                    generated_text = result.get('generated_text', result.get('text', ''))
                else:
                    return None
                
                # Clean up the response
                return self._clean_response(generated_text)
                
            elif response.status_code == 503:
                return "🔄 **Model Loading**: The Hugging Face model is currently loading. Please try again in a moment."
            else:
                return f"❌ **HF API Error**: HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "⏰ **Timeout**: Hugging Face API request timed out"
        except Exception as e:
            return f"❌ **HF Error**: {str(e)}"
    
    def _call_ollama_chat(self, prompt: str) -> Optional[str]:
        """Call local Ollama chat model"""
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 300,
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                return f"❌ **Ollama Error**: HTTP {response.status_code}"
                
        except Exception as e:
            return f"❌ **Ollama Error**: {str(e)}"
    
    def _clean_response(self, response: str) -> str:
        """Clean up AI response"""
        # Remove prompt remnants
        response = re.sub(r'^(Answer:|Assistant:|Human:|<\|.*?\|>)', '', response, flags=re.IGNORECASE)
        
        # Remove repetitive patterns
        lines = response.split('\n')
        cleaned_lines = []
        prev_line = ""
        
        for line in lines:
            line = line.strip()
            if line and line != prev_line:
                cleaned_lines.append(line)
                prev_line = line
        
        return '\n'.join(cleaned_lines).strip()
    
    def ask_question(self, question: str, context: str) -> str:
        """Ask question using best available chat method"""
        
        # Prepare full prompt
        full_prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Truncate context if too long
        if len(full_prompt) > 3000:
            lines = context.split('\n')
            truncated_context = '\n'.join(lines[:20]) + "\n\n[Context truncated...]"
            full_prompt = f"{self.system_prompt}\n\nContext:\n{truncated_context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Try primary method first
        if self.chat_status["primary_method"] == "huggingface":
            with st.spinner("🤖 Hugging Face AI is thinking..."):
                response = self._call_hf_chat(full_prompt)
            
            if response and not response.startswith("❌") and not response.startswith("⏰"):
                return f"**🤖 HF AI Response:**\n{response}"
            else:
                # Fall back to Ollama
                if self.chat_status["ollama_available"]:
                    st.warning("HF failed, trying Ollama...")
                    return self._try_ollama_fallback(full_prompt)
                else:
                    # Fall back to rule-based
                    st.warning("HF failed, using rule-based assistant...")
                    return self._rule_based_response(question, context)
        
        elif self.chat_status["primary_method"] == "ollama":
            return self._try_ollama_fallback(full_prompt)
        
        else:
            # Only rule-based available
            return self._rule_based_response(question, context)
    
    def _try_ollama_fallback(self, prompt: str) -> str:
        """Try Ollama as fallback"""
        with st.spinner("🤖 Local Ollama AI is thinking..."):
            response = self._call_ollama_chat(prompt)
        
        if response and not response.startswith("❌"):
            return f"**🤖 Local AI Response:**\n{response}"
        else:
            return self._rule_based_response("", "")
    
    def _rule_based_response(self, question: str, context: str) -> str:
        """Rule-based fallback response"""
        # Use the same rule-based logic as before
        fallback = RuleBasedChatBot()
        response = fallback.ask_question(question, context)
        return f"**📊 Rule-Based Analysis:**\n{response}"


class RuleBasedChatBot:
    """Rule-based chatbot fallback when AI models are unavailable"""
    
    def __init__(self):
        self.patterns = {
            'best': ['best', 'top', 'highest', 'most suitable', 'strongest'],
            'skills': ['skill', 'technical', 'programming', 'technology', 'expertise'],
            'experience': ['experience', 'years', 'worked', 'background', 'career'],
            'compare': ['compare', 'comparison', 'versus', 'vs', 'difference'],
            'education': ['education', 'degree', 'university', 'college', 'academic'],
            'recommend': ['recommend', 'suggest', 'advice', 'choose', 'select'],
            'weakness': ['weakness', 'gap', 'missing', 'lack', 'deficiency']
        }
    
    def ask_question(self, question: str, context: str) -> str:
        """Provide rule-based responses"""
        question_lower = question.lower()
        candidates = self._parse_candidates(context)
        
        if not candidates:
            return "📊 **Analysis**: No candidate data found. Please ensure you've screened candidates first."
        
        # Route to appropriate response based on question type
        if any(word in question_lower for word in self.patterns['best']):
            return self._answer_best_candidate(candidates)
        elif any(word in question_lower for word in self.patterns['skills']):
            return self._answer_skills_question(candidates)
        elif any(word in question_lower for word in self.patterns['compare']):
            return self._answer_comparison(candidates)
        elif any(word in question_lower for word in self.patterns['recommend']):
            return self._answer_recommendation(candidates)
        else:
            return self._general_analysis(candidates)
    
    def _parse_candidates(self, context: str) -> list:
        """Extract candidate information from context"""
        candidates = []
        lines = context.split('\n')
        current_candidate = None
        
        for line in lines:
            if re.match(r'^\d+\.', line.strip()):
                if current_candidate:
                    candidates.append(current_candidate)
                
                parts = line.split('(')
                if len(parts) >= 2:
                    name = parts[0].strip()[2:].strip()
                    score_part = parts[1].split(')')[0]
                    current_candidate = {
                        'name': name,
                        'score': score_part,
                        'details': [line.strip()]
                    }
            elif current_candidate and line.strip():
                current_candidate['details'].append(line.strip())
        
        if current_candidate:
            candidates.append(current_candidate)
        
        return candidates
    
    def _answer_best_candidate(self, candidates: list) -> str:
        """Answer about the best candidate"""
        if not candidates:
            return "📊 No candidate data available."
        
        top_candidate = candidates[0]
        return f"""🏆 **Top Candidate**: **{top_candidate['name']}**

📊 **Relevance Score**: {top_candidate['score']}

**Why this candidate ranks highest:**
• Strongest semantic alignment with job requirements
• Best skill category matches among all candidates
• Highest combined relevance score from our analysis

💡 **Recommendation**: Prioritize this candidate for initial screening and interview."""
    
    def _answer_skills_question(self, candidates: list) -> str:
        """Answer skills-related questions"""
        response = "🔧 **Technical Skills Analysis**\n\n"
        response += "**Top candidates by skill alignment:**\n"
        
        for i, candidate in enumerate(candidates[:3]):
            response += f"{i+1}. **{candidate['name']}** ({candidate['score']})\n"
        
        response += "\n💡 **Note**: Detailed skill breakdowns are available in the screening results section."
        return response
    
    def _answer_comparison(self, candidates: list) -> str:
        """Compare top candidates"""
        if len(candidates) < 2:
            return "⚖️ Need at least 2 candidates for comparison."
        
        response = "⚖️ **Top Candidate Comparison**\n\n"
        
        for i, candidate in enumerate(candidates[:3]):
            rank_emoji = ["🥇", "🥈", "🥉"][i]
            response += f"{rank_emoji} **{candidate['name']}**\n"
            response += f"   📊 Score: {candidate['score']}\n"
            response += f"   🎯 Rank: #{i+1}\n\n"
        
        response += "**Differentiators**: Scores reflect semantic similarity, skill matches, and overall job alignment."
        return response
    
    def _answer_recommendation(self, candidates: list) -> str:
        """Provide hiring recommendations"""
        response = "💡 **Hiring Recommendations**\n\n"
        
        if candidates:
            response += f"**🎯 Primary Recommendation**: {candidates[0]['name']} ({candidates[0]['score']})\n\n"
            
            if len(candidates) > 1:
                response += f"**🔄 Alternative**: {candidates[1]['name']} ({candidates[1]['score']})\n\n"
        
        response += """**📋 Next Steps:**
• Schedule technical interviews with top candidates
• Prepare role-specific assessment questions
• Review detailed candidate profiles for interview talking points
• Consider team fit and cultural alignment"""
        
        return response
    
    def _general_analysis(self, candidates: list) -> str:
        """General candidate analysis"""
        response = f"📈 **Candidate Pool Analysis**\n\n"
        response += f"**📊 Total Qualified Candidates**: {len(candidates)}\n"
        
        if candidates:
            response += f"**🏆 Top Performer**: {candidates[0]['name']} ({candidates[0]['score']})\n"
        
        response += f"""
**🎯 Analysis Method**: Advanced semantic similarity using Hugging Face embeddings
**📋 Ranking Factors**: Job-resume alignment, skill matching, experience relevance
**💡 Quality**: Higher scores indicate stronger job fit

**Next Steps**: Review detailed profiles and conduct interviews with top candidates."""
        
        return response