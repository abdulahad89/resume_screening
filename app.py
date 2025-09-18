import streamlit as st
import os
import pandas as pd
from typing import List, Dict, Any
import time

# Set page config first
st.set_page_config(
    page_title="Resume Screening Tool - Hugging Face Enhanced",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom modules
from config import *
from models.embeddings import HuggingFaceEmbeddingManager
from models.parser import EnhancedResumeParser
from models.chat import HuggingFaceChatBot, RuleBasedChatBot
from utils.helpers import save_uploaded_file, calculate_enhanced_similarity, get_scoring_color

# Initialize session state
if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = HuggingFaceEmbeddingManager()
if 'parser' not in st.session_state:
    st.session_state.parser = EnhancedResumeParser()
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = HuggingFaceChatBot()
if 'fallback_bot' not in st.session_state:
    st.session_state.fallback_bot = RuleBasedChatBot()
if 'job_description' not in st.session_state:
    st.session_state.job_description = ""
if 'resumes' not in st.session_state:
    st.session_state.resumes = []
if 'screening_results' not in st.session_state:
    st.session_state.screening_results = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def main():
    # Header with model info
    st.title("🔍 Resume Screening Tool - Hugging Face Enhanced")
    st.markdown("**AI-powered candidate evaluation with Hugging Face Inference API - No local model downloads required!**")
    
    # Display model info
    model_info = get_model_info()
    st.info(f"🤖 Using **{model_info['name']}** via {model_info['provider']} - {model_info['description']}")
    
    # Sidebar for system status and controls
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Check HF API status
        api_status = st.session_state.embedding_manager.get_api_status()
        
        if api_status["model_available"]:
            st.success("✅ Hugging Face API Connected")
            st.success(f"✅ {model_info['name']} Available")
        else:
            st.error("❌ Hugging Face API Issue")
            if api_status["error"]:
                st.error(f"Error: {api_status['error']}")
                
                # Show setup instructions if token missing
                if "token" in api_status["error"].lower():
                    st.subheader("🔑 Setup Required")
                    st.markdown("""
                    **Get your free HF token:**
                    1. Go to [huggingface.co](https://huggingface.co)
                    2. Create free account
                    3. Go to Settings → Access Tokens
                    4. Create new token
                    5. Add to .env file: `HUGGINGFACE_TOKEN=your_token_here`
                    """)
        
        # Check chat system status
        chat_status = st.session_state.chatbot.get_chat_status()
        st.subheader("💬 Chat System")
        
        if chat_status["primary_method"] == "huggingface":
            st.success("🤖 HF Chat Available")
        elif chat_status["primary_method"] == "ollama":
            st.success("🤖 Local Ollama Available")
        else:
            st.info("📊 Rule-based Assistant")
            
        # Model advantages
        st.subheader("🚀 HF Advantages")
        for advantage in model_info['advantages']:
            st.write(f"• {advantage}")
        
        # Statistics
        if st.session_state.resumes:
            st.subheader("📊 Current Session")
            st.metric("Resumes Uploaded", len(st.session_state.resumes))
        if st.session_state.screening_results:
            st.metric("Candidates Screened", len(st.session_state.screening_results))
            avg_score = sum(r['final_score'] for r in st.session_state.screening_results) / len(st.session_state.screening_results)
            st.metric("Average Relevance", f"{avg_score:.1%}")
            
        # Cache management
        st.subheader("🗄️ Cache Management")
        if st.button("Clear Embedding Cache"):
            st.session_state.embedding_manager.clear_cache()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Job Description", "📄 Upload Resumes", "🎯 Screen Candidates", "💬 AI Chat"])
    
    with tab1:
        job_description_tab()
    
    with tab2:
        upload_resumes_tab()
    
    with tab3:
        screening_tab()
    
    with tab4:
        chat_tab()

def job_description_tab():
    st.header("📋 Job Description Input")
    st.markdown("**Create detailed job descriptions for better AI matching with Hugging Face embeddings.**")
    
    # API status check
    api_status = st.session_state.embedding_manager.get_api_status()
    if not api_status["model_available"]:
        st.warning("⚠️ Hugging Face API not available. Screening will use fallback methods.")
    
    # Three input methods
    jd_method = st.radio(
        "Choose input method:",
        ["📝 Text Input", "📁 Upload Document", "📋 Job Template"],
        horizontal=True
    )
    
    if jd_method == "📝 Text Input":
        jd_text = st.text_area(
            "Enter job description:",
            height=250,
            placeholder="Paste your detailed job description here. The more specific you are with skills and requirements, the better the AI matching will be...",
            help="💡 Tip: Detailed descriptions with specific skills lead to more accurate candidate matching"
        )
        if st.button("💾 Save Job Description", type="primary") and jd_text:
            st.session_state.job_description = jd_text
            st.success("✅ Job description saved!")
            
            # Show extracted keywords preview
            with st.spinner("🔍 Analyzing with Hugging Face..."):
                keywords = st.session_state.embedding_manager.extract_enhanced_keywords(jd_text)
                if keywords:
                    st.subheader("🔑 Key Skills Detected")
                    cols = st.columns(2)
                    for i, (category, skills) in enumerate(keywords.items()):
                        with cols[i % 2]:
                            st.write(f"**{category.replace('_', ' ').title()}**")
                            st.write(f"• {', '.join(skills[:5])}")
    
    elif jd_method == "📁 Upload Document":
        uploaded_file = st.file_uploader(
            "Upload job description file:",
            type=['pdf', 'docx', 'txt'],
            help="Supported formats: PDF, DOCX, TXT (Max 10MB)"
        )
        if uploaded_file and st.button("📖 Process Document", type="primary"):
            with st.spinner("Processing document..."):
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    jd_text = st.session_state.parser.extract_text(file_path)
                    if jd_text:
                        st.session_state.job_description = jd_text
                        st.success("✅ Job description extracted and saved!")
                        st.text_area("Extracted text:", jd_text, height=200)
                    else:
                        st.error("❌ Could not extract text from document")
    
    elif jd_method == "📋 Job Template":
        template_name = st.selectbox(
            "Select a job template:",
            [""] + list(JOB_TEMPLATES.keys()),
            help="Pre-built templates optimized for AI matching"
        )
        if template_name:
            template_text = JOB_TEMPLATES[template_name]
            st.text_area("Template preview:", template_text, height=200)
            if st.button("📋 Use This Template", type="primary"):
                st.session_state.job_description = template_text
                st.success(f"✅ {template_name} template loaded!")
    
    # Show current JD
    if st.session_state.job_description:
        st.subheader("📄 Current Job Description")
        with st.expander("View current job description", expanded=False):
            st.write(st.session_state.job_description)
            st.caption(f"📝 Length: {len(st.session_state.job_description.split())} words")

def upload_resumes_tab():
    st.header("📄 Upload Resumes")
    st.markdown("**Upload candidate resumes. Hugging Face AI will analyze them for semantic similarity.**")
    
    # API status reminder
    api_status = st.session_state.embedding_manager.get_api_status()
    if api_status["model_available"]:
        st.info("🤖 Hugging Face embeddings will provide high-quality semantic analysis")
    else:
        st.warning("⚠️ Using fallback analysis - set up HF token for best results")
    
    uploaded_files = st.file_uploader(
        "Upload resume files:",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT (Max 10MB each)"
    )
    
    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files selected for processing")
        
        if st.button("🔄 Process All Resumes", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Save and parse file
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    parsed_data = st.session_state.parser.parse_resume(file_path, uploaded_file.name)
                    
                    if "error" not in parsed_data:
                        # Generate embedding with HF API
                        try:
                            embedding = st.session_state.embedding_manager.encode_text(parsed_data['raw_text'])
                            
                            resume_data = {
                                'filename': uploaded_file.name,
                                'text': parsed_data['raw_text'],
                                'embedding': embedding,
                                'skills': parsed_data.get('skills', {}),
                                'experience': parsed_data.get('experience', []),
                                'education': parsed_data.get('education', []),
                                'contact_info': parsed_data.get('contact_info', {}),
                                'word_count': parsed_data.get('word_count', 0)
                            }
                            st.session_state.resumes.append(resume_data)
                            
                        except Exception as e:
                            st.error(f"❌ Embedding error for {uploaded_file.name}: {str(e)}")
                    else:
                        st.error(f"❌ Failed to parse {uploaded_file.name}: {parsed_data['error']}")
            
            status_text.text("✅ All resumes processed!")
            successful_count = len([r for r in st.session_state.resumes if r['filename'] in [f.name for f in uploaded_files]])
            st.success(f"Successfully processed {successful_count} resumes!")
    
    # Show uploaded resumes
    if st.session_state.resumes:
        st.subheader(f"📊 Uploaded Resumes ({len(st.session_state.resumes)})")
        
        for i, resume in enumerate(st.session_state.resumes):
            with st.expander(f"📄 {resume['filename']} ({resume['word_count']} words)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Skills Categories:**", len(resume['skills']))
                    for category, skills in resume['skills'].items():
                        if skills:
                            st.write(f"• {category.replace('_', ' ').title()}: {len(skills)}")
                
                with col2:
                    st.write("**Experience:**", len(resume['experience']))
                    st.write("**Education:**", len(resume['education']))
                    if resume['contact_info'].get('emails'):
                        st.write("**Email:**", resume['contact_info']['emails'][0])
                
                with col3:
                    if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.resumes.pop(i)
                        st.rerun()

def screening_tab():
    st.header("🎯 Advanced Resume Screening with Hugging Face")
    
    if not st.session_state.job_description:
        st.warning("⚠️ Please add a job description first!")
        return
    
    if not st.session_state.resumes:
        st.warning("⚠️ Please upload some resumes first!")
        return
    
    api_status = st.session_state.embedding_manager.get_api_status()
    if api_status["model_available"]:
        st.markdown("**🤖 Using Hugging Face high-quality embeddings for semantic analysis**")
    else:
        st.markdown("**⚠️ Using fallback analysis - set up HF token for best results**")
    
    # Screening controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_score = st.slider("Minimum Score (%)", 0, 100, int(MIN_SCORE_THRESHOLD * 100), 5)
    with col2:
        max_results = st.slider("Max Results", 5, 50, TOP_CANDIDATES, 5)
    with col3:
        show_details = st.checkbox("Show Score Breakdown", value=True)
    with col4:
        if st.button("🚀 Screen Candidates", type="primary"):
            screen_candidates_enhanced(min_score/100, max_results, show_details)
    
    # Show results
    if st.session_state.screening_results:
        st.subheader("📊 Screening Results")
        
        # Summary stats
        scores = [r['final_score'] for r in st.session_state.screening_results]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📋 Total Candidates", len(st.session_state.screening_results))
        with col2:
            st.metric("📈 Average Score", f"{sum(scores)/len(scores):.1%}")
        with col3:
            st.metric("🏆 Top Score", f"{max(scores):.1%}")
        with col4:
            st.metric("📉 Lowest Score", f"{min(scores):.1%}")
        
        # Results display
        for i, candidate in enumerate(st.session_state.screening_results):
            score_color = get_scoring_color(candidate['final_score'])
            
            with st.container():
                st.markdown("---")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(f"{score_color} #{i+1}: {candidate['filename']}")
                
                with col2:
                    st.metric("Relevance Score", f"{candidate['final_score']:.1%}")
                
                if show_details and 'similarity_breakdown' in candidate:
                    breakdown = candidate['similarity_breakdown']
                    
                    # Score breakdown
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Semantic (HF)**: {breakdown.get('semantic', 0):.1%}")
                        st.caption("Hugging Face AI analysis")
                    with col2:
                        st.write(f"**Skill Categories**: {breakdown.get('keyword_categories', 0):.1%}")
                        st.caption("Category matching")
                    with col3:
                        st.write(f"**Text Similarity**: {breakdown.get('tfidf', 0):.1%}")
                        st.caption("Keyword overlap")
                    
                    # API provider info
                    if 'api_provider' in breakdown:
                        st.caption(f"🤖 Analysis by: {breakdown['api_provider']}")
                    
                    # Show skill matches
                    if 'category_breakdown' in breakdown:
                        st.write("**🎯 Skills Analysis:**")
                        skill_cols = st.columns(2)
                        breakdown_items = list(breakdown['category_breakdown'].items())
                        
                        for j, (category, details) in enumerate(breakdown_items[:4]):  # Show top 4
                            with skill_cols[j % 2]:
                                category_name = category.replace('_', ' ').title()
                                matched = details.get('matched', [])
                                missing = details.get('missing', [])
                                
                                if matched:
                                    st.write(f"✅ **{category_name}**: {', '.join(matched[:3])}")
                                if missing:
                                    st.write(f"❌ **Missing**: {', '.join(missing[:2])}")
                
                # Resume preview
                with st.expander(f"📄 Resume Preview - {candidate['filename']}", expanded=False):
                    preview_text = candidate['text'][:800] + "..." if len(candidate['text']) > 800 else candidate['text']
                    st.text_area("", preview_text, height=150, key=f"preview_{i}")

def screen_candidates_enhanced(min_score: float, max_results: int, show_details: bool):
    """Enhanced screening with HF embeddings"""
    
    api_status = st.session_state.embedding_manager.get_api_status()
    
    if api_status["model_available"]:
        with st.spinner("🤖 Hugging Face is analyzing candidates..."):
            _perform_hf_screening(min_score, max_results, show_details)
    else:
        with st.spinner("📊 Performing fallback analysis..."):
            _perform_fallback_screening(min_score, max_results, show_details)

def _perform_hf_screening(min_score: float, max_results: int, show_details: bool):
    """Perform screening using HF API"""
    
    results = []
    
    for resume in st.session_state.resumes:
        try:
            # Use HF hybrid similarity analysis
            similarity_results = st.session_state.embedding_manager.hybrid_similarity_analysis(
                st.session_state.job_description,
                resume['text']
            )
            
            final_score = similarity_results.get('combined', 0)
            
            if final_score >= min_score:
                candidate_result = {
                    'filename': resume['filename'],
                    'final_score': final_score,
                    'text': resume['text'],
                    'skills': st.session_state.embedding_manager.extract_enhanced_keywords(resume['text']),
                    'word_count': resume['word_count'],
                    'contact_info': resume.get('contact_info', {})
                }
                
                if show_details:
                    candidate_result['similarity_breakdown'] = similarity_results
                
                results.append(candidate_result)
                
        except Exception as e:
            st.error(f"Error processing {resume['filename']}: {str(e)}")
            continue
    
    # Sort and store results
    results.sort(key=lambda x: x['final_score'], reverse=True)
    st.session_state.screening_results = results[:max_results]
    
    if results:
        st.success(f"✅ Screened {len(results)} candidates with Hugging Face AI!")
    else:
        st.warning("⚠️ No candidates met the minimum score threshold.")

def _perform_fallback_screening(min_score: float, max_results: int, show_details: bool):
    """Fallback screening without HF API"""
    
    results = []
    
    for resume in st.session_state.resumes:
        # Simple keyword-based scoring
        jd_keywords = st.session_state.embedding_manager.extract_enhanced_keywords(st.session_state.job_description)
        resume_keywords = st.session_state.embedding_manager.extract_enhanced_keywords(resume['text'])
        
        # Calculate basic similarity
        keyword_analysis = st.session_state.embedding_manager.calculate_keyword_similarity(jd_keywords, resume_keywords)
        tfidf_score = st.session_state.embedding_manager.calculate_tfidf_similarity(
            st.session_state.job_description, resume['text']
        )
        
        final_score = keyword_analysis['overall_score'] * 0.7 + tfidf_score * 0.3
        
        if final_score >= min_score:
            candidate_result = {
                'filename': resume['filename'],
                'final_score': final_score,
                'text': resume['text'],
                'skills': resume_keywords,
                'word_count': resume['word_count']
            }
            
            if show_details:
                candidate_result['similarity_breakdown'] = {
                    'semantic': 0.0,  # Not available in fallback
                    'keyword_categories': keyword_analysis['overall_score'],
                    'tfidf': tfidf_score,
                    'category_breakdown': keyword_analysis['category_breakdown'],
                    'api_provider': 'Fallback Analysis'
                }
            
            results.append(candidate_result)
    
    results.sort(key=lambda x: x['final_score'], reverse=True)
    st.session_state.screening_results = results[:max_results]
    
    if results:
        st.info(f"📊 Screened {len(results)} candidates with fallback analysis")
    else:
        st.warning("⚠️ No candidates met the minimum score threshold.")

def chat_tab():
    st.header("💬 AI Chat about Candidates")
    
    if not st.session_state.screening_results:
        st.warning("⚠️ Please screen some candidates first!")
        return
    
    st.markdown("**Ask questions about your screened candidates using AI or rule-based analysis.**")
    
    # Chat status
    chat_status = st.session_state.chatbot.get_chat_status()
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("🤖 AI Status")
        
        if chat_status["primary_method"] == "huggingface":
            st.success("✅ HF Chat Available")
            ai_mode = "huggingface"
        elif chat_status["primary_method"] == "ollama":
            st.success("✅ Local Ollama Available")
            ai_mode = "ollama"
        else:
            st.info("📊 Rule-based Assistant")
            ai_mode = "fallback"
            
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col1:
        # Suggested questions
        st.subheader("💡 Suggested Questions")
        suggestions = [
            "Which candidates are most suitable for this position?",
            "What are the key skill gaps among the top candidates?",
            "Compare the top 3 candidates' qualifications",
            "Which candidate has the strongest technical background?",
            "What specific skills do the top candidates possess?",
            "Are there any candidates with relevant certifications?",
        ]
        
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(suggestion, key=f"suggest_{i}", help="Click to ask this question"):
                    ask_ai_question(suggestion)
    
    # Chat input
    st.subheader("🗨️ Ask Your Question")
    user_question = st.text_input(
        "Ask about the candidates:",
        placeholder="e.g., Which candidate would be best for a senior Python developer role?",
        help="Ask specific questions about skills, experience, or comparisons"
    )
    
    if st.button("📤 Send", type="primary") and user_question:
        ask_ai_question(user_question)
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Conversation History")
        
        for i, (question, answer, timestamp) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown("---")
                
                st.markdown(f"**👤 Question {i+1}:** {question}")
                st.caption(f"⏰ {timestamp}")
                
                st.markdown(f"**🤖 Answer:**")
                st.markdown(answer)

def ask_ai_question(question: str):
    """Ask AI about candidates"""
    
    # Prepare context
    context = f"Job Description:\n{st.session_state.job_description}\n\n"
    context += "Screened Candidates (ranked by relevance):\n\n"
    
    for i, candidate in enumerate(st.session_state.screening_results[:5]):
        context += f"{i+1}. {candidate['filename']} (Score: {candidate['final_score']:.1%})\n"
        
        if candidate['skills']:
            skills_summary = []
            for category, skills in candidate['skills'].items():
                if skills:
                    skills_summary.append(f"{category.replace('_', ' ')}: {', '.join(skills[:3])}")
            if skills_summary:
                context += f"   Skills: {'; '.join(skills_summary[:3])}\n"
        
        context += f"   Preview: {candidate['text'][:300]}...\n\n"
    
    # Get response from chatbot
    try:
        answer = st.session_state.chatbot.ask_question(question, context)
        
        # Add to chat history
        timestamp = time.strftime("%H:%M:%S")
        st.session_state.chat_history.append((question, answer, timestamp))
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Chat error: {str(e)}")

if __name__ == "__main__":
    create_directories()
    main()
