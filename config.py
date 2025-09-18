import os
import streamlit as st

# Hugging Face Configuration  
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co"

# Try to get token from Streamlit secrets first, then environment
try:
    HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
except:
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Rest of your config remains the same...
HF_EMBEDDING_MODELS = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "distilbert": "sentence-transformers/all-distilroberta-v1", 
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "roberta": "sentence-transformers/all-roberta-large-v1",
    "bge_large": "BAAI/bge-large-en-v1.5",
    "e5_large": "intfloat/e5-large-v2",
}

EMBEDDING_MODEL = HF_EMBEDDING_MODELS["mpnet"]

# Chat models and other settings
HF_CHAT_MODELS = {
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "llama": "meta-llama/Llama-2-7b-chat-hf", 
    "mistral": "mistralai/Mistral-7B-Instruct-v0.1"
}

CHAT_MODEL = HF_CHAT_MODELS["phi3"]

# Ollama backup config
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi3:mini"

# File settings
UPLOAD_FOLDER = "./data/uploads"
VECTOR_STORE_PATH = "./data/vector_store"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Scoring weights
SCORING_WEIGHTS = {
    "semantic": 0.5,
    "keyword_categories": 0.35,
    "tfidf": 0.15
}

# Thresholds
MIN_SCORE_THRESHOLD = 0.25
TOP_CANDIDATES = 20

# API settings
HF_TIMEOUT = 30
MAX_RETRIES = 3
CACHE_EMBEDDINGS = True

# Your skill categories and job templates go here...
SKILL_CATEGORIES = {
    'programming_languages': [
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'r', 'sql', 
        'scala', 'go', 'rust', 'php', 'ruby', 'swift', 'kotlin', 'matlab'
    ],
    'frameworks_libraries': [
        'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
        'spring', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
        'keras', 'opencv', 'bootstrap', 'jquery', 'laravel', 'rails'
    ],
    'databases': [
        'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
        'oracle', 'sql server', 'sqlite', 'dynamodb', 'neo4j', 'bigquery'
    ],
    'cloud_platforms': [
        'aws', 'azure', 'gcp', 'google cloud', 'amazon web services',
        'microsoft azure', 'digital ocean', 'heroku', 'netlify', 'vercel'
    ],
    'data_science_tools': [
        'tableau', 'power bi', 'matplotlib', 'seaborn', 'plotly', 'qlik',
        'looker', 'jupyter', 'r studio', 'spss', 'sas', 'alteryx', 'databricks'
    ],
    'devops_tools': [
        'docker', 'kubernetes', 'jenkins', 'git', 'gitlab', 'github',
        'terraform', 'ansible', 'puppet', 'chef', 'nginx', 'prometheus'
    ],
    'marketing_tools': [
        'google analytics', 'google ads', 'facebook ads', 'hubspot', 'marketo',
        'salesforce', 'mailchimp', 'hootsuite', 'semrush', 'ahrefs', 'optimizely'
    ],
    'methodologies': [
        'agile', 'scrum', 'kanban', 'waterfall', 'lean', 'six sigma',
        'design thinking', 'test driven development', 'devops', 'continuous integration'
    ],
    'business_tools': [
        'excel', 'powerpoint', 'jira', 'confluence', 'asana', 'trello',
        'figma', 'sketch', 'visio', 'slack', 'teams', 'notion'
    ],
    'certifications': [
        'pmp', 'aws certified', 'azure certified', 'google certified',
        'cissp', 'scrum master', 'product owner', 'six sigma', 'itil'
    ]
}

# Job templates
JOB_TEMPLATES = {
    "Data Scientist": """
    Data Scientist position requiring expertise in machine learning algorithms, statistical modeling,
    and advanced data analysis. Technical skills include: Python programming with pandas, numpy, and scipy,
    SQL database management and complex query optimization, machine learning frameworks including 
    scikit-learn, TensorFlow, and PyTorch, data visualization using matplotlib, seaborn, plotly, and Tableau.
    
    Experience with big data technologies: Apache Spark, Hadoop ecosystem, distributed computing,
    cloud platforms including AWS (SageMaker, EC2, S3), Azure Machine Learning, or Google Cloud AI Platform.
    Statistical expertise in hypothesis testing, A/B testing, experimental design, time series analysis,
    regression modeling, classification algorithms, clustering techniques, and deep learning architectures.
    
    Strong mathematical background in statistics, linear algebra, calculus, and probability theory.
    Business intelligence tools experience with Tableau, Power BI, Looker, or similar platforms.
    Version control with Git, collaborative development practices, Jupyter notebooks, and model deployment.
    """,
    
    "Marketing Analyst": """
    Marketing Analyst role focusing on digital marketing analytics, campaign optimization, and customer insights.
    Core technical skills: Google Analytics advanced implementation, Google Ads campaign management and optimization,
    Facebook Ads Manager and Business Manager, marketing automation platforms including HubSpot, Marketo, or Pardot,
    customer relationship management with Salesforce, HubSpot CRM, or similar platforms.
    
    Data analysis expertise: SQL for marketing database queries and customer segmentation, Excel and Google Sheets
    for advanced modeling and pivot table analysis, data visualization using Tableau, Power BI, or Looker,
    A/B testing implementation and statistical significance analysis, conversion rate optimization and funnel analysis,
    customer lifetime value modeling, attribution modeling across multiple touchpoints and channels.
    
    Digital marketing channels expertise: search engine optimization (SEO) and search engine marketing (SEM),
    social media advertising across platforms, email marketing performance tracking, campaign ROI measurement.
    """,
    
    "Software Engineer": """
    Software Engineer position requiring full-stack web development expertise and modern software architecture.
    Programming languages: Python, JavaScript/TypeScript, Java, with frameworks including React, Node.js, Django, Flask,
    Spring Boot, database design with MySQL, PostgreSQL, MongoDB, Redis caching, RESTful API development,
    microservices architecture, Docker containerization, Kubernetes orchestration, cloud deployment on AWS/Azure/GCP.
    
    DevOps practices: CI/CD pipelines with Jenkins or GitHub Actions, infrastructure as code with Terraform,
    monitoring and logging, version control with Git, agile development methodologies, test-driven development,
    code review processes, system design and architecture planning, performance optimization and scalability.
    """
}

# Directory creation function
def create_directories():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    os.makedirs("./data/embeddings_cache", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

# Model info function
def get_model_info():
    model_name = EMBEDDING_MODEL.split('/')[-1] if '/' in EMBEDDING_MODEL else EMBEDDING_MODEL
    return {
        "name": model_name,
        "full_name": EMBEDDING_MODEL,
        "provider": "Hugging Face Inference API",
        "type": "Remote Inference",
        "description": "High-quality embeddings via Hugging Face cloud infrastructure",
        "advantages": [
            "No local model downloads or storage",
            "Always latest model version",
            "Scalable cloud infrastructure", 
            "Reduced local memory usage",
            "Faster application startup"
        ]
    }
