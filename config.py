import os
from typing import Dict, Any

class Config:
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./brainwave_tutor.db')
    
    # Model Configuration
    BASE_MODEL = os.getenv('BASE_MODEL', 'microsoft/DialoGPT-medium')
    MATH_MODEL = os.getenv('MATH_MODEL', 'WizardLM/WizardMath-7B-V1.1')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    
    # Model Fallbacks (if primary models fail to load)
    FALLBACK_BASE_MODEL = 'distilgpt2'
    FALLBACK_MATH_MODEL = 'microsoft/DialoGPT-small'
    
    # Memory and Storage Configuration
    CHROMA_DB_PATH = os.getenv('CHROMA_DB_PATH', './chroma_db')
    MODELS_DIRECTORY = os.getenv('MODELS_DIR', './models')
    CACHE_DIRECTORY = os.getenv('CACHE_DIR', './cache')
    
    # Memory Management
    MAX_MEMORY_GB = float(os.getenv('MAX_MEMORY_GB', '8.0'))
    MEMORY_CLEANUP_THRESHOLD = float(os.getenv('MEMORY_CLEANUP_THRESHOLD', '0.8'))
    
    # Session Configuration
    SESSION_TIMEOUT_SECONDS = int(os.getenv('SESSION_TIMEOUT', '3600'))  # 1 hour
    MAX_ACTIVE_SESSIONS = int(os.getenv('MAX_ACTIVE_SESSIONS', '100'))
    
    # Learning Configuration
    MEMORY_RETENTION_DAYS = {
        'interaction': int(os.getenv('INTERACTION_RETENTION_DAYS', '90')),
        'preference': int(os.getenv('PREFERENCE_RETENTION_DAYS', '365')),
        'achievement': int(os.getenv('ACHIEVEMENT_RETENTION_DAYS', '730')),
        'mistake': int(os.getenv('MISTAKE_RETENTION_DAYS', '180')),
        'insight': int(os.getenv('INSIGHT_RETENTION_DAYS', '365'))
    }
    
    # Model Generation Parameters
    GENERATION_CONFIG = {
        'max_new_tokens': int(os.getenv('MAX_NEW_TOKENS', '256')),
        'temperature': float(os.getenv('TEMPERATURE', '0.7')),
        'top_p': float(os.getenv('TOP_P', '0.9')),
        'repetition_penalty': float(os.getenv('REPETITION_PENALTY', '1.1')),
        'do_sample': True,
        'no_repeat_ngram_size': int(os.getenv('NO_REPEAT_NGRAM_SIZE', '3'))
    }
    
    # Math Query Detection Thresholds
    MATH_DETECTION_CONFIG = {
        'keyword_threshold': int(os.getenv('MATH_KEYWORD_THRESHOLD', '2')),
        'pattern_weight': int(os.getenv('MATH_PATTERN_WEIGHT', '2')),
        'confidence_threshold': float(os.getenv('MATH_CONFIDENCE_THRESHOLD', '0.6'))
    }
    
    # User Profile Configuration
    PROFILE_CONFIG = {
        'initial_confidence': float(os.getenv('INITIAL_PROFILE_CONFIDENCE', '0.1')),
        'max_confidence': float(os.getenv('MAX_PROFILE_CONFIDENCE', '1.0')),
        'confidence_increment': float(os.getenv('CONFIDENCE_INCREMENT', '0.05')),
        'update_weight': float(os.getenv('PROFILE_UPDATE_WEIGHT', '0.3'))
    }
    
    # Memory Retrieval Configuration
    MEMORY_CONFIG = {
        'default_retrieval_limit': int(os.getenv('MEMORY_RETRIEVAL_LIMIT', '10')),
        'similarity_threshold': float(os.getenv('MEMORY_SIMILARITY_THRESHOLD', '0.7')),
        'importance_weight': float(os.getenv('MEMORY_IMPORTANCE_WEIGHT', '0.8'))
    }
    
    # Performance Metrics Configuration
    METRICS_CONFIG = {
        'engagement_threshold': float(os.getenv('ENGAGEMENT_THRESHOLD', '0.6')),
        'performance_threshold': float(os.getenv('PERFORMANCE_THRESHOLD', '0.7')),
        'analytics_days_back': int(os.getenv('ANALYTICS_DAYS_BACK', '30'))
    }
    
    # API Configuration
    API_CONFIG = {
        'host': os.getenv('API_HOST', '0.0.0.0'),
        'port': int(os.getenv('API_PORT', '8000')),
        'reload': os.getenv('API_RELOAD', 'false').lower() == 'true',
        'workers': int(os.getenv('API_WORKERS', '1'))
    }
    
    # Security Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-super-secret-key-change-this-in-production')
    ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '30'))
    
    # External API Configuration
    GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')
    HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN', '')
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    LOG_FILE = os.getenv('LOG_FILE', 'brainwave.log')
    
    # Feature Flags
    FEATURES = {
        'enable_math_routing': os.getenv('ENABLE_MATH_ROUTING', 'true').lower() == 'true',
        'enable_memory_system': os.getenv('ENABLE_MEMORY_SYSTEM', 'true').lower() == 'true',
        'enable_user_profiling': os.getenv('ENABLE_USER_PROFILING', 'true').lower() == 'true',
        'enable_feedback_learning': os.getenv('ENABLE_FEEDBACK_LEARNING', 'true').lower() == 'true',
        'enable_analytics': os.getenv('ENABLE_ANALYTICS', 'true').lower() == 'true',
        'enable_gpu': os.getenv('ENABLE_GPU', 'true').lower() == 'true',
        'auto_cleanup_memory': os.getenv('AUTO_CLEANUP_MEMORY', 'true').lower() == 'true'
    }
    
    # System Prompts
    SYSTEM_PROMPTS = {
        'base_tutor': """You are Dr. Alexandra Chen, an expert AI tutor with comprehensive knowledge across multiple subjects. 
You provide clear, educational responses tailored to each student's learning style and needs. 
Be encouraging, patient, and adapt your explanations to help students understand complex concepts.""",
        
        'math_tutor': """You are a specialized mathematics tutor. Provide clear, step-by-step solutions to mathematical problems.
Break down complex problems into manageable steps and explain the reasoning behind each step.
Use appropriate mathematical notation and encourage students to check their work.""",
        
        'learning_style_visual': """Focus on visual examples, diagrams, and clear formatting. 
Use bullet points, numbered lists, and visual metaphors to explain concepts.""",
        
        'learning_style_auditory': """Use conversational language and verbal explanations. 
Explain concepts as if speaking aloud, using rhythm and repetition for emphasis.""",
        
        'learning_style_kinesthetic': """Emphasize hands-on examples and practical applications. 
Connect abstract concepts to real-world scenarios and physical experiences.""",
        
        'learning_style_reading': """Provide detailed, well-structured text explanations. 
Use comprehensive written descriptions and encourage note-taking."""
    }
    
    # Error Messages
    ERROR_MESSAGES = {
        'model_loading_failed': "I'm having trouble loading the AI models. Please try again later.",
        'memory_error': "I encountered an issue accessing my memory system. The response may be limited.",
        'database_error': "There was a problem connecting to the database. Please try again.",
        'generation_error': "I apologize, but I encountered an error while generating a response.",
        'user_not_found': "User profile not found. Please initialize your profile first.",
        'session_expired': "Your session has expired. Please start a new session."
    }
    
    # Subject Categories for Math Detection
    MATH_SUBJECTS = {
        'arithmetic': ['add', 'subtract', 'multiply', 'divide', '+', '-', '*', '/', 'sum', 'difference', 'product', 'quotient'],
        'algebra': ['equation', 'variable', 'solve', 'x', 'y', 'z', 'polynomial', 'quadratic', 'linear', 'expression'],
        'geometry': ['angle', 'triangle', 'circle', 'area', 'volume', 'perimeter', 'radius', 'diameter', 'polygon'],
        'calculus': ['derivative', 'integral', 'limit', 'differentiate', 'integrate', 'function', 'slope', 'tangent'],
        'statistics': ['mean', 'median', 'mode', 'standard deviation', 'probability', 'variance', 'distribution'],
        'trigonometry': ['sin', 'cos', 'tan', 'sine', 'cosine', 'tangent', 'angle', 'radian', 'degree']
    }
    
    # Difficulty Level Indicators
    DIFFICULTY_INDICATORS = {
        'basic': ['simple', 'easy', 'basic', 'elementary', 'intro', 'beginner'],
        'intermediate': ['solve', 'calculate', 'find', 'determine', 'compute', 'work out'],
        'advanced': ['prove', 'derive', 'optimize', 'evaluate', 'analyze', 'demonstrate', 'verify']
    }
    
    # Performance Thresholds
    PERFORMANCE_THRESHOLDS = {
        'excellent': 0.9,
        'good': 0.7,
        'satisfactory': 0.5,
        'needs_improvement': 0.3,
        'poor': 0.0
    }
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration based on available resources"""
        return {
            'base_model': cls.BASE_MODEL,
            'math_model': cls.MATH_MODEL,
            'embedding_model': cls.EMBEDDING_MODEL,
            'generation_config': cls.GENERATION_CONFIG,
            'max_memory_gb': cls.MAX_MEMORY_GB,
            'enable_gpu': cls.FEATURES['enable_gpu']
        }
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            'url': cls.DATABASE_URL,
            'echo': cls.LOG_LEVEL.lower() == 'debug'
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        required_settings = [
            'DATABASE_URL', 'BASE_MODEL', 'SECRET_KEY'
        ]
        
        for setting in required_settings:
            if not hasattr(cls, setting) or not getattr(cls, setting):
                print(f"Warning: Required setting {setting} not configured")
                return False
        
        # Create necessary directories
        directories = [cls.MODELS_DIRECTORY, cls.CACHE_DIRECTORY]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        return True

# Validate configuration on import
if not Config.validate_config():
    print("Configuration validation failed. Please check your settings.")

# Export commonly used configurations
__all__ = ['Config']