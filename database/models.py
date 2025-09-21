from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./brainwave_tutor.db')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    profile_data = Column(JSON, default={})
    learning_data = Column(JSON, default={})
    subscription_tier = Column(String, default="free")
    is_active = Column(Boolean, default=True)
    
    preferences = relationship("UserPreference", back_populates="user")
    sessions = relationship("LearningSession", back_populates="user")
    interactions = relationship("UserInteraction", back_populates="user")
    study_plans = relationship("StudyPlan", back_populates="user")

class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    preference_type = Column(String)
    preference_value = Column(JSON)
    confidence_score = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="preferences")

class LearningSession(Base):
    __tablename__ = "learning_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(String, unique=True, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    topic_focus = Column(String, nullable=True)
    interaction_count = Column(Integer, default=0)
    session_data = Column(JSON, default={})
    performance_score = Column(Float, nullable=True)
    engagement_score = Column(Float, nullable=True)
    
    user = relationship("User", back_populates="sessions")
    interactions = relationship("UserInteraction", back_populates="session")

class UserInteraction(Base):
    __tablename__ = "user_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(Integer, ForeignKey("learning_sessions.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    interaction_type = Column(String)
    user_input = Column(Text)
    ai_response = Column(Text)
    topic = Column(String, nullable=True)
    complexity_level = Column(String, nullable=True)
    response_time = Column(Float, nullable=True)
    user_rating = Column(Integer, nullable=True)
    feedback_data = Column(JSON, default={})
    context_data = Column(JSON, default={})
    personalization_factors = Column(JSON, default={})
    sources_used = Column(JSON, default={})
    
    user = relationship("User", back_populates="interactions")
    session = relationship("LearningSession", back_populates="interactions")

class StudyPlan(Base):
    __tablename__ = "study_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    plan_name = Column(String)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    target_completion = Column(DateTime, nullable=True)
    plan_data = Column(JSON, default={})
    progress_data = Column(JSON, default={})
    is_active = Column(Boolean, default=True)
    
    user = relationship("User", back_populates="study_plans")

class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    
    id = Column(Integer, primary_key=True, index=True)
    content_id = Column(String, unique=True, index=True)
    title = Column(String)
    content = Column(Text)
    subject = Column(String)
    topic = Column(String, nullable=True)
    difficulty_level = Column(String)
    content_type = Column(String)
    tags = Column(JSON, default=[])
    content_metadata = Column(JSON, default={})
    embedding_vector = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    usage_count = Column(Integer, default=0)
    effectiveness_score = Column(Float, default=0.0)

class UserMemory(Base):
    __tablename__ = "user_memories"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    memory_type = Column(String)
    content = Column(Text)
    context = Column(JSON, default={})
    importance_score = Column(Float, default=0.5)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=1)
    embedding_vector = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

class LearningPattern(Base):
    __tablename__ = "learning_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    pattern_type = Column(String)
    pattern_data = Column(JSON)
    confidence = Column(Float, default=0.5)
    instances_count = Column(Integer, default=1)
    last_reinforced = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

class ContentRecommendation(Base):
    __tablename__ = "content_recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    content_id = Column(String)
    recommendation_score = Column(Float)
    recommendation_reason = Column(Text, nullable=True)
    presented_at = Column(DateTime, nullable=True)
    user_action = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    metric_type = Column(String)
    metric_value = Column(Float)
    metric_context = Column(JSON, default={})
    measurement_date = Column(DateTime, default=datetime.utcnow)
    session_id = Column(String, nullable=True)

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    create_tables()
    print("Database tables created successfully!")