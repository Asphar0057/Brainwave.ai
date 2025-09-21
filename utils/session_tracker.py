import uuid
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from database.models import SessionLocal, LearningSession
import logging

logger = logging.getLogger(__name__)

class SessionTracker:
    def __init__(self):
        self.db = SessionLocal()
        self.active_sessions = {}
        self.session_timeout = 3600
    
    def get_or_create_session(self, user_id: str) -> str:
        if user_id in self.active_sessions:
            session_data = self.active_sessions[user_id]
            if datetime.utcnow() - session_data['last_activity'] < timedelta(seconds=self.session_timeout):
                session_data['last_activity'] = datetime.utcnow()
                return session_data['session_id']
        
        session_id = str(uuid.uuid4())
        
        db_session = LearningSession(
            user_id=user_id,
            session_id=session_id,
            start_time=datetime.utcnow(),
            session_metadata={'created_by': 'session_tracker'}
        )
        self.db.add(db_session)
        self.db.commit()
        
        self.active_sessions[user_id] = {
            'session_id': session_id,
            'start_time': datetime.utcnow(),
            'last_activity': datetime.utcnow(),
            'interaction_count': 0,
            'topics_covered': set(),
            'performance_scores': []
        }
        
        return session_id
    
    def update_session_activity(self, user_id: str, activity_data: Dict):
        if user_id in self.active_sessions:
            session = self.active_sessions[user_id]
            session['last_activity'] = datetime.utcnow()
            session['interaction_count'] += 1
            
            if 'topic' in activity_data:
                session['topics_covered'].add(activity_data['topic'])
            
            if 'performance_score' in activity_data:
                session['performance_scores'].append(activity_data['performance_score'])
    
    def end_session(self, user_id: str):
        if user_id in self.active_sessions:
            session_data = self.active_sessions[user_id]
            
            db_session = self.db.query(LearningSession).filter(
                LearningSession.session_id == session_data['session_id']
            ).first()
            
            if db_session:
                db_session.end_time = datetime.utcnow()
                db_session.interaction_count = session_data['interaction_count']
                db_session.session_metadata = {
                    'topics_covered': list(session_data['topics_covered']),
                    'avg_performance': np.mean(session_data['performance_scores']) if session_data['performance_scores'] else 0.0,
                    'total_interactions': session_data['interaction_count']
                }
                self.db.commit()
            
            del self.active_sessions[user_id]
    
    def get_session_stats(self, user_id: str) -> Dict:
        if user_id not in self.active_sessions:
            return {}
        
        session = self.active_sessions[user_id]
        duration = (datetime.utcnow() - session['start_time']).total_seconds() / 60
        
        return {
            'session_id': session['session_id'],
            'duration_minutes': duration,
            'interaction_count': session['interaction_count'],
            'topics_covered': list(session['topics_covered']),
            'avg_performance': np.mean(session['performance_scores']) if session['performance_scores'] else 0.0,
            'interactions_per_minute': session['interaction_count'] / max(1, duration)
        }