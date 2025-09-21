#!/usr/bin/env python3

import os
import sys
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Add the current directory to Python path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from database.models import SessionLocal, User, UserInteraction, create_tables
except ImportError as e:
    print(f"Database import error: {e}")
    print("Creating mock database classes...")
    
    class MockDB:
        def query(self, *args): return self
        def filter(self, *args): return self
        def first(self): return None
        def all(self): return []
        def add(self, *args): pass
        def commit(self): pass
        def count(self): return 0
    
    class MockUser:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class MockUserInteraction:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.id = 1
    
    SessionLocal = MockDB
    User = MockUser
    UserInteraction = MockUserInteraction
    create_tables = lambda: None

try:
    from agents.math_routing_agent import QueryRouter
except ImportError:
    print("QueryRouter import failed, creating mock...")
    class QueryRouter:
        async def route_query(self, query, user_id, profile):
            return {
                'response': f"Mock response for: {query}",
                'query_type': 'general',
                'model_used': 'mock_model',
                'confidence': 0.8,
                'subjects': ['general'],
                'difficulty': 'medium',
                'response_time': 1.0
            }

try:
    from ml.user_profiler import UserProfiler
except ImportError:
    print("UserProfiler import failed, creating mock...")
    class UserProfiler:
        async def get_or_create_profile(self, user_id):
            return {'user_id': user_id, 'preferences': {}}
        async def get_profile(self, user_id):
            return {'user_id': user_id, 'preferences': {}}
        async def update_preferences(self, user_id, prefs):
            pass
        async def update_profile_from_feedback(self, user_id, feedback):
            pass
        async def get_performance_metrics(self, user_id):
            return {'average': 0.7, 'improvement': 0.1}

# Import the classes from the feedback learning system
try:
    # Try to import from the ml module
    from ml.feedback_learning_system import FeedbackLearningSystem, AdaptiveResponseGenerator
except ImportError as e:
    print(f"FeedbackLearningSystem import failed: {e}")
    print("Creating mock classes...")
    
    class FeedbackLearningSystem:
        def __init__(self):
            self.db = SessionLocal()
            self.models_trained = False
        
        async def process_user_feedback(self, user_id, interaction_id, feedback):
            print(f"Processing feedback for user {user_id}, interaction {interaction_id}")
            return True
        
        async def get_user_routing_preferences(self, user_id):
            return {
                'math_model_preference': 0.5,
                'general_model_preference': 0.5,
                'math_topics_strength': {},
                'general_topics_strength': {}
            }
    
    class AdaptiveResponseGenerator:
        def __init__(self):
            self.feedback_system = FeedbackLearningSystem()
        
        async def generate_adaptive_response(self, query, user_id, context):
            return {
                'response': context.get('base_response', 'Adaptive response'),
                'adaptations_applied': {},
                'user_preferences': {}
            }

try:
    from utils.memory_manager import MemoryManager
except ImportError:
    print("MemoryManager import failed, creating mock...")
    class MemoryManager:
        async def retrieve_relevant_memories(self, user_id, query, limit=5):
            return []
        async def store_interaction(self, user_id, query, response, context, metadata):
            pass
        async def mark_interaction_as_successful(self, user_id, interaction_id):
            pass

try:
    from utils.session_tracker import SessionTracker
except ImportError:
    print("SessionTracker import failed, creating mock...")
    class SessionTracker:
        def __init__(self):
            self.sessions = {}
        def get_or_create_session(self, user_id):
            if user_id not in self.sessions:
                self.sessions[user_id] = f"session_{user_id}_{datetime.now().timestamp()}"
            return self.sessions[user_id]
        def end_session(self, user_id):
            if user_id in self.sessions:
                del self.sessions[user_id]

try:
    from utils.metrics_collector import MetricsCollector
except ImportError:
    print("MetricsCollector import failed, creating mock...")
    class MetricsCollector:
        async def get_user_analytics(self, user_id):
            return {
                'performance': {'average': 0.6},
                'engagement': {'average': 0.7},
                'total_interactions': 10
            }

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainwaveAITutor:
    def __init__(self):
        self.db = SessionLocal()
        self.query_router = QueryRouter()
        self.user_profiler = UserProfiler()
        self.feedback_system = FeedbackLearningSystem()
        self.adaptive_generator = AdaptiveResponseGenerator()
        self.memory_manager = MemoryManager()
        self.session_tracker = SessionTracker()
        self.metrics_collector = MetricsCollector()
        
        self.active_users = {}
        
        # Create tables and directories
        try:
            create_tables()
        except Exception as e:
            print(f"Table creation failed: {e}")
        
        os.makedirs('models', exist_ok=True)
        
    async def initialize_user(self, user_id: str, initial_data: Dict = None) -> Dict:
        try:
            user = self.db.query(User).filter(User.user_id == user_id).first()
        except:
            user = None
        
        if not user:
            user = User(
                user_id=user_id,
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow(),
                profile_data=initial_data or {},
                learning_metadata={}
            )
            try:
                self.db.add(user)
                self.db.commit()
            except Exception as e:
                print(f"Database error: {e}")
        
        user_profile = await self.user_profiler.get_or_create_profile(user_id)
        
        if initial_data:
            await self._process_initial_data(user_id, initial_data)
        
        self.active_users[user_id] = {
            'profile': user_profile,
            'session_start': datetime.utcnow(),
            'interaction_count': 0
        }
        
        return {
            'user_id': user_id,
            'profile': user_profile,
            'status': 'initialized'
        }
    
    async def _process_initial_data(self, user_id: str, data: Dict):
        preferences = {
            'learning_style': data.get('learning_style', 'balanced'),
            'difficulty_preference': data.get('difficulty', 'medium'),
            'subjects_of_interest': data.get('subjects', []),
            'time_availability': data.get('time_availability', 30)
        }
        
        await self.user_profiler.update_preferences(user_id, preferences)
    
    async def process_query(self, user_id: str, query: str) -> Dict:
        if user_id not in self.active_users:
            await self.initialize_user(user_id)
        
        session_id = self.session_tracker.get_or_create_session(user_id)
        user_profile = self.active_users[user_id]['profile']
        
        start_time = datetime.utcnow()
        
        # Route the query through the routing agent
        routing_response = await self.query_router.route_query(query, user_id, user_profile)
        
        # Retrieve relevant memories
        relevant_memories = await self.memory_manager.retrieve_relevant_memories(
            user_id, query, limit=5
        )
        
        # Build context for adaptive response generation
        context = {
            'base_response': routing_response['response'],
            'query_type': routing_response['query_type'],
            'model_used': routing_response['model_used'],
            'memories': relevant_memories,
            'user_profile': user_profile
        }
        
        # Generate adaptive response
        adaptive_response = await self.adaptive_generator.generate_adaptive_response(
            query, user_id, context
        )
        
        # Store the interaction
        interaction_id = await self._store_interaction(
            user_id, session_id, query, adaptive_response['response'], routing_response, context
        )
        
        # Store in memory for future reference
        await self.memory_manager.store_interaction(
            user_id, query, adaptive_response['response'], context, {
                'session_id': session_id,
                'interaction_id': interaction_id,
                'model_used': routing_response['model_used']
            }
        )
        
        self.active_users[user_id]['interaction_count'] += 1
        
        return {
            'response': adaptive_response['response'],
            'interaction_id': interaction_id,
            'model_used': routing_response['model_used'],
            'query_type': routing_response['query_type'],
            'confidence': routing_response['confidence'],
            'adaptations': adaptive_response.get('adaptations_applied', {}),
            'session_id': session_id
        }
    
    async def _store_interaction(self, user_id: str, session_id: str, query: str, 
                                response: str, routing_data: Dict, context: Dict) -> int:
        interaction = UserInteraction(
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.utcnow(),
            interaction_type=routing_data['query_type'],
            user_input=query,
            ai_response=response,
            topic=routing_data.get('subjects', ['general'])[0],
            complexity_level=routing_data.get('difficulty', 'medium'),
            response_time=routing_data.get('response_time', 0),
            context_data={
                'model_used': routing_data['model_used'],
                'confidence': routing_data['confidence'],
                'memories_used': len(context.get('memories', []))
            }
        )
        
        try:
            self.db.add(interaction)
            self.db.commit()
            return getattr(interaction, 'id', 1)  # Return 1 as fallback for mock DB
        except Exception as e:
            print(f"Database error storing interaction: {e}")
            return 1  # Return mock ID
    
    async def submit_feedback(self, user_id: str, interaction_id: int, feedback: Dict) -> Dict:
        try:
            interaction = self.db.query(UserInteraction).filter(
                UserInteraction.id == interaction_id
            ).first()
        except:
            interaction = None
        
        if not interaction:
            return {'status': 'error', 'message': 'Interaction not found'}
        
        try:
            interaction.user_rating = feedback.get('rating')
            interaction.feedback_data = feedback
            self.db.commit()
        except Exception as e:
            print(f"Database error updating feedback: {e}")
        
        # Process feedback through the learning system
        await self.feedback_system.process_user_feedback(user_id, interaction_id, feedback)
        
        # Update user profile based on feedback
        await self.user_profiler.update_profile_from_feedback(user_id, feedback)
        
        # Mark successful interactions in memory
        if feedback.get('rating', 0) >= 4:
            await self.memory_manager.mark_interaction_as_successful(user_id, interaction_id)
        
        return {'status': 'success', 'message': 'Feedback processed'}
    
    async def get_user_analytics(self, user_id: str) -> Dict:
        analytics = await self.metrics_collector.get_user_analytics(user_id)
        
        user_profile = await self.user_profiler.get_profile(user_id)
        performance_metrics = await self.user_profiler.get_performance_metrics(user_id)
        
        return {
            'profile': user_profile,
            'performance': performance_metrics,
            'metrics': analytics,
            'recommendations': await self._generate_recommendations(user_id, analytics)
        }
    
    async def _generate_recommendations(self, user_id: str, analytics: Dict) -> List[str]:
        recommendations = []
        
        overall_performance = analytics.get('performance', {}).get('average', 0.5)
        if overall_performance < 0.4:
            recommendations.append("Consider reviewing fundamental concepts")
        
        engagement = analytics.get('engagement', {}).get('average', 0.5)
        if engagement < 0.4:
            recommendations.append("Try varying your study approach for better engagement")
        
        math_prefs = await self.feedback_system.get_user_routing_preferences(user_id)
        if math_prefs['math_model_preference'] < 0.3:
            recommendations.append("Focus on building stronger math fundamentals")
        
        return recommendations
    
    def cleanup_user_session(self, user_id: str):
        if user_id in self.active_users:
            self.session_tracker.end_session(user_id)
            del self.active_users[user_id]

async def main():
    print("Initializing Brainwave AI Tutor System...")
    
    try:
        tutor = BrainwaveAITutor()
        print("✓ System initialized successfully!")
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        return
    
    print("Brainwave AI Tutor System")
    print("=" * 40)
    print("Features:")
    print("- Intelligent query routing")
    print("- Machine Learning personalization")
    print("- Memory and feedback learning")
    print("- Adaptive response generation")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Initialize User")
        print("2. Ask Question")
        print("3. Submit Feedback")
        print("4. View Analytics")
        print("5. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            user_id = input("Enter user ID: ")
            learning_style = input("Learning style (visual/auditory/kinesthetic/reading): ") or "balanced"
            subjects = input("Subjects of interest (comma-separated): ").split(',')
            difficulty = input("Preferred difficulty (easy/medium/hard): ") or "medium"
            
            initial_data = {
                'learning_style': learning_style,
                'subjects': [s.strip() for s in subjects if s.strip()],
                'difficulty': difficulty,
                'time_availability': 30
            }
            
            try:
                result = await tutor.initialize_user(user_id, initial_data)
                print(f"✓ User initialized: {result}")
            except Exception as e:
                print(f"✗ User initialization error: {e}")
            
        elif choice == '2':
            user_id = input("Enter user ID: ")
            question = input("Enter your question: ")
            
            try:
                result = await tutor.process_query(user_id, question)
                
                print(f"\n🤖 AI Tutor ({result['model_used']}): {result['response']}")
                print(f"📊 Query Type: {result['query_type']}")
                print(f"🎯 Confidence: {result['confidence']:.2f}")
                if result.get('adaptations'):
                    print(f"🔧 Adaptations: {result['adaptations']}")
                
                # Auto-prompt for feedback
                rating = input("\nRate this response (1-5) or press Enter to skip: ")
                if rating.isdigit():
                    feedback = {
                        'rating': int(rating),
                        'helpful': int(rating) >= 4,
                        'clarity': int(rating),
                        'accuracy': int(rating)
                    }
                    feedback_result = await tutor.submit_feedback(user_id, result['interaction_id'], feedback)
                    print(f"✓ {feedback_result['message']}")
                    
            except Exception as e:
                print(f"✗ Query processing error: {e}")
                
        elif choice == '3':
            user_id = input("Enter user ID: ")
            interaction_id = input("Enter interaction ID: ")
            rating = input("Rating (1-5): ")
            helpful = input("Was it helpful? (y/n): ").lower() == 'y'
            text_feedback = input("Additional feedback (optional): ")
            
            feedback = {
                'rating': int(rating) if rating.isdigit() else 3,
                'helpful': helpful,
                'clarity': int(rating) if rating.isdigit() else 3,
                'accuracy': int(rating) if rating.isdigit() else 3,
                'text_feedback': text_feedback
            }
            
            try:
                result = await tutor.submit_feedback(user_id, int(interaction_id), feedback)
                print(f"✓ Feedback result: {result}")
            except Exception as e:
                print(f"✗ Feedback submission error: {e}")
            
        elif choice == '4':
            user_id = input("Enter user ID: ")
            
            try:
                analytics = await tutor.get_user_analytics(user_id)
                
                print(f"\n📈 Analytics for {user_id}:")
                print(f"👤 Profile: {analytics['profile']}")
                print(f"📊 Performance: {analytics['performance']}")
                print(f"💡 Recommendations: {analytics['recommendations']}")
            except Exception as e:
                print(f"✗ Analytics error: {e}")
            
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    asyncio.run(main())