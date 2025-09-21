#!/usr/bin/env python3

import os
import sys
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging

from database.models import SessionLocal, User, UserInteraction, create_tables
from agents.math_routing_agent import QueryRouter
from ml.user_profiler import UserProfiler
from ml.feedback_learning_system import FeedbackLearningSystem, AdaptiveResponseGenerator
from utils.memory_manager import MemoryManager
from utils.session_tracker import SessionTracker
from utils.metrics_collector import MetricsCollector

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
        
        create_tables()
        os.makedirs('models', exist_ok=True)
        
    async def initialize_user(self, user_id: str, initial_data: Dict = None) -> Dict:
        user = self.db.query(User).filter(User.user_id == user_id).first()
        
        if not user:
            user = User(
                user_id=user_id,
                created_at=datetime.utcnow(),
                last_active=datetime.utcnow(),
                profile_data=initial_data or {},
                learning_metadata={}
            )
            self.db.add(user)
            self.db.commit()
        
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
        
        routing_response = await self.query_router.route_query(query, user_id, user_profile)
        
        relevant_memories = await self.memory_manager.retrieve_relevant_memories(
            user_id, query, limit=5
        )
        
        context = {
            'base_response': routing_response['response'],
            'query_type': routing_response['query_type'],
            'model_used': routing_response['model_used'],
            'memories': relevant_memories,
            'user_profile': user_profile
        }
        
        adaptive_response = await self.adaptive_generator.generate_adaptive_response(
            query, user_id, context
        )
        
        interaction_id = await self._store_interaction(
            user_id, session_id, query, adaptive_response['response'], routing_response, context
        )
        
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
        
        self.db.add(interaction)
        self.db.commit()
        
        return interaction.id
    
    async def submit_feedback(self, user_id: str, interaction_id: int, feedback: Dict) -> Dict:
        interaction = self.db.query(UserInteraction).filter(
            UserInteraction.id == interaction_id
        ).first()
        
        if not interaction:
            return {'status': 'error', 'message': 'Interaction not found'}
        
        interaction.user_rating = feedback.get('rating')
        interaction.feedback_data = feedback
        self.db.commit()
        
        await self.feedback_system.process_user_feedback(user_id, interaction_id, feedback)
        
        await self.user_profiler.update_profile_from_feedback(user_id, feedback)
        
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
    tutor = BrainwaveAITutor()
    
    print("Brainwave AI Tutor System")
    print("=" * 40)
    print("Features:")
    print("- HuggingFace DialoGPT for general questions")
    print("- WizardMath for mathematical problems")
    print("- Machine Learning personalization")
    print("- Memory and feedback learning")
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
            
            result = await tutor.initialize_user(user_id, initial_data)
            print(f"User initialized: {result}")
            
        elif choice == '2':
            user_id = input("Enter user ID: ")
            question = input("Enter your question: ")
            
            result = await tutor.process_query(user_id, question)
            
            print(f"\nAI Tutor ({result['model_used']}): {result['response']}")
            print(f"Query Type: {result['query_type']}")
            print(f"Confidence: {result['confidence']:.2f}")
            
            rating = input("\nRate this response (1-5): ")
            if rating.isdigit():
                feedback = {
                    'rating': int(rating),
                    'helpful': int(rating) >= 4,
                    'clarity': int(rating),
                    'accuracy': int(rating)
                }
                await tutor.submit_feedback(user_id, result['interaction_id'], feedback)
                print("Feedback recorded!")
                
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
            
            result = await tutor.submit_feedback(user_id, int(interaction_id), feedback)
            print(f"Feedback result: {result}")
            
        elif choice == '4':
            user_id = input("Enter user ID: ")
            analytics = await tutor.get_user_analytics(user_id)
            
            print(f"\nAnalytics for {user_id}:")
            print(f"Profile: {analytics['profile']}")
            print(f"Performance: {analytics['performance']}")
            print(f"Recommendations: {analytics['recommendations']}")
            
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    asyncio.run(main())