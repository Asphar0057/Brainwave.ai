import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import json
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import logging
from database.models import SessionLocal, User, UserInteraction, UserPreference, LearningPattern

logger = logging.getLogger(__name__)

class UserProfiler:
    def __init__(self):
        self.db = SessionLocal()
        self.scaler = StandardScaler()
        self.learning_style_classifier = None
        self.difficulty_predictor = None
        self.engagement_model = None
        
        self.feature_extractors = {
            'behavioral': self._extract_behavioral_features,
            'performance': self._extract_performance_features,
            'temporal': self._extract_temporal_features,
            'content': self._extract_content_features,
            'interaction': self._extract_interaction_features
        }
        
        self.profile_dimensions = [
            'learning_style', 'difficulty_preference', 'engagement_level',
            'knowledge_retention', 'help_seeking_behavior', 'persistence_level',
            'preferred_content_length', 'interaction_frequency', 'topic_diversity',
            'cognitive_load_preference'
        ]
        
        self._initialize_models()
    
    def _initialize_models(self):
        try:
            self.learning_style_classifier = joblib.load('models/learning_style_classifier.pkl')
            self.difficulty_predictor = joblib.load('models/difficulty_predictor.pkl')
            self.engagement_model = joblib.load('models/engagement_model.pkl')
        except FileNotFoundError:
            logger.info("Pre-trained models not found. Will train new models.")
            self._train_initial_models()
    
    def _train_initial_models(self):
        self.learning_style_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.difficulty_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.engagement_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        synthetic_data = self._generate_synthetic_training_data()
        self._train_models_with_data(synthetic_data)
    
    async def get_or_create_profile(self, user_id: str) -> Dict:
        existing_profile = await self._load_existing_profile(user_id)
        if existing_profile:
            return existing_profile
        
        initial_profile = self._create_initial_profile()
        await self._save_profile(user_id, initial_profile)
        return initial_profile
    
    async def get_profile(self, user_id: str) -> Dict:
        profile = await self._load_existing_profile(user_id)
        if not profile:
            return await self.get_or_create_profile(user_id)
        return profile
    
    async def update_profile_from_interaction(self, user_id: str, interaction_data: Dict):
        current_profile = await self.get_profile(user_id)
        
        features = await self._extract_interaction_features(user_id, interaction_data)
        
        updated_predictions = await self._update_predictions(current_profile, features, interaction_data)
        
        updated_profile = self._merge_profile_updates(current_profile, updated_predictions)
        
        await self._save_profile(user_id, updated_profile)
        
        await self._update_learning_patterns(user_id, interaction_data, features)
        
        return updated_profile
    
    async def update_profile_from_feedback(self, user_id: str, feedback: Dict):
        current_profile = await self.get_profile(user_id)
        
        feedback_adjustments = self._process_feedback_for_profile(feedback, current_profile)
        
        updated_profile = self._apply_feedback_adjustments(current_profile, feedback_adjustments)
        
        await self._save_profile(user_id, updated_profile)
        
        return updated_profile
    
    async def update_preferences(self, user_id: str, preferences: Dict):
        for pref_type, pref_value in preferences.items():
            await self._store_or_update_preference(user_id, pref_type, pref_value, 0.8)
    
    async def update_subject_proficiency(self, user_id: str, proficiency_data: Dict):
        current_profile = await self.get_profile(user_id)
        expertise_areas = current_profile.get('expertise_areas', {})
        
        subject = proficiency_data['subject']
        confidence = proficiency_data['confidence']
        
        if subject in expertise_areas:
            expertise_areas[subject] = (expertise_areas[subject] + confidence) / 2.0
        else:
            expertise_areas[subject] = confidence
        
        current_profile['expertise_areas'] = expertise_areas
        await self._save_profile(user_id, current_profile)
    
    def _create_initial_profile(self) -> Dict:
        return {
            'learning_style': 'balanced',
            'difficulty_preference': 'medium',
            'engagement_level': 0.5,
            'knowledge_retention': 0.5,
            'help_seeking_behavior': 'moderate',
            'persistence_level': 0.5,
            'preferred_content_length': 'medium',
            'interaction_frequency': 'regular',
            'topic_diversity': 0.5,
            'cognitive_load_preference': 'moderate',
            'expertise_areas': {},
            'weak_areas': {},
            'learning_patterns': {},
            'confidence_scores': {},
            'created_at': datetime.utcnow().isoformat(),
            'last_updated': datetime.utcnow().isoformat(),
            'interaction_count': 0,
            'profile_confidence': 0.1
        }
    
    async def _extract_behavioral_features(self, user_id: str, interaction_data: Dict = None) -> Dict:
        features = {}
        
        interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id
        ).order_by(UserInteraction.timestamp.desc()).limit(50).all()
        
        if not interactions:
            return {'avg_engagement': 0.5, 'engagement_trend': 'stable'}
        
        engagement_scores = []
        for interaction in interactions:
            score = 0.5
            
            if interaction.user_rating:
                score += (interaction.user_rating - 3) * 0.1
            
            if interaction.response_time:
                if interaction.response_time < 30:
                    score += 0.2
                elif interaction.response_time > 120:
                    score -= 0.1
            
            if 'follow up' in interaction.user_input.lower() or 'also' in interaction.user_input.lower():
                score += 0.15
            
            engagement_scores.append(max(0.0, min(1.0, score)))
        
        avg_engagement = np.mean(engagement_scores)
        
        engagement_trend = 'stable'
        if len(engagement_scores) > 10:
            recent_engagement = np.mean(engagement_scores[:5])
            older_engagement = np.mean(engagement_scores[5:10])
            if recent_engagement > older_engagement + 0.1:
                engagement_trend = 'increasing'
            elif recent_engagement < older_engagement - 0.1:
                engagement_trend = 'decreasing'
        
        return {
            'avg_engagement': avg_engagement,
            'engagement_trend': engagement_trend,
            'peak_engagement_times': self._identify_peak_engagement_times(interactions),
            'engagement_factors': self._identify_engagement_factors(interactions)
        }
    
    def _identify_peak_engagement_times(self, interactions) -> List[int]:
        hour_engagement = {}
        for interaction in interactions:
            hour = interaction.timestamp.hour
            rating = interaction.user_rating or 3
            if hour not in hour_engagement:
                hour_engagement[hour] = []
            hour_engagement[hour].append(rating)
        
        avg_by_hour = {hour: np.mean(ratings) for hour, ratings in hour_engagement.items()}
        sorted_hours = sorted(avg_by_hour.items(), key=lambda x: x[1], reverse=True)
        
        return [hour for hour, avg in sorted_hours[:3]]
    
    def _identify_engagement_factors(self, interactions) -> Dict:
        factors = {
            'prefers_short_responses': 0,
            'prefers_examples': 0,
            'prefers_detailed_explanations': 0,
            'responds_well_to_encouragement': 0
        }
        
        high_rated_interactions = [i for i in interactions if i.user_rating and i.user_rating >= 4]
        
        for interaction in high_rated_interactions:
            response = interaction.ai_response or ''
            
            if len(response) < 300:
                factors['prefers_short_responses'] += 1
            if 'example' in response.lower():
                factors['prefers_examples'] += 1
            if len(response) > 800:
                factors['prefers_detailed_explanations'] += 1
            if any(word in response.lower() for word in ['great', 'excellent', 'good job']):
                factors['responds_well_to_encouragement'] += 1
        
        total_high_rated = len(high_rated_interactions)
        if total_high_rated > 0:
            factors = {k: v / total_high_rated > 0.5 for k, v in factors.items()}
        
        return factors
    
    async def continuous_profile_update(self, user_id: str):
        recent_interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id,
            UserInteraction.timestamp >= datetime.utcnow() - timedelta(days=7)
        ).all()
        
        if not recent_interactions:
            return
        
        aggregated_features = await self._extract_aggregated_features(user_id, recent_interactions)
        
        current_profile = await self.get_profile(user_id)
        updated_profile = await self._update_profile_with_aggregated_data(
            current_profile, aggregated_features
        )
        
        await self._save_profile(user_id, updated_profile)
    
    
    async def _update_profile_with_aggregated_data(self, current_profile: Dict, aggregated_features: Dict) -> Dict:
        predictions = await self._update_predictions(current_profile, aggregated_features, {})
        return self._merge_profile_updates(current_profile, predictions)
    
    async def _extract_aggregated_features(self, user_id: str, interactions) -> Dict:
        features = {}
        
        if not interactions:
            return self._default_behavioral_features()
        
        response_times = [i.response_time for i in interactions if i.response_time]
        if response_times:
            features['avg_response_time'] = np.mean(response_times)
            features['response_time_variability'] = np.std(response_times)
        
        complexity_levels = [i.complexity_level for i in interactions if i.complexity_level]
        if complexity_levels:
            features['preferred_complexity'] = max(set(complexity_levels), key=complexity_levels.count)
            features['complexity_diversity'] = len(set(complexity_levels)) / len(complexity_levels)
        
        features['total_interactions'] = len(interactions)
        features['avg_session_length'] = self._calculate_avg_session_length(interactions)
        features['question_type_distribution'] = self._analyze_question_types(interactions)
        
        features['help_frequency'] = len([i for i in interactions if 'help' in i.user_input.lower()]) / len(interactions)
        features['clarification_requests'] = len([i for i in interactions if any(word in i.user_input.lower() for word in ['clarify', 'explain again', 'don\'t understand'])])
        
        follow_up_count = 0
        for i, interaction in enumerate(interactions[:-1]):
            if interactions[i+1].topic == interaction.topic:
                follow_up_count += 1
        features['topic_persistence'] = follow_up_count / max(1, len(interactions) - 1)
        
        features.update(await self._extract_behavioral_features(user_id))
        features.update(await self._extract_performance_features(user_id))
        features.update(await self._extract_temporal_features(user_id))
        features.update(await self._extract_content_features(user_id))
        
        return features
    
    async def _extract_performance_features(self, user_id: str, interaction_data: Dict = None) -> Dict:
        features = {}
        
        interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id
        ).order_by(UserInteraction.timestamp.desc()).limit(100).all()
        
        if not interactions:
            return self._default_performance_features()
        
        ratings = [i.user_rating for i in interactions if i.user_rating]
        if ratings:
            features['avg_rating'] = np.mean(ratings)
            features['rating_trend'] = self._calculate_trend(ratings)
            features['rating_consistency'] = 1.0 - (np.std(ratings) / 5.0)
        
        topic_performance = {}
        for interaction in interactions:
            if interaction.topic and interaction.user_rating:
                if interaction.topic not in topic_performance:
                    topic_performance[interaction.topic] = []
                topic_performance[interaction.topic].append(interaction.user_rating)
        
        features['topic_mastery_scores'] = {
            topic: np.mean(ratings) / 5.0 for topic, ratings in topic_performance.items()
        }
        features['performance_variability'] = np.std(list(features['topic_mastery_scores'].values())) if features['topic_mastery_scores'] else 0
        
        if len(ratings) > 5:
            recent_performance = np.mean(ratings[-5:])
            early_performance = np.mean(ratings[:5])
            features['learning_improvement'] = (recent_performance - early_performance) / 5.0
        
        return features
    
    async def _extract_temporal_features(self, user_id: str, interaction_data: Dict = None) -> Dict:
        features = {}
        
        interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id
        ).order_by(UserInteraction.timestamp.asc()).all()
        
        if len(interactions) < 2:
            return self._default_temporal_features()
        
        time_gaps = []
        for i in range(1, len(interactions)):
            gap = (interactions[i].timestamp - interactions[i-1].timestamp).total_seconds() / 3600
            time_gaps.append(gap)
        
        if time_gaps:
            features['avg_time_between_interactions'] = np.mean(time_gaps)
            features['interaction_regularity'] = 1.0 / (1.0 + np.std(time_gaps))
        
        hours = [interaction.timestamp.hour for interaction in interactions]
        features['preferred_study_time'] = max(set(hours), key=hours.count)
        features['time_consistency'] = len(set(hours)) / 24.0
        
        sessions = self._identify_study_sessions(interactions)
        if sessions:
            session_lengths = [len(session) for session in sessions]
            features['avg_session_interactions'] = np.mean(session_lengths)
            features['session_consistency'] = 1.0 - (np.std(session_lengths) / np.mean(session_lengths))
        
        return features
    
    async def _extract_content_features(self, user_id: str, interaction_data: Dict = None) -> Dict:
        features = {}
        
        interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id
        ).order_by(UserInteraction.timestamp.desc()).limit(100).all()
        
        if not interactions:
            return self._default_content_features()
        
        response_lengths = [len(i.ai_response) for i in interactions if i.ai_response]
        if response_lengths:
            features['preferred_response_length'] = np.mean(response_lengths)
            features['length_tolerance'] = np.std(response_lengths)
        
        topics = [i.topic for i in interactions if i.topic]
        unique_topics = set(topics)
        features['topic_diversity_score'] = len(unique_topics) / max(1, len(topics))
        features['topic_focus_distribution'] = {topic: topics.count(topic) / len(topics) for topic in unique_topics}
        
        complexity_progression = []
        for interaction in interactions:
            if interaction.complexity_level:
                complexity_map = {'easy': 1, 'medium': 2, 'hard': 3}
                complexity_progression.append(complexity_map.get(interaction.complexity_level, 2))
        
        if len(complexity_progression) > 1:
            features['complexity_trend'] = self._calculate_trend(complexity_progression)
        
        return features
    
    async def _extract_interaction_features(self, user_id: str, interaction_data: Dict = None) -> Dict:
        features = {}
        
        if interaction_data:
            features['current_question_length'] = len(interaction_data.get('user_input', ''))
            features['current_topic'] = interaction_data.get('topic', '')
            features['current_complexity'] = interaction_data.get('complexity_level', 'medium')
            features['current_rating'] = interaction_data.get('user_rating', 0)
        
        recent_interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id,
            UserInteraction.timestamp >= datetime.utcnow() - timedelta(days=7)
        ).all()
        
        if recent_interactions:
            features['recent_interaction_count'] = len(recent_interactions)
            features['recent_avg_rating'] = np.mean([i.user_rating for i in recent_interactions if i.user_rating])
            features['recent_topic_focus'] = max([i.topic for i in recent_interactions if i.topic], 
                                               key=[i.topic for i in recent_interactions if i.topic].count) if recent_interactions else None
        
        return features
    
    def _default_behavioral_features(self) -> Dict:
        return {
            'avg_response_time': 30.0,
            'response_time_variability': 10.0,
            'preferred_complexity': 'medium',
            'complexity_diversity': 0.5,
            'total_interactions': 0,
            'avg_session_length': 5,
            'help_frequency': 0.1,
            'clarification_requests': 0,
            'topic_persistence': 0.3
        }
    
    def _default_performance_features(self) -> Dict:
        return {
            'avg_rating': 3.0,
            'rating_trend': 0.0,
            'rating_consistency': 0.5,
            'topic_mastery_scores': {},
            'performance_variability': 0.0,
            'learning_improvement': 0.0
        }
    
    def _default_temporal_features(self) -> Dict:
        return {
            'avg_time_between_interactions': 24.0,
            'interaction_regularity': 0.5,
            'preferred_study_time': 14,
            'time_consistency': 0.5,
            'avg_session_interactions': 5,
            'session_consistency': 0.5
        }
    
    def _default_content_features(self) -> Dict:
        return {
            'preferred_response_length': 500,
            'length_tolerance': 200,
            'topic_diversity_score': 0.5,
            'topic_focus_distribution': {},
            'complexity_trend': 0.0
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        trend = np.polyfit(x, values, 1)[0]
        return trend
    
    def _calculate_avg_session_length(self, interactions) -> float:
        sessions = self._identify_study_sessions(interactions)
        if not sessions:
            return 5.0
        return np.mean([len(session) for session in sessions])
    
    def _identify_study_sessions(self, interactions) -> List[List]:
        if not interactions:
            return []
        
        sessions = []
        current_session = [interactions[0]]
        
        for i in range(1, len(interactions)):
            time_gap = (interactions[i].timestamp - interactions[i-1].timestamp).total_seconds() / 3600
            
            if time_gap <= 2:
                current_session.append(interactions[i])
            else:
                sessions.append(current_session)
                current_session = [interactions[i]]
        
        sessions.append(current_session)
        return sessions
    
    def _analyze_question_types(self, interactions) -> Dict:
        question_types = {}
        for interaction in interactions:
            q_type = interaction.interaction_type or 'general'
            question_types[q_type] = question_types.get(q_type, 0) + 1
        
        total = sum(question_types.values())
        return {k: v/total for k, v in question_types.items()} if total > 0 else {}
    
    async def _update_predictions(self, current_profile: Dict, features: Dict, interaction_data: Dict) -> Dict:
        predictions = {}
        
        feature_vector = self._prepare_feature_vector(features)
        
        try:
            if hasattr(self.learning_style_classifier, 'predict_proba'):
                style_probs = self.learning_style_classifier.predict_proba([feature_vector])[0]
                styles = ['visual', 'auditory', 'kinesthetic', 'reading']
                predictions['learning_style'] = styles[np.argmax(style_probs)]
                predictions['learning_style_confidence'] = np.max(style_probs)
            
            if hasattr(self.difficulty_predictor, 'predict'):
                difficulty_pred = self.difficulty_predictor.predict([feature_vector])[0]
                predictions['difficulty_preference'] = difficulty_pred
            
            if hasattr(self.engagement_model, 'predict'):
                engagement_pred = self.engagement_model.predict([feature_vector])[0]
                predictions['engagement_level'] = max(0.0, min(1.0, engagement_pred))
        
        except Exception as e:
            logger.warning(f"Prediction error: {e}")
            predictions = self._rule_based_predictions(features, interaction_data)
        
        return predictions
    
    def _prepare_feature_vector(self, features: Dict) -> List[float]:
        vector = []
        
        vector.append(features.get('avg_response_time', 30.0))
        vector.append(features.get('response_time_variability', 10.0))
        vector.append(features.get('complexity_diversity', 0.5))
        vector.append(features.get('help_frequency', 0.1))
        vector.append(features.get('topic_persistence', 0.3))
        
        vector.append(features.get('avg_rating', 3.0))
        vector.append(features.get('rating_consistency', 0.5))
        vector.append(features.get('learning_improvement', 0.0))
        
        vector.append(features.get('interaction_regularity', 0.5))
        vector.append(features.get('time_consistency', 0.5))
        
        vector.append(features.get('preferred_response_length', 500) / 1000.0)
        vector.append(features.get('topic_diversity_score', 0.5))
        
        return vector
    
    def _rule_based_predictions(self, features: Dict, interaction_data: Dict) -> Dict:
        predictions = {}
        
        help_freq = features.get('help_frequency', 0.1)
        topic_persistence = features.get('topic_persistence', 0.3)
        response_length_pref = features.get('preferred_response_length', 500)
        
        if response_length_pref < 300:
            predictions['learning_style'] = 'visual'
        elif help_freq > 0.2:
            predictions['learning_style'] = 'auditory'
        elif topic_persistence > 0.5:
            predictions['learning_style'] = 'kinesthetic'
        else:
            predictions['learning_style'] = 'reading'
        
        avg_rating = features.get('avg_rating', 3.0)
        complexity_diversity = features.get('complexity_diversity', 0.5)
        
        if avg_rating > 4.0 and complexity_diversity > 0.6:
            predictions['difficulty_preference'] = 'hard'
        elif avg_rating < 3.0:
            predictions['difficulty_preference'] = 'easy'
        else:
            predictions['difficulty_preference'] = 'medium'
        
        interaction_regularity = features.get('interaction_regularity', 0.5)
        rating_consistency = features.get('rating_consistency', 0.5)
        
        predictions['engagement_level'] = (interaction_regularity + rating_consistency) / 2.0
        
        return predictions
    
    def _merge_profile_updates(self, current_profile: Dict, updates: Dict) -> Dict:
        merged = current_profile.copy()
        
        current_confidence = current_profile.get('profile_confidence', 0.1)
        update_weight = min(0.3, 0.1 + current_confidence)
        
        for key, new_value in updates.items():
            if key in merged:
                if isinstance(new_value, (int, float)):
                    merged[key] = merged[key] * (1 - update_weight) + new_value * update_weight
                else:
                    merged[key] = new_value
            else:
                merged[key] = new_value
        
        merged['last_updated'] = datetime.utcnow().isoformat()
        merged['interaction_count'] = merged.get('interaction_count', 0) + 1
        merged['profile_confidence'] = min(1.0, current_confidence + 0.05)
        
        return merged
    
    def _process_feedback_for_profile(self, feedback: Dict, current_profile: Dict) -> Dict:
        adjustments = {}
        
        rating = feedback.get('rating', 3)
        helpful = feedback.get('helpful', True)
        clarity = feedback.get('clarity', 3)
        
        current_engagement = current_profile.get('engagement_level', 0.5)
        if rating >= 4 and helpful:
            adjustments['engagement_level'] = min(1.0, current_engagement + 0.1)
        elif rating <= 2:
            adjustments['engagement_level'] = max(0.0, current_engagement - 0.1)
        
        if feedback.get('too_easy'):
            adjustments['difficulty_preference'] = self._increase_difficulty(
                current_profile.get('difficulty_preference', 'medium')
            )
        elif feedback.get('too_hard'):
            adjustments['difficulty_preference'] = self._decrease_difficulty(
                current_profile.get('difficulty_preference', 'medium')
            )
        
        if clarity <= 2:
            current_length = current_profile.get('preferred_content_length', 'medium')
            if current_length == 'long':
                adjustments['preferred_content_length'] = 'medium'
            elif current_length == 'medium':
                adjustments['preferred_content_length'] = 'short'
        
        return adjustments
    
    def _increase_difficulty(self, current_difficulty: str) -> str:
        progression = {'easy': 'medium', 'medium': 'hard', 'hard': 'hard'}
        return progression.get(current_difficulty, 'medium')
    
    def _decrease_difficulty(self, current_difficulty: str) -> str:
        regression = {'hard': 'medium', 'medium': 'easy', 'easy': 'easy'}
        return regression.get(current_difficulty, 'medium')
    
    def _apply_feedback_adjustments(self, current_profile: Dict, adjustments: Dict) -> Dict:
        updated = current_profile.copy()
        
        for key, value in adjustments.items():
            updated[key] = value
        
        updated['last_updated'] = datetime.utcnow().isoformat()
        return updated
    
    async def _load_existing_profile(self, user_id: str) -> Optional[Dict]:
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if user and user.profile_data:
            return user.profile_data
        return None
    
    async def _save_profile(self, user_id: str, profile: Dict):
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if user:
            user.profile_data = profile
            user.last_active = datetime.utcnow()
            self.db.commit()
    
    async def _update_learning_patterns(self, user_id: str, interaction_data: Dict, features: Dict):
        pattern_type = f"{interaction_data.get('topic', 'general')}_{interaction_data.get('complexity_level', 'medium')}"
        
        existing_pattern = self.db.query(LearningPattern).filter(
            LearningPattern.user_id == user_id,
            LearningPattern.pattern_type == pattern_type
        ).first()
        
        if existing_pattern:
            existing_pattern.instances_count += 1
            existing_pattern.last_reinforced = datetime.utcnow()
            pattern_data = existing_pattern.pattern_data or {}
            pattern_data.update(features)
            existing_pattern.pattern_data = pattern_data
        else:
            new_pattern = LearningPattern(
                user_id=user_id,
                pattern_type=pattern_type,
                pattern_data=features,
                confidence=0.5,
                instances_count=1,
                last_reinforced=datetime.utcnow()
            )
            self.db.add(new_pattern)
        
        self.db.commit()
    
    async def _store_or_update_preference(self, user_id: str, pref_type: str, pref_value: Any, confidence: float):
        existing_pref = self.db.query(UserPreference).filter(
            UserPreference.user_id == user_id,
            UserPreference.preference_type == pref_type
        ).first()
        
        if existing_pref:
            current_confidence = existing_pref.confidence_score
            current_value = existing_pref.preference_value
            
            total_confidence = current_confidence + confidence
            if isinstance(pref_value, (int, float)) and isinstance(current_value, (int, float)):
                new_value = (current_value * current_confidence + pref_value * confidence) / total_confidence
            else:
                new_value = pref_value if confidence > current_confidence else current_value
            
            existing_pref.preference_value = new_value
            existing_pref.confidence_score = min(1.0, total_confidence / 2.0)
            existing_pref.updated_at = datetime.utcnow()
        else:
            new_pref = UserPreference(
                user_id=user_id,
                preference_type=pref_type,
                preference_value=pref_value,
                confidence_score=confidence,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            self.db.add(new_pref)
        
        self.db.commit()
    
    async def get_performance_metrics(self, user_id: str) -> Dict:
        interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id
        ).all()
        
        if not interactions:
            return {'overall_score': 0.5, 'subject_scores': {}, 'recent_trend': 'stable'}
        
        ratings = [i.user_rating for i in interactions if i.user_rating]
        overall_score = np.mean(ratings) / 5.0 if ratings else 0.5
        
        subject_scores = {}
        for interaction in interactions:
            if interaction.topic and interaction.user_rating:
                if interaction.topic not in subject_scores:
                    subject_scores[interaction.topic] = []
                subject_scores[interaction.topic].append(interaction.user_rating)
        
        subject_scores = {topic: np.mean(scores) / 5.0 for topic, scores in subject_scores.items()}
        
        recent_trend = 'stable'
        if len(ratings) > 10:
            recent_avg = np.mean(ratings[-5:])
            older_avg = np.mean(ratings[-10:-5])
            if recent_avg > older_avg + 0.5:
                recent_trend = 'improving'
            elif recent_avg < older_avg - 0.5:
                recent_trend = 'declining'
        
        return {
            'overall_score': overall_score,
            'subject_scores': subject_scores,
            'recent_trend': recent_trend,
            'total_interactions': len(interactions),
            'avg_session_rating': overall_score
        }
    
    def _generate_synthetic_training_data(self) -> Dict:
        n_samples = 1000
        
        features = np.random.rand(n_samples, 12)
        
        learning_styles = []
        difficulties = []
        engagements = []
        
        for i in range(n_samples):
            if features[i][0] > 0.7:
                learning_styles.append('auditory')
            elif features[i][1] > 0.7:
                learning_styles.append('kinesthetic')
            elif features[i][2] > 0.7:
                learning_styles.append('visual')
            else:
                learning_styles.append('reading')
            
            if features[i][3] > 0.8:
                difficulties.append('hard')
            elif features[i][3] < 0.3:
                difficulties.append('easy')
            else:
                difficulties.append('medium')
            
            engagement = (features[i][4] + features[i][5]) / 2.0
            engagements.append(engagement)
        
        return {
            'features': features,
            'learning_styles': learning_styles,
            'difficulties': difficulties,
            'engagements': engagements
        }
    
    def _train_models_with_data(self, training_data: Dict):
        features = training_data['features']
        
        style_labels = training_data['learning_styles']
        self.learning_style_classifier.fit(features, style_labels)
        
        difficulty_labels = training_data['difficulties']
        self.difficulty_predictor.fit(features, difficulty_labels)
        
        engagement_labels = training_data['engagements']
        self.engagement_model.fit(features, engagement_labels)
        
        joblib.dump(self.learning_style_classifier, 'models/learning_style_classifier.pkl')
        joblib.dump(self.difficulty_predictor, 'models/difficulty_predictor.pkl')
        joblib.dump(self.engagement_model, 'models/engagement_model.pkl')
        
        logger.info("Trained initial feedback models")
    
    async def get_engagement_patterns(self, user_id: str) -> Dict:
        interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id
        ).order_by(UserInteraction.timestamp.desc()).limit(50).all()
        
        if not interactions:
            return {'avg_engagement': 0.5, 'engagement_trend': 'stable'}
        
        engagement_scores = []
        for interaction in interactions:
            score = 0.5
            
            if interaction.user_rating:
                score += (interaction.user_rating - 3) * 0.1
            
            if interaction.response_time:
                if interaction.response_time < 30:
                    score += 0.2
                elif interaction.response_time > 120:
                    score -= 0.1
            
            if 'follow up' in interaction.user_input.lower() or 'also' in interaction.user_input.lower():
                score += 0.15
            
            engagement_scores.append(max(0.0, min(1.0, score)))
        
        avg_engagement = np.mean(engagement_scores)
        
        engagement_trend = 'stable'
        if len(engagement_scores) > 10:
            recent_engagement = np.mean(engagement_scores[:5])
            older_engagement = np.mean(engagement_scores[5:10])
            if recent_engagement > older_engagement + 0.1:
                engagement_trend = 'increasing'
            elif recent_engagement < older_engagement - 0.1:
                engagement_trend = 'decreasing'
        
        return {
            'avg_engagement': avg_engagement,
            'engagement_trend': engagement_trend,
            'peak_engagement_times': self._identify_peak_engagement_times(interactions),
            'engagement_factors': self._identify_engagement_factors(interactions)
        }
    
    def _identify_peak_engagement_times(self, interactions) -> List[int]:
        hour_engagement = {}
        for interaction in interactions:
            hour = interaction.timestamp.hour
            rating = interaction.user_rating or 3
            if hour not in hour_engagement:
                hour_engagement[hour] = []
            hour_engagement[hour].append(rating)
        
        avg_by_hour = {hour: np.mean(ratings) for hour, ratings in hour_engagement.items()}
        sorted_hours = sorted(avg_by_hour.items(), key=lambda x: x[1], reverse=True)
        
        return [hour for hour, avg in sorted_hours[:3]]
    
    def _identify_engagement_factors(self, interactions) -> Dict:
        factors = {
            'prefers_short_responses': 0,
            'prefers_examples': 0,
            'prefers_detailed_explanations': 0,
            'responds_well_to_encouragement': 0
        }
        
        high_rated_interactions = [i for i in interactions if i.user_rating and i.user_rating >= 4]
        
        for interaction in high_rated_interactions:
            response = interaction.ai_response or ''
            
            if len(response) < 300:
                factors['prefers_short_responses'] += 1
            if 'example' in response.lower():
                factors['prefers_examples'] += 1
            if len(response) > 800:
                factors['prefers_detailed_explanations'] += 1
            if any(word in response.lower() for word in ['great', 'excellent', 'good job']):
                factors['responds_well_to_encouragement'] += 1
        
        total_high_rated = len(high_rated_interactions)
        if total_high_rated > 0:
            factors = {k: v / total_high_rated > 0.5 for k, v in factors.items()}
        
        return factors
    
    async def continuous_profile_update(self, user_id: str):
        recent_interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id,
            UserInteraction.timestamp >= datetime.utcnow() - timedelta(days=7)
        ).all()
        
        if not recent_interactions:
            return
        
        aggregated_features = await self._extract_aggregated_features(user_id, recent_interactions)
        
        current_profile = await self.get_profile(user_id)
        updated_profile = await self._update_profile_with_aggregated_data(
            current_profile, aggregated_features
        )
        
        await self._save_profile(user_id, updated_profile)
    
    async def _extract_aggregated_features(self, user_id: str, interactions) -> Dict:
        features = {}
        
        features.update(await self._extract_behavioral_features(user_id))
        features.update(await self._extract_performance_features(user_id))
        features.update(await self._extract_temporal_features(user_id))
        features.update(await self._extract_content_features(user_id))
        
        return features
    
    async def _update_profile_with_aggregated_data(self, current_profile: Dict, aggregated_features: Dict) -> Dict:
        predictions = await self._update_predictions(current_profile, aggregated_features, {})
        return self._merge_profile_updates(current_profile, predictions)