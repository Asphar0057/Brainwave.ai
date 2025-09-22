import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from database.models import SessionLocal, UserInteraction, PerformanceMetric
import logging

logger = logging.getLogger(__name__)

class FeedbackLearningSystem:
    def __init__(self):
        self.db = SessionLocal()
        self.scaler = StandardScaler()
        
        # Machine learning models for different aspects
        self.response_quality_predictor = None
        self.user_preference_classifier = None
        self.difficulty_adjuster = None
        self.engagement_predictor = None
        
        # Feedback processing weights
        self.feedback_weights = {
            'rating': 0.4,
            'helpful': 0.3,
            'clarity': 0.2,
            'accuracy': 0.1
        }
        
        # Model performance tracking
        self.model_performance = {
            'response_quality': {'accuracy': 0.0, 'last_updated': None},
            'user_preference': {'accuracy': 0.0, 'last_updated': None},
            'difficulty_adjustment': {'accuracy': 0.0, 'last_updated': None},
            'engagement': {'accuracy': 0.0, 'last_updated': None}
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize or load existing machine learning models"""
        try:
            self.response_quality_predictor = joblib.load('models/response_quality_model.pkl')
            self.user_preference_classifier = joblib.load('models/user_preference_model.pkl')
            self.difficulty_adjuster = joblib.load('models/difficulty_adjuster_model.pkl')
            self.engagement_predictor = joblib.load('models/engagement_predictor_model.pkl')
            self.scaler = joblib.load('models/feedback_scaler.pkl')
            logger.info("Loaded existing feedback learning models")
        except FileNotFoundError:
            logger.info("No existing models found. Training new models...")
            self._train_initial_models()
    
    def _train_initial_models(self):
        """Train initial models with synthetic data"""
        # Create basic models
        self.response_quality_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.user_preference_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.difficulty_adjuster = RandomForestClassifier(n_estimators=100, random_state=42)
        self.engagement_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Generate synthetic training data
        synthetic_data = self._generate_synthetic_training_data()
        
        # Train models with synthetic data
        self._train_models_with_data(synthetic_data)
        
        # Save initial models
        self._save_models()
        
        logger.info("Initial feedback learning models trained and saved")
    
    def _generate_synthetic_training_data(self) -> Dict:
        """Generate synthetic training data for initial model training"""
        n_samples = 1000
        
        # Feature generation
        features = {
            'query_length': np.random.normal(50, 20, n_samples),
            'response_length': np.random.normal(300, 100, n_samples),
            'complexity_score': np.random.uniform(0, 1, n_samples),
            'topic_familiarity': np.random.uniform(0, 1, n_samples),
            'response_time': np.random.exponential(2, n_samples),
            'model_confidence': np.random.beta(2, 2, n_samples),
            'user_experience_level': np.random.uniform(0, 1, n_samples),
            'previous_ratings': np.random.normal(3.5, 1, n_samples)
        }
        
        # Target generation based on features
        quality_scores = []
        preferences = []
        difficulty_adjustments = []
        engagement_scores = []
        
        for i in range(n_samples):
            # Quality score based on various factors
            quality = (
                0.3 * features['model_confidence'][i] +
                0.2 * (1 - min(features['response_time'][i] / 10, 1)) +
                0.3 * features['topic_familiarity'][i] +
                0.2 * (features['response_length'][i] / 500)
            )
            quality_scores.append(max(0, min(1, quality)))
            
            # User preference (simplified: short vs detailed responses)
            pref = 'detailed' if features['response_length'][i] > 300 else 'concise'
            preferences.append(pref)
            
            # Difficulty adjustment
            if features['previous_ratings'][i] > 4:
                difficulty_adjustments.append('increase')
            elif features['previous_ratings'][i] < 2.5:
                difficulty_adjustments.append('decrease')
            else:
                difficulty_adjustments.append('maintain')
            
            # Engagement score
            engagement = (
                0.4 * quality_scores[i] +
                0.3 * features['user_experience_level'][i] +
                0.3 * features['topic_familiarity'][i]
            )
            engagement_scores.append(max(0, min(1, engagement)))
        
        return {
            'features': features,
            'quality_scores': quality_scores,
            'preferences': preferences,
            'difficulty_adjustments': difficulty_adjustments,
            'engagement_scores': engagement_scores
        }
    
    def _train_models_with_data(self, training_data: Dict):
        """Train models with provided training data"""
        # Prepare feature matrix
        feature_names = list(training_data['features'].keys())
        X = np.column_stack([training_data['features'][name] for name in feature_names])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train response quality predictor
        y_quality = training_data['quality_scores']
        self.response_quality_predictor.fit(X_scaled, y_quality)
        
        # Train user preference classifier
        y_preferences = training_data['preferences']
        self.user_preference_classifier.fit(X_scaled, y_preferences)
        
        # Train difficulty adjuster
        y_difficulty = training_data['difficulty_adjustments']
        self.difficulty_adjuster.fit(X_scaled, y_difficulty)
        
        # Train engagement predictor
        y_engagement = training_data['engagement_scores']
        self.engagement_predictor.fit(X_scaled, y_engagement)
        
        logger.info("Models trained successfully")
    
    async def process_user_feedback(self, user_id: str, interaction_id: int, feedback: Dict) -> bool:
        """Process user feedback and update learning models"""
        try:
            # Store feedback in database
            await self._store_feedback(user_id, interaction_id, feedback)
            
            # Extract features from the interaction
            features = await self._extract_interaction_features(user_id, interaction_id)
            
            # Update models with new feedback
            await self._update_models_with_feedback(features, feedback)
            
            # Update user routing preferences
            await self._update_user_routing_preferences(user_id, feedback, features)
            
            # Retrain models if enough new data is available
            await self._periodic_model_retraining()
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing feedback for user {user_id}: {e}")
            return False
    
    async def _store_feedback(self, user_id: str, interaction_id: int, feedback: Dict):
        """Store feedback data in the database"""
        # Update the interaction record with feedback
        interaction = self.db.query(UserInteraction).filter(
            UserInteraction.id == interaction_id
        ).first()
        
        if interaction:
            interaction.user_rating = feedback.get('rating')
            interaction.feedback_data = feedback
            self.db.commit()
            
            # Store as performance metric
            feedback_score = self._calculate_feedback_score(feedback)
            
            metric = PerformanceMetric(
                user_id=user_id,
                metric_type='feedback_score',
                metric_value=feedback_score,
                metric_context={
                    'interaction_id': interaction_id,
                    'feedback_type': 'user_rating',
                    'rating': feedback.get('rating', 0)
                },
                measurement_date=datetime.utcnow()
            )
            self.db.add(metric)
            self.db.commit()
    
    def _calculate_feedback_score(self, feedback: Dict) -> float:
        """Calculate a composite feedback score"""
        score = 0.0
        total_weight = 0.0
        
        for key, weight in self.feedback_weights.items():
            if key in feedback:
                value = feedback[key]
                if isinstance(value, bool):
                    value = 1.0 if value else 0.0
                elif isinstance(value, int) and key == 'rating':
                    value = value / 5.0  # Normalize rating to 0-1
                
                score += value * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.5
    
    async def _extract_interaction_features(self, user_id: str, interaction_id: int) -> Dict:
        """Extract features from an interaction for model training"""
        interaction = self.db.query(UserInteraction).filter(
            UserInteraction.id == interaction_id
        ).first()
        
        if not interaction:
            return {}
        
        # Recent user history for context
        recent_interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id,
            UserInteraction.timestamp >= datetime.utcnow() - timedelta(days=30)
        ).limit(20).all()
        
        recent_ratings = [i.user_rating for i in recent_interactions if i.user_rating]
        
        features = {
            'query_length': len(interaction.user_input or ''),
            'response_length': len(interaction.ai_response or ''),
            'complexity_score': self._estimate_complexity(interaction.complexity_level),
            'topic_familiarity': self._estimate_topic_familiarity(user_id, interaction.topic),
            'response_time': interaction.response_time or 1.0,
            'model_confidence': getattr(interaction.context_data, 'confidence', 0.5) if interaction.context_data else 0.5,
            'user_experience_level': self._estimate_user_experience(recent_interactions),
            'previous_ratings': np.mean(recent_ratings) if recent_ratings else 3.0
        }
        
        return features
    
    def _estimate_complexity(self, complexity_level: str) -> float:
        """Convert complexity level to numeric score"""
        complexity_map = {'easy': 0.2, 'medium': 0.5, 'hard': 0.8}
        return complexity_map.get(complexity_level, 0.5)
    
    def _estimate_topic_familiarity(self, user_id: str, topic: str) -> float:
        """Estimate user's familiarity with a topic based on history"""
        if not topic:
            return 0.5
        
        topic_interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id,
            UserInteraction.topic == topic
        ).limit(10).all()
        
        if not topic_interactions:
            return 0.2  # Low familiarity for new topics
        
        ratings = [i.user_rating for i in topic_interactions if i.user_rating]
        if ratings:
            return np.mean(ratings) / 5.0
        
        return 0.5
    
    def _estimate_user_experience(self, interactions: List) -> float:
        """Estimate user's overall experience level"""
        if len(interactions) < 5:
            return 0.2  # Beginner
        elif len(interactions) < 20:
            return 0.5  # Intermediate
        else:
            return 0.8  # Experienced
    
    async def _update_models_with_feedback(self, features: Dict, feedback: Dict):
        """Update models with new feedback data"""
        if not features:
            return
        
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            
            # Update response quality predictor
            quality_score = self._calculate_feedback_score(feedback)
            # Note: In a production system, you'd want to accumulate data and retrain periodically
            # For now, we'll just log the data point
            
            logger.info(f"New feedback data point: quality={quality_score}, features={features}")
            
        except Exception as e:
            logger.error(f"Error updating models with feedback: {e}")
    
    def _prepare_feature_vector(self, features: Dict) -> np.ndarray:
        """Convert features dict to numpy array for model input"""
        feature_order = [
            'query_length', 'response_length', 'complexity_score',
            'topic_familiarity', 'response_time', 'model_confidence',
            'user_experience_level', 'previous_ratings'
        ]
        
        vector = [features.get(key, 0.0) for key in feature_order]
        return np.array(vector).reshape(1, -1)
    
    async def _update_user_routing_preferences(self, user_id: str, feedback: Dict, features: Dict):
        """Update user's model routing preferences based on feedback"""
        rating = feedback.get('rating', 3)
        query_type = features.get('query_type', 'general')
        
        # Store routing preference data
        routing_metric = PerformanceMetric(
            user_id=user_id,
            metric_type='routing_preference',
            metric_value=rating / 5.0,
            metric_context={
                'query_type': query_type,
                'model_used': features.get('model_used', 'unknown'),
                'complexity': features.get('complexity_score', 0.5)
            },
            measurement_date=datetime.utcnow()
        )
        self.db.add(routing_metric)
        self.db.commit()
    
    async def get_user_routing_preferences(self, user_id: str) -> Dict:
        """Get user's preferences for model routing"""
        routing_metrics = self.db.query(PerformanceMetric).filter(
            PerformanceMetric.user_id == user_id,
            PerformanceMetric.metric_type == 'routing_preference'
        ).order_by(PerformanceMetric.measurement_date.desc()).limit(50).all()
        
        if not routing_metrics:
            return {
                'math_model_preference': 0.5,
                'general_model_preference': 0.5,
                'math_topics_strength': {},
                'general_topics_strength': {}
            }
        
        math_scores = []
        general_scores = []
        topic_scores = {}
        
        for metric in routing_metrics:
            context = metric.metric_context or {}
            query_type = context.get('query_type', 'general')
            topic = context.get('topic', 'unknown')
            score = metric.metric_value
            
            if query_type == 'math':
                math_scores.append(score)
            else:
                general_scores.append(score)
            
            if topic not in topic_scores:
                topic_scores[topic] = []
            topic_scores[topic].append(score)
        
        # Calculate preferences
        math_preference = np.mean(math_scores) if math_scores else 0.5
        general_preference = np.mean(general_scores) if general_scores else 0.5
        
        # Separate topic strengths by query type
        math_topics = {}
        general_topics = {}
        
        for topic, scores in topic_scores.items():
            avg_score = np.mean(scores)
            # Determine if this topic is typically math-related
            if any(math_word in topic.lower() for math_word in ['math', 'algebra', 'calculus', 'geometry', 'arithmetic']):
                math_topics[topic] = avg_score
            else:
                general_topics[topic] = avg_score
        
        return {
            'math_model_preference': math_preference,
            'general_model_preference': general_preference,
            'math_topics_strength': math_topics,
            'general_topics_strength': general_topics
        }
    
    async def _periodic_model_retraining(self):
        """Periodically retrain models with accumulated feedback data"""
        # Check if enough new data has been collected since last training
        recent_feedback = self.db.query(PerformanceMetric).filter(
            PerformanceMetric.metric_type == 'feedback_score',
            PerformanceMetric.measurement_date >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        if recent_feedback >= 100:  # Retrain if we have 100+ new feedback points
            logger.info("Initiating periodic model retraining...")
            await self._retrain_models()
    
    async def _retrain_models(self):
        """Retrain models with accumulated real user data"""
        try:
            # Get feedback data from the last 6 months
            cutoff_date = datetime.utcnow() - timedelta(days=180)
            
            feedback_data = self.db.query(PerformanceMetric).filter(
                PerformanceMetric.metric_type == 'feedback_score',
                PerformanceMetric.measurement_date >= cutoff_date
            ).all()
            
            if len(feedback_data) < 50:  # Need minimum data for training
                logger.info("Insufficient data for retraining")
                return
            
            # Extract features and targets from real data
            training_data = await self._prepare_retraining_data(feedback_data)
            
            # Retrain models
            self._train_models_with_data(training_data)
            
            # Save updated models
            self._save_models()
            
            # Update performance tracking
            for model_type in self.model_performance:
                self.model_performance[model_type]['last_updated'] = datetime.utcnow()
            
            logger.info("Model retraining completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
    
    async def _prepare_retraining_data(self, feedback_data: List) -> Dict:
        """Prepare real user data for model retraining"""
        features = {
            'query_length': [],
            'response_length': [],
            'complexity_score': [],
            'topic_familiarity': [],
            'response_time': [],
            'model_confidence': [],
            'user_experience_level': [],
            'previous_ratings': []
        }
        
        quality_scores = []
        preferences = []
        difficulty_adjustments = []
        engagement_scores = []
        
        for metric in feedback_data:
            context = metric.metric_context or {}
            interaction_id = context.get('interaction_id')
            
            if interaction_id:
                # Get the original interaction
                interaction = self.db.query(UserInteraction).filter(
                    UserInteraction.id == interaction_id
                ).first()
                
                if interaction:
                    # Extract features
                    interaction_features = await self._extract_interaction_features(
                        metric.user_id, interaction_id
                    )
                    
                    for key in features:
                        features[key].append(interaction_features.get(key, 0.0))
                    
                    # Extract targets
                    quality_scores.append(metric.metric_value)
                    
                    # Determine preference based on response characteristics
                    response_len = len(interaction.ai_response or '')
                    preferences.append('detailed' if response_len > 300 else 'concise')
                    
                    # Determine difficulty adjustment based on rating
                    rating = context.get('rating', 3)
                    if rating >= 4:
                        difficulty_adjustments.append('maintain')
                    elif rating >= 2:
                        difficulty_adjustments.append('maintain')
                    else:
                        difficulty_adjustments.append('decrease')
                    
                    # Use quality score as engagement proxy
                    engagement_scores.append(metric.metric_value)
        
        return {
            'features': features,
            'quality_scores': quality_scores,
            'preferences': preferences,
            'difficulty_adjustments': difficulty_adjustments,
            'engagement_scores': engagement_scores
        }
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            import os
            os.makedirs('models', exist_ok=True)
            
            joblib.dump(self.response_quality_predictor, 'models/response_quality_model.pkl')
            joblib.dump(self.user_preference_classifier, 'models/user_preference_model.pkl')
            joblib.dump(self.difficulty_adjuster, 'models/difficulty_adjuster_model.pkl')
            joblib.dump(self.engagement_predictor, 'models/engagement_predictor_model.pkl')
            joblib.dump(self.scaler, 'models/feedback_scaler.pkl')
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    async def predict_response_quality(self, features: Dict) -> float:
        """Predict the quality of a response before generating it"""
        try:
            if not self.response_quality_predictor:
                return 0.5  # Default prediction
            
            feature_vector = self._prepare_feature_vector(features)
            scaled_features = self.scaler.transform(feature_vector)
            
            prediction = self.response_quality_predictor.predict(scaled_features)[0]
            return max(0.0, min(1.0, prediction))
            
        except Exception as e:
            logger.error(f"Error predicting response quality: {e}")
            return 0.5
    
    async def predict_user_preference(self, features: Dict) -> str:
        """Predict user's response style preference"""
        try:
            if not self.user_preference_classifier:
                return 'detailed'  # Default preference
            
            feature_vector = self._prepare_feature_vector(features)
            scaled_features = self.scaler.transform(feature_vector)
            
            prediction = self.user_preference_classifier.predict(scaled_features)[0]
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting user preference: {e}")
            return 'detailed'
    
    async def suggest_difficulty_adjustment(self, features: Dict) -> str:
        """Suggest difficulty adjustment based on user patterns"""
        try:
            if not self.difficulty_adjuster:
                return 'maintain'  # Default suggestion
            
            feature_vector = self._prepare_feature_vector(features)
            scaled_features = self.scaler.transform(feature_vector)
            
            prediction = self.difficulty_adjuster.predict(scaled_features)[0]
            return prediction
            
        except Exception as e:
            logger.error(f"Error suggesting difficulty adjustment: {e}")
            return 'maintain'
    
    async def predict_engagement(self, features: Dict) -> float:
        """Predict user engagement level for given features"""
        try:
            if not self.engagement_predictor:
                return 0.5  # Default prediction
            
            feature_vector = self._prepare_feature_vector(features)
            scaled_features = self.scaler.transform(feature_vector)
            
            prediction = self.engagement_predictor.predict(scaled_features)[0]
            return max(0.0, min(1.0, prediction))
            
        except Exception as e:
            logger.error(f"Error predicting engagement: {e}")
            return 0.5


class AdaptiveResponseGenerator:
    def __init__(self):
        self.feedback_system = FeedbackLearningSystem()
        
        self.adaptation_strategies = {
            'response_length': self._adapt_response_length,
            'complexity_level': self._adapt_complexity_level,
            'examples_inclusion': self._adapt_examples,
            'encouragement_level': self._adapt_encouragement,
            'formatting_style': self._adapt_formatting
        }
    
    async def generate_adaptive_response(self, query: str, user_id: str, context: Dict) -> Dict:
        """Generate an adaptive response based on user patterns and feedback"""
        base_response = context.get('base_response', '')
        user_profile = context.get('user_profile', {})
        
        # Extract features for prediction
        features = await self._extract_generation_features(query, context, user_profile)
        
        # Get user preferences from feedback system
        user_prefs = await self.feedback_system.get_user_routing_preferences(user_id)
        
        # Predict optimal response characteristics
        predicted_quality = await self.feedback_system.predict_response_quality(features)
        preferred_style = await self.feedback_system.predict_user_preference(features)
        difficulty_suggestion = await self.feedback_system.suggest_difficulty_adjustment(features)
        predicted_engagement = await self.feedback_system.predict_engagement(features)
        
        # Apply adaptations to base response
        adaptations_applied = {}
        adapted_response = base_response
        
        for strategy_name, strategy_func in self.adaptation_strategies.items():
            adaptation_result = await strategy_func(
                adapted_response, features, user_prefs, preferred_style, user_profile
            )
            
            if adaptation_result['modified']:
                adapted_response = adaptation_result['response']
                adaptations_applied[strategy_name] = adaptation_result['description']
        
        return {
            'response': adapted_response,
            'adaptations_applied': adaptations_applied,
            'user_preferences': user_prefs,
            'predicted_quality': predicted_quality,
            'predicted_engagement': predicted_engagement,
            'preferred_style': preferred_style,
            'difficulty_suggestion': difficulty_suggestion
        }
    
    async def _extract_generation_features(self, query: str, context: Dict, user_profile: Dict) -> Dict:
        """Extract features for response generation prediction"""
        memories = context.get('memories', [])
        
        return {
            'query_length': len(query),
            'response_length': len(context.get('base_response', '')),
            'complexity_score': self._estimate_query_complexity(query),
            'topic_familiarity': len(memories) / 10.0,  # Proxy for familiarity
            'response_time': 1.0,  # Default
            'model_confidence': context.get('confidence', 0.5),
            'user_experience_level': user_profile.get('interaction_count', 0) / 100.0,
            'previous_ratings': user_profile.get('average_rating', 3.0)
        }
    
    def _estimate_query_complexity(self, query: str) -> float:
        """Estimate query complexity based on content"""
        complexity_indicators = [
            'analyze', 'evaluate', 'compare', 'synthesize', 'derive',
            'prove', 'optimize', 'calculate', 'determine', 'solve'
        ]
        
        query_lower = query.lower()
        complexity_score = 0.3  # Base complexity
        
        for indicator in complexity_indicators:
            if indicator in query_lower:
                complexity_score += 0.1
        
        # Length-based complexity
        if len(query) > 100:
            complexity_score += 0.2
        
        return min(1.0, complexity_score)
    
    async def _adapt_response_length(self, response: str, features: Dict, user_prefs: Dict, 
                                   style: str, profile: Dict) -> Dict:
        """Adapt response length based on user preferences"""
        current_length = len(response)
        target_length = current_length
        
        if style == 'concise' and current_length > 400:
            # Shorten response
            sentences = response.split('.')
            target_sentences = max(2, len(sentences) // 2)
            adapted_response = '. '.join(sentences[:target_sentences]) + '.'
            
            return {
                'modified': True,
                'response': adapted_response,
                'description': 'Shortened response for concise preference'
            }
        
        elif style == 'detailed' and current_length < 200:
            # Add more detail
            adapted_response = response + "\n\nWould you like me to elaborate on any specific aspect of this topic?"
            
            return {
                'modified': True,
                'response': adapted_response,
                'description': 'Added detail for comprehensive preference'
            }
        
        return {'modified': False, 'response': response, 'description': 'No length adaptation needed'}
    
    async def _adapt_complexity_level(self, response: str, features: Dict, user_prefs: Dict,
                                    style: str, profile: Dict) -> Dict:
        """Adapt complexity level based on user performance patterns"""
        user_experience = features.get('user_experience_level', 0.5)
        
        if user_experience < 0.3:  # Beginner
            # Simplify language and add more explanations
            if any(word in response for word in ['complex', 'sophisticated', 'advanced']):
                adapted_response = response.replace('complex', 'simple')
                adapted_response = adapted_response.replace('sophisticated', 'straightforward')
                adapted_response = adapted_response.replace('advanced', 'basic')
                adapted_response += "\n\nLet me know if you'd like me to explain any of these concepts in more detail!"
                
                return {
                    'modified': True,
                    'response': adapted_response,
                    'description': 'Simplified language for beginner level'
                }
        
        return {'modified': False, 'response': response, 'description': 'No complexity adaptation needed'}
    
    async def _adapt_examples(self, response: str, features: Dict, user_prefs: Dict,
                            style: str, profile: Dict) -> Dict:
        """Add or modify examples based on user learning style"""
        learning_style = profile.get('learning_style', 'balanced')
        
        if learning_style == 'kinesthetic' and 'example' not in response.lower():
            # Add practical example
            adapted_response = response + "\n\nFor example, imagine applying this concept in a real-world scenario..."
            
            return {
                'modified': True,
                'response': adapted_response,
                'description': 'Added practical example for kinesthetic learner'
            }
        
        elif learning_style == 'visual' and not any(char in response for char in ['•', '-', '1.', '2.']):
            # Add visual formatting
            sentences = response.split('. ')
            if len(sentences) > 2:
                formatted_sentences = []
                for i, sentence in enumerate(sentences[:3]):
                    formatted_sentences.append(f"{i+1}. {sentence.strip()}")
                
                adapted_response = '\n'.join(formatted_sentences)
                if len(sentences) > 3:
                    adapted_response += '\n' + '. '.join(sentences[3:])
                
                return {
                    'modified': True,
                    'response': adapted_response,
                    'description': 'Added visual formatting for visual learner'
                }
        
        return {'modified': False, 'response': response, 'description': 'No example adaptation needed'}
    
    async def _adapt_encouragement(self, response: str, features: Dict, user_prefs: Dict,
                                 style: str, profile: Dict) -> Dict:
        """Adapt encouragement level based on user performance"""
        recent_performance = features.get('previous_ratings', 3.0)
        
        if recent_performance < 2.5:  # User struggling
            if not any(word in response.lower() for word in ['great', 'good', 'keep', 'you can']):
                adapted_response = response + "\n\nYou're doing great! Keep asking questions - that's how we learn best."
                
                return {
                    'modified': True,
                    'response': adapted_response,
                    'description': 'Added encouragement for struggling user'
                }
        
        return {'modified': False, 'response': response, 'description': 'No encouragement adaptation needed'}
    
    async def _adapt_formatting(self, response: str, features: Dict, user_prefs: Dict,
                              style: str, profile: Dict) -> Dict:
        """Adapt formatting based on learning style"""
        learning_style = profile.get('learning_style', 'balanced')
        
        if learning_style == 'reading' and len(response) > 300:
            # Add clear paragraph breaks for better readability
            sentences = response.split('. ')
            if len(sentences) > 4:
                mid_point = len(sentences) // 2
                first_part = '. '.join(sentences[:mid_point]) + '.'
                second_part = '. '.join(sentences[mid_point:])
                
                adapted_response = first_part + '\n\n' + second_part
                
                return {
                    'modified': True,
                    'response': adapted_response,
                    'description': 'Added paragraph breaks for better readability'
                }
        
        return {'modified': False, 'response': response, 'description': 'No formatting adaptation needed'}