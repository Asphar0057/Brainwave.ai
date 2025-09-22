import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from database.models import SessionLocal, UserInteraction, PerformanceMetric
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        self.db = SessionLocal()
        
        self.metric_types = {
            'engagement': self._calculate_engagement_metrics,
            'performance': self._calculate_performance_metrics,
            'learning_velocity': self._calculate_learning_velocity,
            'retention': self._calculate_retention_metrics,
            'satisfaction': self._calculate_satisfaction_metrics
        }
    
    async def record_interaction(self, user_id: str, query: str, response: Dict, analysis: Dict):
        metrics = {}
        
        engagement_score = self._calculate_engagement_score(query, response, analysis)
        metrics['engagement'] = engagement_score
        
        performance_score = self._calculate_performance_score(response, analysis)
        metrics['performance'] = performance_score
        
        velocity_score = await self._calculate_velocity_score(user_id, analysis)
        metrics['learning_velocity'] = velocity_score
        
        for metric_type, value in metrics.items():
            db_metric = PerformanceMetric(
                user_id=user_id,
                metric_type=metric_type,
                metric_value=value,
                metric_context={
                    'topic': analysis.get('topic', ''),
                    'complexity': analysis.get('complexity_level', ''),
                    'session_context': response.get('session_id', '')
                },
                measurement_date=datetime.utcnow()
            )
            self.db.add(db_metric)
        
        self.db.commit()
    
    def _calculate_engagement_score(self, query: str, response: Dict, analysis: Dict) -> float:
        score = 0.5
        
        query_length = len(query.split())
        if query_length > 10:
            score += 0.2
        elif query_length < 3:
            score -= 0.1
        
        if any(word in query.lower() for word in ['also', 'furthermore', 'what about', 'how about']):
            score += 0.3
        
        if analysis.get('topic') in response.get('context', {}).get('recent_topics', []):
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_performance_score(self, response: Dict, analysis: Dict) -> float:
        base_score = 0.6
        
        complexity_level = analysis.get('complexity_level', 'medium')
        complexity_bonus = {
            'easy': 0.0,
            'medium': 0.1,
            'hard': 0.2
        }.get(complexity_level, 0.1)
        
        confidence = response.get('confidence_score', 0.5)
        confidence_bonus = (confidence - 0.5) * 0.3
        
        return max(0.0, min(1.0, base_score + complexity_bonus + confidence_bonus))
    
    async def _calculate_velocity_score(self, user_id: str, analysis: Dict) -> float:
        recent_metrics = self.db.query(PerformanceMetric).filter(
            PerformanceMetric.user_id == user_id,
            PerformanceMetric.metric_type == 'performance',
            PerformanceMetric.measurement_date >= datetime.utcnow() - timedelta(days=7)
        ).order_by(PerformanceMetric.measurement_date.desc()).limit(10).all()
        
        if len(recent_metrics) < 3:
            return 0.5
        
        performance_values = [m.metric_value for m in recent_metrics]
        if len(performance_values) > 1:
            trend = np.polyfit(range(len(performance_values)), performance_values, 1)[0]
            velocity_score = 0.5 + (trend * 2)
            return max(0.0, min(1.0, velocity_score))
        
        return 0.5
    
    # Missing methods that need to be implemented
    async def _calculate_engagement_metrics(self, user_id: str, days_back: int = 30) -> Dict:
        """Calculate engagement metrics for a user"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id,
            UserInteraction.timestamp >= cutoff_date
        ).all()
        
        if not interactions:
            return {'average': 0.5, 'trend': 'stable', 'total_sessions': 0}
        
        engagement_scores = []
        for interaction in interactions:
            score = self._calculate_engagement_score(
                interaction.user_input or '',
                {'confidence_score': 0.5},
                {'topic': interaction.topic}
            )
            engagement_scores.append(score)
        
        avg_engagement = np.mean(engagement_scores)
        trend = self._calculate_trend(engagement_scores)
        
        return {
            'average': avg_engagement,
            'trend': 'improving' if trend > 0.1 else 'declining' if trend < -0.1 else 'stable',
            'total_sessions': len(interactions),
            'peak_engagement': max(engagement_scores),
            'low_engagement': min(engagement_scores)
        }
    
    async def _calculate_performance_metrics(self, user_id: str, days_back: int = 30) -> Dict:
        """Calculate performance metrics for a user"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        metrics = self.db.query(PerformanceMetric).filter(
            PerformanceMetric.user_id == user_id,
            PerformanceMetric.metric_type == 'performance',
            PerformanceMetric.measurement_date >= cutoff_date
        ).all()
        
        if not metrics:
            return {'average': 0.5, 'trend': 'stable', 'improvement': 0.0}
        
        performance_values = [m.metric_value for m in metrics]
        avg_performance = np.mean(performance_values)
        trend = self._calculate_trend(performance_values)
        improvement = performance_values[-1] - performance_values[0] if len(performance_values) > 1 else 0.0
        
        return {
            'average': avg_performance,
            'trend': 'improving' if trend > 0.1 else 'declining' if trend < -0.1 else 'stable',
            'improvement': improvement,
            'consistency': 1.0 - np.std(performance_values) if len(performance_values) > 1 else 1.0
        }
    
    async def _calculate_learning_velocity(self, user_id: str, days_back: int = 30) -> Dict:
        """Calculate learning velocity metrics"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        metrics = self.db.query(PerformanceMetric).filter(
            PerformanceMetric.user_id == user_id,
            PerformanceMetric.metric_type == 'learning_velocity',
            PerformanceMetric.measurement_date >= cutoff_date
        ).all()
        
        if not metrics:
            return {'velocity': 0.0, 'acceleration': 0.0, 'learning_rate': 'moderate'}
        
        velocity_values = [m.metric_value for m in metrics]
        avg_velocity = np.mean(velocity_values)
        
        # Calculate acceleration (second derivative)
        acceleration = 0.0
        if len(velocity_values) > 2:
            acceleration = np.gradient(np.gradient(velocity_values))[-1]
        
        learning_rate = 'fast' if avg_velocity > 0.7 else 'slow' if avg_velocity < 0.3 else 'moderate'
        
        return {
            'velocity': avg_velocity,
            'acceleration': acceleration,
            'learning_rate': learning_rate,
            'consistency': 1.0 - np.std(velocity_values) if len(velocity_values) > 1 else 1.0
        }
    
    async def _calculate_retention_metrics(self, user_id: str, days_back: int = 30) -> Dict:
        """Calculate knowledge retention metrics"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id,
            UserInteraction.timestamp >= cutoff_date
        ).order_by(UserInteraction.timestamp.asc()).all()
        
        if not interactions:
            return {'retention_score': 0.5, 'knowledge_decay': 0.0}
        
        # Calculate retention based on repeated topics and performance over time
        topic_performance = {}
        for interaction in interactions:
            topic = interaction.topic or 'general'
            rating = interaction.user_rating or 3
            
            if topic not in topic_performance:
                topic_performance[topic] = []
            topic_performance[topic].append(rating)
        
        retention_scores = []
        for topic, ratings in topic_performance.items():
            if len(ratings) > 1:
                # Check if performance is maintained or improved over time
                trend = self._calculate_trend(ratings)
                retention_score = 0.5 + (trend * 0.5)  # Convert trend to retention score
                retention_scores.append(max(0.0, min(1.0, retention_score)))
        
        avg_retention = np.mean(retention_scores) if retention_scores else 0.5
        
        return {
            'retention_score': avg_retention,
            'knowledge_decay': max(0.0, 0.5 - avg_retention),
            'topics_tracked': len(topic_performance),
            'repeat_learning_needed': avg_retention < 0.4
        }
    
    async def _calculate_satisfaction_metrics(self, user_id: str, days_back: int = 30) -> Dict:
        """Calculate user satisfaction metrics"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        interactions = self.db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id,
            UserInteraction.timestamp >= cutoff_date,
            UserInteraction.user_rating.isnot(None)
        ).all()
        
        if not interactions:
            return {'satisfaction': 0.5, 'trend': 'stable', 'consistency': 0.5}
        
        ratings = [interaction.user_rating for interaction in interactions]
        avg_satisfaction = np.mean(ratings) / 5.0  # Normalize to 0-1
        trend = self._calculate_trend(ratings)
        consistency = 1.0 - (np.std(ratings) / 5.0) if len(ratings) > 1 else 1.0
        
        return {
            'satisfaction': avg_satisfaction,
            'trend': 'improving' if trend > 0.2 else 'declining' if trend < -0.2 else 'stable',
            'consistency': consistency,
            'total_ratings': len(ratings),
            'high_satisfaction_rate': len([r for r in ratings if r >= 4]) / len(ratings)
        }
    
    async def get_user_analytics(self, user_id: str, days_back: int = 30) -> Dict:
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        metrics = self.db.query(PerformanceMetric).filter(
            PerformanceMetric.user_id == user_id,
            PerformanceMetric.measurement_date >= cutoff_date
        ).all()
        
        if not metrics:
            return self._default_analytics()
        
        metrics_by_type = {}
        for metric in metrics:
            if metric.metric_type not in metrics_by_type:
                metrics_by_type[metric.metric_type] = []
            metrics_by_type[metric.metric_type].append(metric.metric_value)
        
        analytics = {}
        
        for metric_type, values in metrics_by_type.items():
            analytics[metric_type] = {
                'average': np.mean(values),
                'trend': self._calculate_trend(values),
                'latest': values[-1] if values else 0.0,
                'improvement': values[-1] - values[0] if len(values) > 1 else 0.0
            }
        
        analytics['overall'] = {
            'total_interactions': len(metrics),
            'active_days': len(set(m.measurement_date.date() for m in metrics)),
            'avg_daily_interactions': len(metrics) / max(1, days_back),
            'performance_trend': analytics.get('performance', {}).get('trend', 0.0),
            'engagement_level': analytics.get('engagement', {}).get('average', 0.5)
        }
        
        return analytics
    
    def _calculate_trend(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        trend = np.polyfit(x, values, 1)[0]
        return trend
    
    def _default_analytics(self) -> Dict:
        return {
            'engagement': {'average': 0.5, 'trend': 0.0, 'latest': 0.5, 'improvement': 0.0},
            'performance': {'average': 0.5, 'trend': 0.0, 'latest': 0.5, 'improvement': 0.0},
            'learning_velocity': {'average': 0.5, 'trend': 0.0, 'latest': 0.5, 'improvement': 0.0},
            'overall': {
                'total_interactions': 0,
                'active_days': 0,
                'avg_daily_interactions': 0.0,
                'performance_trend': 0.0,
                'engagement_level': 0.5
            }
        }