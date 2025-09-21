import re
import numpy as np
from typing import Dict, List, Any, Optional
from ml.user_profiler import UserProfiler
from utils.local_model_client import LocalModelClient
import logging

logger = logging.getLogger(__name__)

class MathRoutingAgent:
    def __init__(self):
        self.model_client = LocalModelClient()
        self.user_profiler = UserProfiler()
        
        self.math_subjects = {
            'arithmetic': ['add', 'subtract', 'multiply', 'divide', '+', '-', '*', '/'],
            'algebra': ['equation', 'variable', 'solve', 'x', 'y', 'polynomial'],
            'geometry': ['angle', 'triangle', 'circle', 'area', 'volume', 'perimeter'],
            'calculus': ['derivative', 'integral', 'limit', 'differentiate'],
            'statistics': ['mean', 'median', 'standard deviation', 'probability'],
            'trigonometry': ['sin', 'cos', 'tan', 'sine', 'cosine', 'tangent']
        }
        
        self.difficulty_indicators = {
            'basic': ['simple', 'easy', 'basic', 'elementary'],
            'intermediate': ['solve', 'calculate', 'find', 'determine'],
            'advanced': ['prove', 'derive', 'optimize', 'evaluate', 'analyze']
        }
    
    async def route_and_respond(self, query: str, user_id: str, user_profile: Dict) -> Dict:
        query_analysis = self._analyze_math_query(query)
        
        if query_analysis['is_math']:
            response = await self._handle_math_query(query, query_analysis, user_profile)
            await self._update_math_proficiency(user_id, query_analysis, response)
        else:
            response = await self._handle_general_query(query, user_profile)
        
        return response
    
    def _analyze_math_query(self, query: str) -> Dict:
        query_lower = query.lower()
        
        math_score = 0
        detected_subjects = []
        difficulty_level = 'basic'
        
        for subject, keywords in self.math_subjects.items():
            if any(keyword in query_lower for keyword in keywords):
                math_score += 2
                detected_subjects.append(subject)
        
        math_patterns = [
            r'\d+\s*[\+\-\*/\^]\s*\d+',
            r'[xyz]\s*[=<>]\s*\d+',
            r'\b\d+x\b',
            r'\bf\(x\)',
            r'\d+\.\d+',
            r'\b\d+/\d+\b',
            r'∫|∑|∏|√|π|θ|α|β|γ'
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, query):
                math_score += 3
        
        for level, indicators in self.difficulty_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                difficulty_level = level
                break
        
        is_math = math_score >= 3
        
        return {
            'is_math': is_math,
            'confidence': min(1.0, math_score / 10.0),
            'subjects': detected_subjects,
            'difficulty': difficulty_level,
            'patterns_found': math_score
        }
    
    async def _handle_math_query(self, query: str, analysis: Dict, user_profile: Dict) -> Dict:
        math_system_prompt = self._build_math_system_prompt(analysis, user_profile)
        
        response = await self.model_client.generate_response(query, math_system_prompt)
        
        enhanced_response = self._enhance_math_response(response.content, analysis, user_profile)
        
        return {
            'response': enhanced_response,
            'model_used': response.model_used,
            'query_type': 'math',
            'subjects': analysis['subjects'],
            'difficulty': analysis['difficulty'],
            'confidence': response.confidence,
            'response_time': response.response_time
        }
    
    async def _handle_general_query(self, query: str, user_profile: Dict) -> Dict:
        general_system_prompt = self._build_general_system_prompt(user_profile)
        
        response = await self.model_client.generate_response(query, general_system_prompt)
        
        return {
            'response': response.content,
            'model_used': response.model_used,
            'query_type': 'general',
            'subjects': ['general'],
            'difficulty': 'medium',
            'confidence': response.confidence,
            'response_time': response.response_time
        }
    
    def _build_math_system_prompt(self, analysis: Dict, user_profile: Dict) -> str:
        base_prompt = "You are a specialized mathematics tutor. Provide clear, step-by-step solutions to mathematical problems."
        
        if analysis['difficulty'] == 'basic':
            base_prompt += " Use simple language and explain each step clearly."
        elif analysis['difficulty'] == 'advanced':
            base_prompt += " Provide detailed mathematical reasoning and proofs where appropriate."
        
        if user_profile.get('learning_style') == 'visual':
            base_prompt += " Use clear formatting and step-by-step layouts."
        elif user_profile.get('learning_style') == 'kinesthetic':
            base_prompt += " Include practical examples and real-world applications."
        
        return base_prompt
    
    def _build_general_system_prompt(self, user_profile: Dict) -> str:
        base_prompt = "You are a helpful AI tutor. Provide clear, educational responses to help students learn."
        
        learning_style = user_profile.get('learning_style', 'balanced')
        if learning_style == 'visual':
            base_prompt += " Use examples and clear formatting."
        elif learning_style == 'auditory':
            base_prompt += " Use conversational language and explanations."
        elif learning_style == 'kinesthetic':
            base_prompt += " Include hands-on examples and practical applications."
        
        return base_prompt
    
    def _enhance_math_response(self, response: str, analysis: Dict, user_profile: Dict) -> str:
        enhanced = response
        
        if analysis['difficulty'] == 'basic' and 'step' not in response.lower():
            enhanced += "\n\nStep-by-step breakdown:\n1. Identify what we need to find\n2. Apply the appropriate method\n3. Calculate the result\n4. Check our answer"
        
        if user_profile.get('learning_style') == 'visual':
            enhanced += "\n\n📊 Visual tip: Try drawing this problem or using diagrams to help visualize the solution."
        
        if analysis['subjects'] and 'algebra' in analysis['subjects']:
            enhanced += "\n\n🔍 Remember: In algebra, we're finding unknown values. Always check your answer by substituting back!"
        
        return enhanced
    
    async def _update_math_proficiency(self, user_id: str, analysis: Dict, response: Dict):
        try:
            for subject in analysis['subjects']:
                proficiency_data = {
                    'subject': subject,
                    'difficulty': analysis['difficulty'],
                    'confidence': response['confidence'],
                    'response_time': response['response_time']
                }
                
                await self.user_profiler.update_subject_proficiency(user_id, proficiency_data)
        except Exception as e:
            logger.error(f"Error updating math proficiency for user {user_id}: {e}")

class QueryRouter:
    def __init__(self):
        self.math_agent = MathRoutingAgent()
    
    async def route_query(self, query: str, user_id: str, user_profile: Dict) -> Dict:
        return await self.math_agent.route_and_respond(query, user_id, user_profile)