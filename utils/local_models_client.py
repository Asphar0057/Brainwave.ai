import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import asyncio
import gc
import psutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from config import Config

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    content: str
    model_used: str
    response_time: float
    confidence: float

class LocalModelClient:
    def __init__(self):
        self.base_model = None
        self.base_tokenizer = None
        self.math_model = None
        self.math_tokenizer = None
        self.models_loaded = False
        
        self.base_model_name = Config.BASE_MODEL
        self.math_model_name = Config.MATH_MODEL
        
        self.math_keywords = [
            'solve', 'calculate', 'equation', 'formula', 'mathematics', 'math',
            'algebra', 'geometry', 'calculus', 'statistics', 'probability',
            'derivative', 'integral', 'polynomial', 'matrix', 'vector',
            '+', '-', '*', '/', '=', '<', '>', 'x', 'y', 'z'
        ]
        
        self.math_patterns = [
            r'\d+\s*[\+\-\*/\^]\s*\d+',
            r'[xyz]\s*[=<>]\s*\d+',
            r'\b\d+x\b',
            r'\d+\.\d+',
            r'\b\d+/\d+\b'
        ]
    
    def clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def check_memory(self):
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        return available_gb
    
    def load_models(self):
        print("Loading AI models...")
        self.clear_memory()
        
        available_memory = self.check_memory()
        print(f"Available RAM: {available_memory:.1f} GB")
        
        if available_memory < 4:
            print("Warning: Low memory. Models may load slowly.")
        
        try:
            print(f"Loading base model: {self.base_model_name}")
            self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            if torch.cuda.is_available():
                print("Using GPU for base model")
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                print("Using CPU for base model")
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            print("✓ Base model loaded successfully")
            
            print(f"Loading math model: {self.math_model_name}")
            self.math_tokenizer = AutoTokenizer.from_pretrained(self.math_model_name)
            
            if torch.cuda.is_available() and available_memory > 8:
                print("Using GPU for math model")
                self.math_model = AutoModelForCausalLM.from_pretrained(
                    self.math_model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                print("Using CPU for math model (or loading smaller version)")
                try:
                    self.math_model = AutoModelForCausalLM.from_pretrained(
                        self.math_model_name,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                except:
                    print("Math model too large, using base model for math")
                    self.math_model = self.base_model
                    self.math_tokenizer = self.base_tokenizer
            
            print("✓ Math model loaded successfully")
            self.models_loaded = True
            
            if torch.cuda.is_available():
                print(f"GPU memory used: {torch.cuda.memory_allocated() / (1024**3):.1f} GB")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Falling back to smaller models...")
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        try:
            fallback_model = "distilgpt2"
            print(f"Loading fallback model: {fallback_model}")
            
            self.base_tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.base_model = AutoModelForCausalLM.from_pretrained(fallback_model)
            
            self.math_tokenizer = self.base_tokenizer
            self.math_model = self.base_model
            
            if self.base_tokenizer.pad_token is None:
                self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
            
            self.models_loaded = True
            print("✓ Fallback models loaded")
            
        except Exception as e:
            print(f"Failed to load fallback models: {e}")
            self.models_loaded = False
    
    def is_math_query(self, query: str) -> bool:
        query_lower = query.lower()
        
        keyword_score = sum(1 for keyword in self.math_keywords if keyword in query_lower)
        pattern_score = sum(1 for pattern in self.math_patterns if re.search(pattern, query))
        
        total_score = keyword_score + (pattern_score * 2)
        return total_score >= 2
    
    async def generate_response(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        if not self.models_loaded:
            self.load_models()
        
        if not self.models_loaded:
            return LLMResponse(
                content="Sorry, I'm having trouble loading the AI models. Please try again later.",
                model_used="error",
                response_time=0.0,
                confidence=0.0
            )
        
        start_time = asyncio.get_event_loop().time()
        
        is_math = self.is_math_query(prompt)
        
        if is_math:
            model = self.math_model
            tokenizer = self.math_tokenizer
            model_name = "WizardMath"
        else:
            model = self.base_model
            tokenizer = self.base_tokenizer
            model_name = "DialoGPT"
        
        try:
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            else:
                full_prompt = prompt
            
            inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512)
            
            if next(model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9,
                    no_repeat_ngram_size=3
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prompt_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            response = response[len(prompt_text):].strip()
            
            if not response:
                response = "I understand your question, but I'm having trouble generating a complete response. Could you please rephrase it?"
            
            response_time = asyncio.get_event_loop().time() - start_time
            
            return LLMResponse(
                content=response,
                model_used=model_name,
                response_time=response_time,
                confidence=0.8 if is_math else 0.7
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return LLMResponse(
                content="I apologize, but I encountered an error while generating a response. Please try again.",
                model_used=f"{model_name} (error)",
                response_time=asyncio.get_event_loop().time() - start_time,
                confidence=0.0
            )

class ResponseGenerator:
    def __init__(self):
        self.model_client = LocalModelClient()
    
    async def generate(self, prompt: str, context: Any) -> str:
        system_prompt = self._build_system_prompt(context)
        response = await self.model_client.generate_response(prompt, system_prompt)
        return response.content
    
    def _build_system_prompt(self, context) -> str:
        base_prompt = "You are an AI tutor. Be helpful, clear, and educational in your responses."
        
        if hasattr(context, 'learning_style'):
            style_prompts = {
                'visual': "Focus on visual examples, diagrams, and clear formatting.",
                'auditory': "Use conversational language and verbal explanations.",
                'kinesthetic': "Emphasize hands-on examples and practical applications.",
                'reading': "Provide detailed, well-structured text explanations."
            }
            base_prompt += f" {style_prompts.get(context.learning_style, '')}"
        
        return base_prompt