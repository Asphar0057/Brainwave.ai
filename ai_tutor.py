# Simple AI Tutor with ChromaDB Memory - VSCode Starter
# Run this step by step to build your system

import chromadb
import ollama
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
import sqlite3
import re

# ===================== SETUP =====================

def setup_environment():
    """First-time setup - run this once"""
    print("Setting up AI Tutor environment...")
    
    # Create directories
    os.makedirs("./ai_tutor_data", exist_ok=True)
    os.makedirs("./ai_tutor_data/chroma_db", exist_ok=True)
    
    print("Directories created")
    print("Next steps:")
    print("1. Install dependencies: pip install chromadb")
    print("2. Make sure Ollama is running: ollama serve")
    print("3. You already have qwen3 - great!")

# ===================== ZERO-KNOWLEDGE LEARNING SYSTEM =====================

class ZeroKnowledgeAITutor:
    """AI Tutor that starts knowing nothing and learns from every interaction"""
    
    def __init__(self):
        # Initialize ChromaDB for memory storage
        self.chroma_client = chromadb.PersistentClient(
            path="./ai_tutor_data/chroma_db"
        )
        
        # Create user-specific collections (each user gets their own memory)
        self.user_collections = {}
        
        # Simple user profile storage
        self.user_db_path = "./ai_tutor_data/users.db"
        self._init_user_database()
        
        # Machine learning adaptation data
        self.learning_patterns = {}
        
        print("Zero-Knowledge AI Tutor initialized")
        print("Each user starts with blank memory")
        print("System learns from every interaction")
    
    def _init_user_database(self):
        """Initialize user profiles and learning data"""
        with sqlite3.connect(self.user_db_path) as conn:
            # User profiles
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_interactions INTEGER DEFAULT 0,
                    learning_velocity REAL DEFAULT 0.0,
                    preferred_response_length TEXT DEFAULT 'medium',
                    adaptation_score REAL DEFAULT 0.0
                )
            """)
            
            # Interaction log for machine learning
            conn.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    question TEXT,
                    answer TEXT,
                    user_rating REAL,
                    response_time REAL,
                    question_complexity REAL,
                    answer_length INTEGER,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    improvement_feedback TEXT
                )
            """)
    
    def user_exists(self, user_id: str) -> bool:
        """Check if user has existing data"""
        try:
            # Check if user has ChromaDB collection
            collection_name = f"user_{user_id}_memory"
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
                if collection.count() > 0:
                    return True
            except:
                pass
            
            # Check if user has SQL records
            with sqlite3.connect(self.user_db_path) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM interactions WHERE user_id = ?
                """, (user_id,))
                count = cursor.fetchone()[0]
                return count > 0
        except:
            return False
    
    def get_or_create_user_memory(self, user_id: str):
        """Create personal memory collection for user"""
        if user_id not in self.user_collections:
            try:
                # Each user gets their own ChromaDB collection
                collection_name = f"user_{user_id}_memory"
                
                # Check if this is truly a new user
                is_new_user = not self.user_exists(user_id)
                
                self.user_collections[user_id] = self.chroma_client.get_or_create_collection(
                    name=collection_name,
                    metadata={"user_id": user_id, "created": datetime.now().isoformat()}
                )
                
                # Initialize user in database only if new
                with sqlite3.connect(self.user_db_path) as conn:
                    conn.execute("""
                        INSERT OR IGNORE INTO users (user_id) VALUES (?)
                    """, (user_id,))
                
                if is_new_user:
                    print(f"Created new memory for user: {user_id}")
                else:
                    print(f"Loaded existing memory for user: {user_id}")
                
            except Exception as e:
                print(f"Error creating user memory: {e}")
                return None
        
        return self.user_collections[user_id]
    
    def simple_embeddings(self, text: str) -> List[float]:
        """Generate simple embeddings using available models"""
        try:
            # Try to use nomic-embed-text directly (don't auto-pull due to encoding issues)
            response = ollama.embeddings(model="nomic-embed-text", prompt=text)
            return response['embedding']
        except:
            pass
        
        try:
            # Fallback: Use one of your existing models for embeddings
            response = ollama.embeddings(model="llama3:latest", prompt=text)
            return response['embedding']
        except:
            # Final fallback: create simple hash-based embedding for development
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            # Convert to simple 384-dim vector
            hash_bytes = hash_obj.hexdigest().encode()
            return [float(b) / 255.0 for b in hash_bytes[:384]] + [0.0] * (384 - len(hash_bytes))
    
    def analyze_user_pattern(self, user_id: str, question: str, feedback: Optional[float] = None):
        """Machine Learning: Analyze user patterns and adapt"""
        try:
            # Get user's interaction history
            with sqlite3.connect(self.user_db_path) as conn:
                cursor = conn.execute("""
                    SELECT question, user_rating, question_complexity, answer_length 
                    FROM interactions 
                    WHERE user_id = ? AND user_rating IS NOT NULL
                    ORDER BY timestamp DESC LIMIT 10
                """, (user_id,))
                
                recent_interactions = cursor.fetchall()
            
            if len(recent_interactions) < 3:
                return "learning"  # Not enough data yet
            
            # Calculate learning patterns
            avg_rating = sum(r[1] for r in recent_interactions) / len(recent_interactions)
            preferred_complexity = sum(r[2] for r in recent_interactions) / len(recent_interactions)
            preferred_length = sum(r[3] for r in recent_interactions) / len(recent_interactions)
            
            # Adapt response style based on patterns
            adaptation = {
                "response_style": "detailed" if preferred_length > 150 else "concise",
                "complexity_preference": "high" if preferred_complexity > 0.7 else "medium",
                "satisfaction_trend": "improving" if avg_rating > 3.5 else "needs_adjustment",
                "adaptation_confidence": min(len(recent_interactions) / 10.0, 1.0)
            }
            
            # Store learning pattern
            self.learning_patterns[user_id] = adaptation
            
            return adaptation
            
        except Exception as e:
            print(f"Error analyzing patterns: {e}")
            return {"response_style": "medium", "complexity_preference": "medium"}
    
    def _clean_response(self, response_text: str) -> str:
        """Remove thinking tags and clean response"""
        if not response_text:
            return "I apologize, but I wasn't able to generate a proper response. Please try asking your question again."
        
        # Remove <think>...</think> blocks only
        cleaned = re.sub(r'<think>.*?</think>\s*', '', response_text, flags=re.DOTALL)
        
        # Only remove obvious thinking patterns at the start of lines, not mid-content
        cleaned = re.sub(r'^\s*Let me think.*?\n', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^\s*I need to think.*?\n', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^\s*Okay, let me.*?\n', '', cleaned, flags=re.MULTILINE)
        
        result = cleaned.strip()
        
        # Debug: Let's see what we're getting
        if not result:
            print(f"DEBUG: Original response: {response_text[:200]}...")
            print(f"DEBUG: Cleaned response: '{result}'")
            return "I apologize, but I wasn't able to generate a proper response. Please try asking your question again."
        
        return result
    
    def chat(self, user_id: str, question: str) -> Dict[str, Any]:
        """Main chat function - learns from every interaction"""
        
        # Get or create user's personal memory
        user_memory = self.get_or_create_user_memory(user_id)
        if not user_memory:
            return {"error": "Could not create user memory"}
        
        print(f"\nUser {user_id} asks: {question}")
        
        # Step 1: Search user's personal memory for relevant context
        relevant_context = []
        try:
            if user_memory.count() > 0:  # Only search if user has previous conversations
                query_embedding = self.simple_embeddings(question)
                
                memory_results = user_memory.query(
                    query_embeddings=[query_embedding],
                    n_results=min(3, user_memory.count())  # Get up to 3 relevant memories
                )
                
                if memory_results['documents'] and memory_results['documents'][0]:
                    relevant_context = memory_results['documents'][0]
                    print(f"Found {len(relevant_context)} relevant memories")
            else:
                print("No previous memories found")
        except Exception as e:
            print(f"Memory search failed: {e}")
        
        # Step 2: Get user's learning pattern (ML adaptation)
        user_pattern = self.analyze_user_pattern(user_id, question)
        
        # Step 3: Build adaptive prompt based on user's learning history
        prompt = self._build_adaptive_prompt(question, relevant_context, user_pattern)
        
        # Step 4: Generate response using Qwen3
        try:
            start_time = datetime.now()
            
            response = ollama.generate(
                model="qwen3:14b",
                prompt=prompt,
                options={
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "stop": ["<think>", "</think>"],  # Stop generation at thinking tags
                    "num_predict": 300  # Limit response length for faster responses
                }
            )
            
            # Get the raw response
            raw_answer = response.get('response', '')
            response_time = (datetime.now() - start_time).total_seconds()
            
            print(f"Generated response in {response_time:.2f}s")
            print(f"DEBUG: Raw response length: {len(raw_answer)}")
            print(f"DEBUG: First 300 chars: {raw_answer[:300]}")
            
            # Clean the response to remove any thinking tags
            answer = self._clean_response(raw_answer)
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return {"error": f"Could not generate response: {e}"}
        
        # Step 5: Store this interaction as a vector in user's memory
        interaction_id = self._store_interaction_vector(
            user_id, user_memory, question, answer, response_time
        )
        
        # Step 6: Update user learning metrics
        self._update_user_metrics(user_id)
        
        return {
            "answer": answer,
            "interaction_id": interaction_id,
            "user_memories_count": user_memory.count(),
            "response_time": response_time,
            "adaptation_used": user_pattern != "learning",
            "context_memories": len(relevant_context)
        }
    
    def _build_adaptive_prompt(self, question: str, context: List[str], pattern: Dict) -> str:
        """Build prompt that adapts to user's learning pattern"""
        
        prompt_parts = [
            "You are an AI tutor that adapts to each student's learning style.",
            "Respond directly without showing your thinking process.",
            "Do not use <think> tags or show internal reasoning.",
            "Give clear, direct answers immediately."
        ]
        
        # Add user's learning pattern adaptation
        if isinstance(pattern, dict):
            if pattern.get("response_style") == "concise":
                prompt_parts.append("Keep responses brief and to the point.")
            elif pattern.get("response_style") == "detailed":
                prompt_parts.append("Provide detailed explanations with examples.")
            
            if pattern.get("complexity_preference") == "high":
                prompt_parts.append("Use advanced concepts and terminology.")
            else:
                prompt_parts.append("Keep explanations simple and accessible.")
        
        # Add relevant memory context
        if context:
            prompt_parts.append("\nRelevant previous conversations:")
            for i, memory in enumerate(context[:2]):  # Limit to 2 memories
                prompt_parts.append(f"{i+1}. {memory[:200]}...")
        
        # Add current question
        prompt_parts.append(f"\nStudent question: {question}")
        prompt_parts.append("\nProvide a helpful, educational response:")
        
        return "\n".join(prompt_parts)
    
    def _store_interaction_vector(self, user_id: str, user_memory, question: str, answer: str, response_time: float) -> str:
        """Store interaction as vector in ChromaDB"""
        try:
            interaction_id = str(uuid.uuid4())
            
            # Create combined text for embedding
            combined_text = f"Question: {question}\nAnswer: {answer}"
            
            # Generate embedding
            embedding = self.simple_embeddings(combined_text)
            
            # Store in user's personal ChromaDB collection
            user_memory.add(
                embeddings=[embedding],
                documents=[combined_text],
                metadatas=[{
                    "user_id": user_id,
                    "question": question[:100],  # Truncated for metadata
                    "timestamp": datetime.now().isoformat(),
                    "response_time": response_time,
                    "answer_length": len(answer)
                }],
                ids=[interaction_id]
            )
            
            # Also store in SQL for structured analysis
            with sqlite3.connect(self.user_db_path) as conn:
                conn.execute("""
                    INSERT INTO interactions 
                    (id, user_id, question, answer, response_time, question_complexity, answer_length)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    interaction_id, user_id, question, answer, response_time,
                    len(question.split()) / 10.0,  # Simple complexity measure
                    len(answer)
                ))
            
            print(f"Stored interaction vector: {interaction_id}")
            return interaction_id
            
        except Exception as e:
            print(f"Error storing interaction: {e}")
            return ""
    
    def _update_user_metrics(self, user_id: str):
        """Update user learning metrics"""
        try:
            with sqlite3.connect(self.user_db_path) as conn:
                conn.execute("""
                    UPDATE users 
                    SET total_interactions = total_interactions + 1
                    WHERE user_id = ?
                """, (user_id,))
        except Exception as e:
            print(f"Error updating metrics: {e}")
    
    def provide_feedback(self, user_id: str, interaction_id: str, rating: float, feedback_text: str = ""):
        """Machine Learning: User provides feedback to improve system"""
        try:
            with sqlite3.connect(self.user_db_path) as conn:
                conn.execute("""
                    UPDATE interactions 
                    SET user_rating = ?, improvement_feedback = ?
                    WHERE id = ? AND user_id = ?
                """, (rating, feedback_text, interaction_id, user_id))
            
            # Re-analyze user patterns with new feedback
            self.analyze_user_pattern(user_id, "", rating)
            
            print(f"Feedback recorded: {rating}/5.0")
            print("User pattern analysis updated")
            
        except Exception as e:
            print(f"Error recording feedback: {e}")
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get user's learning statistics"""
        try:
            user_memory = self.get_or_create_user_memory(user_id)
            
            with sqlite3.connect(self.user_db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        total_interactions,
                        AVG(user_rating) as avg_rating,
                        AVG(response_time) as avg_response_time,
                        COUNT(CASE WHEN user_rating >= 4 THEN 1 END) as high_ratings
                    FROM users u
                    LEFT JOIN interactions i ON u.user_id = i.user_id
                    WHERE u.user_id = ?
                """, (user_id,))
                
                row = cursor.fetchone()
                
                return {
                    "total_memories": user_memory.count() if user_memory else 0,
                    "total_interactions": row[0] or 0,
                    "average_rating": round(row[1] or 0, 2),
                    "average_response_time": round(row[2] or 0, 2),
                    "high_quality_responses": row[3] or 0,
                    "learning_pattern": self.learning_patterns.get(user_id, "still_learning")
                }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}

# ===================== SIMPLE CLI INTERFACE =====================

def main():
    """Simple command-line interface to test the system"""
    
    print("Zero-Knowledge AI Tutor - Starting Fresh")
    print("=" * 50)
    
    # Initialize the AI tutor
    tutor = ZeroKnowledgeAITutor()
    
    # Get user ID
    user_id = input("Enter your user ID (e.g., 'student1'): ").strip()
    if not user_id:
        user_id = "test_user"
    
    # Check if user has existing data
    if tutor.user_exists(user_id):
        stats = tutor.get_user_stats(user_id)
        print(f"\nWelcome back {user_id}! I have {stats.get('total_memories', 0)} memories of our conversations.")
        print("I'll continue learning your preferences from our interactions.")
    else:
        print(f"\nHello {user_id}! I'm starting with zero knowledge about you.")
        print("I'll learn your preferences from our conversations.")
    
    print("Type 'quit' to exit, 'stats' to see your learning data\n")
    
    while True:
        # Get user question
        question = input(f"{user_id}: ").strip()
        
        if question.lower() == 'quit':
            break
        elif question.lower() == 'stats':
            stats = tutor.get_user_stats(user_id)
            print(f"\nYour Learning Stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()
            continue
        elif not question:
            continue
        
        # Generate response
        result = tutor.chat(user_id, question)
        
        if "error" in result:
            print(f"Error: {result['error']}\n")
            continue
        
        # Display response
        print(f"\nAI Tutor: {result['answer']}")
        print(f"Memories: {result['user_memories_count']} | Response time: {result['response_time']:.2f}s")
        
        # Get feedback
        feedback = input("\nRate this response (1-5) or press Enter to skip: ").strip()
        if feedback.isdigit() and 1 <= int(feedback) <= 5:
            tutor.provide_feedback(user_id, result['interaction_id'], float(feedback))
        
        print("\n" + "-" * 30 + "\n")

if __name__ == "__main__":
    # First run setup if needed
    setup_environment()
    
    # Start the main application
    main()