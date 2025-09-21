import chromadb
import numpy as np
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from database.models import SessionLocal, UserMemory
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.db = SessionLocal()
        
        self.memory_types = {
            'interaction': {'retention_days': 90, 'importance_weight': 0.7},
            'preference': {'retention_days': 365, 'importance_weight': 0.9},
            'achievement': {'retention_days': 730, 'importance_weight': 0.8},
            'mistake': {'retention_days': 180, 'importance_weight': 0.6},
            'insight': {'retention_days': 365, 'importance_weight': 0.85}
        }
    
    def _get_user_collection(self, user_id: str):
        collection_name = f"user_memories_{user_id}"
        try:
            return self.chroma_client.get_collection(name=collection_name)
        except:
            return self.chroma_client.create_collection(
                name=collection_name,
                metadata={"user_id": user_id}
            )
    
    async def store_interaction(self, user_id: str, query: str, response: str, context: Dict, metadata: Dict):
        memory_content = f"Query: {query}\nResponse: {response}"
        
        memory_data = {
            'user_id': user_id,
            'type': 'interaction',
            'content': memory_content,
            'context': context,
            'metadata': metadata,
            'timestamp': datetime.utcnow().isoformat(),
            'importance_score': self._calculate_importance_score(query, response, metadata)
        }
        
        await self._store_memory(user_id, memory_data)
        
        db_memory = UserMemory(
            user_id=user_id,
            memory_type='interaction',
            content=memory_content,
            context=context,
            importance_score=memory_data['importance_score'],
            created_at=datetime.utcnow()
        )
        self.db.add(db_memory)
        self.db.commit()
    
    async def store_user_metadata(self, user_id: str, metadata_type: str, data: Dict):
        memory_content = f"Metadata: {metadata_type}\nData: {json.dumps(data, indent=2)}"
        
        memory_data = {
            'user_id': user_id,
            'type': 'preference',
            'content': memory_content,
            'context': {'metadata_type': metadata_type},
            'metadata': data,
            'timestamp': datetime.utcnow().isoformat(),
            'importance_score': 0.8
        }
        
        await self._store_memory(user_id, memory_data)
    
    async def retrieve_relevant_memories(self, user_id: str, query: str, limit: int = 10) -> List[Dict]:
        try:
            collection = self._get_user_collection(user_id)
            query_embedding = self.embedding_model.encode([query])[0]
            
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=limit
            )
            
            memories = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    memory = {
                        'content': doc,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'similarity_score': 1.0 - results['distances'][0][i],
                        'id': results['ids'][0][i]
                    }
                    memories.append(memory)
            
            memories.sort(key=lambda x: (
                x.get('metadata', {}).get('importance_score', 0.5),
                x.get('metadata', {}).get('timestamp', '')
            ), reverse=True)
            
            return memories
        
        except Exception as e:
            logger.error(f"Error retrieving memories for user {user_id}: {e}")
            return []
    
    async def get_learning_history(self, user_id: str, days_back: int = 30) -> List[Dict]:
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        memories = self.db.query(UserMemory).filter(
            UserMemory.user_id == user_id,
            UserMemory.memory_type == 'interaction',
            UserMemory.created_at >= cutoff_date
        ).order_by(UserMemory.created_at.desc()).all()
        
        history = []
        for memory in memories:
            history.append({
                'content': memory.content,
                'context': memory.context,
                'timestamp': memory.created_at.isoformat(),
                'importance': memory.importance_score
            })
        
        return history
    
    async def mark_interaction_as_successful(self, user_id: str, interaction_id: int):
        try:
            collection = self._get_user_collection(user_id)
            logger.info(f"Marked interaction {interaction_id} as successful for user {user_id}")
        except Exception as e:
            logger.error(f"Error marking interaction as successful: {e}")
    
    def _calculate_importance_score(self, query: str, response: str, metadata: Dict) -> float:
        base_score = 0.5
        
        query_length = len(query.split())
        if query_length > 15:
            base_score += 0.2
        
        if 'user_rating' in metadata and metadata['user_rating'] >= 4:
            base_score += 0.3
        
        topic = metadata.get('topic', '')
        if topic and 'new_topic' in metadata:
            base_score += 0.2
        
        if any(word in response.lower() for word in ['understand', 'got it', 'makes sense']):
            base_score += 0.15
        
        return min(1.0, base_score)
    
    async def _store_memory(self, user_id: str, memory_data: Dict):
        collection = self._get_user_collection(user_id)
        content = memory_data['content']
        
        embedding = self.embedding_model.encode([content])[0]
        
        memory_id = str(uuid.uuid4())
        
        collection.add(
            documents=[content],
            metadatas=[{
                'user_id': user_id,
                'type': memory_data['type'],
                'timestamp': memory_data['timestamp'],
                'importance_score': memory_data['importance_score'],
                **memory_data.get('metadata', {})
            }],
            ids=[memory_id],
            embeddings=[embedding.tolist()]
        )
    
    async def cleanup_old_memories(self, user_id: str):
        cutoff_date = datetime.utcnow() - timedelta(days=90)
        
        old_memories = self.db.query(UserMemory).filter(
            UserMemory.user_id == user_id,
            UserMemory.created_at < cutoff_date,
            UserMemory.importance_score < 0.6
        ).all()
        
        for memory in old_memories:
            self.db.delete(memory)
        
        self.db.commit()
        logger.info(f"Cleaned up {len(old_memories)} old memories for user {user_id}")