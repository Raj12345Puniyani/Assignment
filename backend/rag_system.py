# import ollama
# from typing import List, Dict, Any
# from sqlalchemy.orm import Session
# from sqlalchemy import text
# from sentence_transformers import SentenceTransformer
# from database import DocumentChunk
# import asyncio
#
#
# class RAGSystem:
#     def __init__(self, model_name: str = "llama3", embedding_model: str = "all-MiniLM-L6-v2"):
#         self.model_name = model_name
#         self.embedding_model_name = embedding_model
#         self.embedding_model = None
#         self.client = ollama.Client()
#
#     async def initialize(self):
#         """Initialize the RAG system"""
#         try:
#             # Initialize embedding model
#             self.embedding_model = SentenceTransformer(self.embedding_model_name)
#
#             # Check if Ollama model is available
#             models = self.client.list()
#             model_names = [model['model'] for model in models['models']]
#
#             if self.model_name not in model_names:
#                 print(f"Model {self.model_name} not found. Available models: {model_names}")
#                 print(f"Pulling {self.model_name} model...")
#                 self.client.pull(self.model_name)
#                 print(f"Successfully pulled {self.model_name}")
#
#             print("RAG System initialized successfully")
#
#         except Exception as e:
#             print(f"Error initializing RAG system: {str(e)}")
#             raise
#
#     async def get_embedding(self, text: str) -> List[float]:
#         """Generate embedding for text using SentenceTransformer"""
#         try:
#             # Run embedding generation in thread pool to avoid blocking
#             loop = asyncio.get_event_loop()
#             embedding = await loop.run_in_executor(
#                 None,
#                 self.embedding_model.encode,
#                 text
#             )
#             return embedding.tolist()
#         except Exception as e:
#             print(f"Error generating embedding: {str(e)}")
#             raise
#
#     async def get_relevant_chunks(self, query: str, db: Session, top_k: int = 5) -> List[Dict[str, Any]]:
#         """Retrieve relevant document chunks using vector similarity"""
#         try:
#             # Get query embedding
#             query_embedding = await self.get_embedding(query)
#
#             # Use pgvector for similarity search
#             similarity_query = text("""
#                 SELECT
#                     dc.chunk_text,
#                     dc.chunk_index,
#                     d.filename,
#                     dc.embedding <-> :query_embedding as distance
#                 FROM document_chunks dc
#                 JOIN documents d ON dc.document_id = d.id
#                 ORDER BY dc.embedding <-> :query_embedding
#                 LIMIT :limit
#             """)
#
#             result = db.execute(
#                 similarity_query,
#                 {
#                     "query_embedding": str(query_embedding),
#                     "limit": top_k
#                 }
#             )
#
#             relevant_chunks = []
#             for row in result:
#                 relevant_chunks.append({
#                     "text": row.chunk_text,
#                     "chunk_index": row.chunk_index,
#                     "filename": row.filename,
#                     "similarity_score": 1 - row.distance  # Convert distance to similarity
#                 })
#
#             return relevant_chunks
#
#         except Exception as e:
#             print(f"Error retrieving relevant chunks: {str(e)}")
#             # Fallback: return some chunks without similarity scoring
#             chunks = db.query(DocumentChunk).limit(top_k).all()
#             return [
#                 {
#                     "text": chunk.chunk_text,
#                     "chunk_index": chunk.chunk_index,
#                     "filename": "unknown",
#                     "similarity_score": 0.5
#                 }
#                 for chunk in chunks
#             ]
#
#     async def generate_response(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
#         """Generate response using Ollama Llama3 model"""
#         try:
#             # Prepare context from relevant chunks
#             context = ""
#             for i, chunk in enumerate(relevant_chunks):
#                 context += f"Document {i + 1} ({chunk['filename']}):\n{chunk['text']}\n\n"
#
#             # Create prompt
#             prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say so.
#
# Context:
# {context}
#
# Question: {query}
#
# Answer:"""
#
#             # Generate response using Ollama
#             response = self.client.chat(
#                 model=self.model_name,
#                 messages=[
#                     {
#                         'role': 'user',
#                         'content': prompt
#                     }
#                 ],
#                 options={
#                     'temperature': 0.7,
#                     'top_p': 0.9,
#                     'max_tokens': 1000
#                 }
#             )
#
#             return response['message']['content']
#
#         except Exception as e:
#             print(f"Error generating response: {str(e)}")
#             return f"I apologize, but I encountered an error while generating a response: {str(e)}"
#
#     async def generate_chat_title(self, first_message: str) -> str:
#         """Generate a title for the chat based on the first message"""
#         try:
#             prompt = f"""Generate a short, descriptive title (maximum 6 words) for a chat that starts with this message: "{first_message}"
#
# Title:"""
#
#             response = self.client.chat(
#                 model=self.model_name,
#                 messages=[
#                     {
#                         'role': 'user',
#                         'content': prompt
#                     }
#                 ],
#                 options={
#                     'temperature': 0.5,
#                     'max_tokens': 50
#                 }
#             )
#
#             title = response['message']['content'].strip()
#             # Ensure title is not too long
#             if len(title) > 50:
#                 title = title[:47] + "..."
#
#             return title
#
#         except Exception as e:
#             print(f"Error generating chat title: {str(e)}")
#             return "New Chat"

import ollama
import numpy as np
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import text
from sentence_transformers import SentenceTransformer
from database import DocumentChunk
import uuid
import asyncio


class RAGSystem:
    def __init__(self, model_name: str = "llama3", embedding_model: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.client = ollama.Client()

    async def initialize(self):
        """Initialize the RAG system"""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

            # Check if Ollama model is available
            models = self.client.list()
            model_names = [model['model'] for model in models['models']]

            if self.model_name not in model_names:
                print(f"Model {self.model_name} not found. Available models: {model_names}")
                print(f"Pulling {self.model_name} model...")
                self.client.pull(self.model_name)
                print(f"Successfully pulled {self.model_name}")

            print("RAG System initialized successfully")

        except Exception as e:
            print(f"Error initializing RAG system: {str(e)}")
            raise

    async def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using SentenceTransformer"""
        try:
            # Run embedding generation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self.embedding_model.encode,
                text
            )
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            raise

    async def get_relevant_chunks(self, query: str, chat_id: str, db: Session, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks using vector similarity for a specific chat"""
        try:
            # Get query embedding
            query_embedding = await self.get_embedding(query)

            # Use pgvector for similarity search within the specific chat
            similarity_query = text("""
                SELECT 
                    dc.chunk_text,
                    dc.chunk_index,
                    d.filename,
                    dc.embedding <-> :query_embedding as distance
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE dc.chat_id = :chat_id
                ORDER BY dc.embedding <-> :query_embedding
                LIMIT :limit
            """)

            result = db.execute(
                similarity_query,
                {
                    "query_embedding": str(query_embedding),
                    "chat_id": chat_id,
                    "limit": top_k
                }
            )

            relevant_chunks = []
            for row in result:
                relevant_chunks.append({
                    "text": row.chunk_text,
                    "chunk_index": row.chunk_index,
                    "filename": row.filename,
                    "similarity_score": 1 - row.distance  # Convert distance to similarity
                })

            return relevant_chunks

        except Exception as e:
            print(f"Error retrieving relevant chunks: {str(e)}")
            # Fallback: return some chunks without similarity scoring from the specific chat
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.chat_id == uuid.UUID(chat_id)
            ).limit(top_k).all()

            return [
                {
                    "text": chunk.chunk_text,
                    "chunk_index": chunk.chunk_index,
                    "filename": "unknown",
                    "similarity_score": 0.5
                }
                for chunk in chunks
            ]

    async def generate_response(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Generate response using Ollama Llama3 model"""
        try:
            # Prepare context from relevant chunks
            context = ""
            for i, chunk in enumerate(relevant_chunks):
                context += f"Document {i + 1} ({chunk['filename']}):\n{chunk['text']}\n\n"

            # Create prompt
            prompt = f"""Based on the following context, please answer the question. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {query}

Answer:"""

            # Generate response using Ollama
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'max_tokens': 1000
                }
            )

            return response['message']['content']

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"

    async def generate_chat_title(self, first_message: str) -> str:
        """Generate a title for the chat based on the first message"""
        try:
            prompt = f"""Generate a short, descriptive title (maximum 6 words) for a chat that starts with this message: "{first_message}"

Title:"""

            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.5,
                    'max_tokens': 50
                }
            )

            title = response['message']['content'].strip()
            # Ensure title is not too long
            if len(title) > 50:
                title = title[:47] + "..."

            return title

        except Exception as e:
            print(f"Error generating chat title: {str(e)}")
            return "New Chat"