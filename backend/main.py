from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
from datetime import datetime
import io
import os
from pydantic import BaseModel

# Import our database models and functions
from database import get_db, Chat, ChatMessage, Document, DocumentChunk, create_tables

# Import document processing and RAG components
from document_processor import DocumentProcessor
from rag_system import RAGSystem

app = FastAPI(title="RAG System API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize systems
doc_processor = DocumentProcessor()
rag_system = RAGSystem()


# Pydantic models
class ChatCreate(BaseModel):
    title: str


class ChatResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageCreate(BaseModel):
    chat_id: str
    message: str


class MessageResponse(BaseModel):
    id: str
    chat_id: str
    message: str
    response: str
    timestamp: datetime


class QueryRequest(BaseModel):
    query: str
    chat_id: str  # Make chat_id required


@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    create_tables()
    await rag_system.initialize()


@app.post("/upload-documents/{chat_id}")
async def upload_documents(
        chat_id: str,
        files: List[UploadFile] = File(...),
        db: Session = Depends(get_db)
):
    """Upload and process multiple documents for a specific chat"""

    # Verify chat exists
    chat = db.query(Chat).filter(Chat.id == uuid.UUID(chat_id)).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Check for existing documents with same filenames in this chat
    existing_filenames = db.query(Document.filename).filter(
        Document.chat_id == uuid.UUID(chat_id)
    ).all()
    existing_filenames = [filename[0] for filename in existing_filenames]

    uploaded_docs = []
    skipped_docs = []

    for file in files:
        try:
            # Check if file already exists in this chat
            if file.filename in existing_filenames:
                skipped_docs.append({
                    "filename": file.filename,
                    "reason": "File already exists in this chat"
                })
                continue

            # Read file content
            content = await file.read()

            # Process document based on file type
            if file.filename.endswith('.pdf'):
                text_content = doc_processor.process_pdf(io.BytesIO(content))
            elif file.filename.endswith('.docx'):
                text_content = doc_processor.process_docx(io.BytesIO(content))
            elif file.filename.endswith('.txt'):
                text_content = content.decode('utf-8')
            else:
                skipped_docs.append({
                    "filename": file.filename,
                    "reason": f"Unsupported file type"
                })
                continue

            # Save document to database with chat_id
            doc = Document(
                chat_id=uuid.UUID(chat_id),
                filename=file.filename,
                content=text_content
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)

            # Process and store chunks with embeddings
            chunks = doc_processor.chunk_text(text_content)
            for i, chunk in enumerate(chunks):
                embedding = await rag_system.get_embedding(chunk)

                doc_chunk = DocumentChunk(
                    chat_id=uuid.UUID(chat_id),
                    document_id=doc.id,
                    chunk_text=chunk,
                    chunk_index=i,
                    embedding=embedding
                )
                db.add(doc_chunk)

            db.commit()
            uploaded_docs.append({
                "id": str(doc.id),
                "filename": file.filename,
                "chunks_count": len(chunks)
            })

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing {file.filename}: {str(e)}")

    return {
        "message": f"Successfully uploaded {len(uploaded_docs)} documents",
        "uploaded": uploaded_docs,
        "skipped": skipped_docs
    }


@app.post("/chats", response_model=ChatResponse)
async def create_chat(chat: ChatCreate, db: Session = Depends(get_db)):
    """Create a new chat"""
    new_chat = Chat(title=chat.title)
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)

    return ChatResponse(
        id=str(new_chat.id),
        title=new_chat.title,
        created_at=new_chat.created_at,
        updated_at=new_chat.updated_at
    )


@app.get("/chats", response_model=List[ChatResponse])
async def get_chats(db: Session = Depends(get_db)):
    """Get all chats"""
    chats = db.query(Chat).order_by(Chat.updated_at.desc()).all()
    return [
        ChatResponse(
            id=str(chat.id),
            title=chat.title,
            created_at=chat.created_at,
            updated_at=chat.updated_at
        )
        for chat in chats
    ]


@app.get("/chats/{chat_id}/messages", response_model=List[MessageResponse])
async def get_chat_messages(chat_id: str, db: Session = Depends(get_db)):
    """Get messages for a specific chat"""
    messages = db.query(ChatMessage).filter(
        ChatMessage.chat_id == uuid.UUID(chat_id)
    ).order_by(ChatMessage.timestamp).all()

    return [
        MessageResponse(
            id=str(msg.id),
            chat_id=str(msg.chat_id),
            message=msg.message,
            response=msg.response,
            timestamp=msg.timestamp
        )
        for msg in messages
    ]


@app.post("/query")
async def query_documents(request: QueryRequest, db: Session = Depends(get_db)):
    """Query documents using RAG system"""
    try:
        if not request.chat_id:
            raise HTTPException(status_code=400, detail="chat_id is required")

        # Get relevant chunks using vector similarity for the specific chat
        relevant_chunks = await rag_system.get_relevant_chunks(request.query, request.chat_id, db)

        if not relevant_chunks:
            return {
                "query": request.query,
                "response": "I don't have any documents uploaded for this chat yet. Please upload some documents first to ask questions about them.",
                "sources": 0
            }

        # Generate response using Ollama Llama3
        response = await rag_system.generate_response(request.query, relevant_chunks)

        # Save to chat history
        chat_message = ChatMessage(
            chat_id=uuid.UUID(request.chat_id),
            message=request.query,
            response=response
        )
        db.add(chat_message)

        # Update chat's updated_at timestamp
        chat = db.query(Chat).filter(Chat.id == uuid.UUID(request.chat_id)).first()
        if chat:
            chat.updated_at = datetime.utcnow()

        db.commit()

        return {
            "query": request.query,
            "response": response,
            "sources": len(relevant_chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/chats/{chat_id}/documents")
async def get_chat_documents(chat_id: str, db: Session = Depends(get_db)):
    """Get all documents for a specific chat"""
    documents = db.query(Document).filter(
        Document.chat_id == uuid.UUID(chat_id)
    ).order_by(Document.upload_date.desc()).all()

    return [
        {
            "id": str(doc.id),
            "filename": doc.filename,
            "upload_date": doc.upload_date.isoformat()
        }
        for doc in documents
    ]


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a document and its chunks"""
    # Delete chunks first
    db.query(DocumentChunk).filter(DocumentChunk.document_id == uuid.UUID(document_id)).delete()

    # Delete document
    db.query(Document).filter(Document.id == uuid.UUID(document_id)).delete()

    db.commit()
    return {"message": "Document deleted successfully"}


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, db: Session = Depends(get_db)):
    """Delete a chat, its messages, documents, and chunks"""
    chat_uuid = uuid.UUID(chat_id)

    # Delete document chunks first
    db.query(DocumentChunk).filter(DocumentChunk.chat_id == chat_uuid).delete()

    # Delete documents
    db.query(Document).filter(Document.chat_id == chat_uuid).delete()

    # Delete messages
    db.query(ChatMessage).filter(ChatMessage.chat_id == chat_uuid).delete()

    # Delete chat
    db.query(Chat).filter(Chat.id == chat_uuid).delete()

    db.commit()
    return {"message": "Chat and all associated data deleted successfully"}


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG System API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)