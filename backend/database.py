from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer, UUID, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid
import os
from typing import Generator

# Database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://puniyani:puniyani@localhost:5433/rag_system")

# Create engine
engine = create_engine(DATABASE_URL, echo=True)

# Create session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# Database models
class Chat(Base):
    __tablename__ = "chats"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id = Column(PG_UUID(as_uuid=True), nullable=False)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    __tablename__ = "documents"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id = Column(PG_UUID(as_uuid=True), nullable=False)
    filename = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_id = Column(PG_UUID(as_uuid=True), nullable=False)
    document_id = Column(PG_UUID(as_uuid=True), nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(Vector(384))  # 384 dimensions for sentence-transformers


def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_pgvector_extension():
    """Ensure pgvector extension is created"""
    try:
        with engine.connect() as connection:
            # Check if extension exists
            result = connection.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            )

            if not result.fetchone():
                print("Creating pgvector extension...")
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                connection.commit()
                print("✅ pgvector extension created successfully")
            else:
                print("✅ pgvector extension already exists")

    except Exception as e:
        print(f"❌ Error ensuring pgvector extension: {e}")
        # Try alternative approach
        try:
            print("Trying alternative approach...")
            with engine.connect() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                connection.commit()
                print("✅ pgvector extension created with alternative approach")
        except Exception as e2:
            print(f"❌ Failed to create pgvector extension: {e2}")
            raise


def create_tables():
    """Create all database tables"""
    try:
        # First ensure pgvector extension is available
        ensure_pgvector_extension()

        # Then create tables
        print("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")

    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        raise


def test_connection():
    """Test database connection"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("✅ Database connection successful")
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test the database connection and setup
    print("Testing database connection...")
    if test_connection():
        print("Creating tables...")
        create_tables()
        print("Database setup complete!")
    else:
        print("Database setup failed!")