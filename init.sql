-- Initialize pgvector extension
-- This file should be placed in the root directory alongside docker-compose.yml

-- Create the vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE rag_system TO puniyani;