import PyPDF2
import docx
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]  # Try these separators in order
        )

    def process_pdf(self, file_content) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file_content)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")

    def process_docx(self, file_content) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_content)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error processing DOCX: {str(e)}")

    def process_txt(self, file_content) -> str:
        """Process plain text file"""
        try:
            if isinstance(file_content, bytes):
                return file_content.decode('utf-8')
            return file_content
        except Exception as e:
            raise Exception(f"Error processing TXT: {str(e)}")

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using LlamaIndex RecursiveCharacterTextSplitter"""
        try:
            chunks = self.text_splitter.split_text(text)
            return [chunk for chunk in chunks if len(chunk.strip()) > 50]  # Filter out very short chunks
        except Exception as e:
            raise Exception(f"Error chunking text: {str(e)}")

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove special characters that might interfere with processing
        text = text.replace('\x00', '')  # Remove null characters

        return text.strip()