# start_backend.py
# !/usr/bin/env python3
"""
Backend startup script
"""

import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    print("🚀 Starting RAG System Backend...")
    print("=" * 40)

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8002))

    print(f"🌐 Backend will be available at: http://{host}:{port}")
    print("📝 API documentation at: http://localhost:8001/docs")
    print("=" * 40)

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

# start_frontend.py
# !/usr/bin/env python3
"""
Frontend startup script
"""

import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    print("🎨 Starting RAG System Frontend...")
    print("=" * 40)

    print("🌐 Frontend will be available at: http://localhost:8501")
    print("🔗 Make sure backend is running at: http://localhost:8000")
    print("=" * 40)

    # Set Streamlit configuration
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

    # Start Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])