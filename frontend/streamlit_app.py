import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict, Any
import uuid

# Configuration
API_BASE_URL = "http://localhost:8003"

# Initialize session state
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chats" not in st.session_state:
    st.session_state.chats = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_documents" not in st.session_state:
    st.session_state.chat_documents = []


def load_chats():
    """Load all chats from the backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/chats")
        if response.status_code == 200:
            st.session_state.chats = response.json()
        else:
            st.error("Failed to load chats")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {str(e)}")


def create_new_chat(title: str = None):
    """Create a new chat"""
    if not title:
        title = f"New Chat {datetime.now().strftime('%H:%M')}"

    try:
        response = requests.post(
            f"{API_BASE_URL}/chats",
            json={"title": title}
        )
        if response.status_code == 200:
            new_chat = response.json()
            st.session_state.current_chat_id = new_chat["id"]
            st.session_state.messages = []
            st.session_state.chat_documents = []
            load_chats()
            st.rerun()
        else:
            st.error("Failed to create new chat")
    except requests.exceptions.RequestException as e:
        st.error(f"Error creating chat: {str(e)}")


def load_chat_messages(chat_id: str):
    """Load messages for a specific chat"""
    try:
        response = requests.get(f"{API_BASE_URL}/chats/{chat_id}/messages")
        if response.status_code == 200:
            st.session_state.messages = response.json()
        else:
            st.error("Failed to load chat messages")
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading messages: {str(e)}")


def load_chat_documents(chat_id: str):
    """Load documents for a specific chat"""
    try:
        response = requests.get(f"{API_BASE_URL}/chats/{chat_id}/documents")
        if response.status_code == 200:
            st.session_state.chat_documents = response.json()
        else:
            st.error("Failed to load chat documents")
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading documents: {str(e)}")


def send_query(query: str, chat_id: str):
    """Send a query to the RAG system"""
    try:
        payload = {"query": query, "chat_id": chat_id}

        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Query failed: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending query: {str(e)}")
        return None


def upload_documents(files, chat_id: str):
    """Upload documents to the backend for a specific chat"""
    try:
        files_data = []
        for file in files:
            files_data.append(("files", (file.name, file.getvalue(), file.type)))

        response = requests.post(
            f"{API_BASE_URL}/upload-documents/{chat_id}",
            files=files_data
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Upload failed: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading documents: {str(e)}")
        return None


def delete_chat(chat_id: str):
    """Delete a chat"""
    try:
        response = requests.delete(f"{API_BASE_URL}/chats/{chat_id}")
        if response.status_code == 200:
            if st.session_state.current_chat_id == chat_id:
                st.session_state.current_chat_id = None
                st.session_state.messages = []
                st.session_state.chat_documents = []
            load_chats()
            st.rerun()
        else:
            st.error("Failed to delete chat")
    except requests.exceptions.RequestException as e:
        st.error(f"Error deleting chat: {str(e)}")


def delete_document(document_id: str):
    """Delete a document"""
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{document_id}")
        if response.status_code == 200:
            # Reload documents for current chat
            if st.session_state.current_chat_id:
                load_chat_documents(st.session_state.current_chat_id)
            st.rerun()
        else:
            st.error("Failed to delete document")
    except requests.exceptions.RequestException as e:
        st.error(f"Error deleting document: {str(e)}")


# Main UI
st.set_page_config(
    page_title="RAG System with Ollama & Llama3",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG System with Ollama & Llama3")

# Sidebar for chat management
with st.sidebar:
    st.header("💬 Chat Management")

    # New Chat Button
    if st.button("➕ Create New Chat", type="primary", use_container_width=True):
        create_new_chat()

    st.divider()

    # Load chats on first run
    if not st.session_state.chats:
        load_chats()

    # Display existing chats
    if st.session_state.chats:
        st.subheader("Your Chats")
        for chat in st.session_state.chats:
            col1, col2 = st.columns([4, 1])

            with col1:
                button_type = "primary" if st.session_state.current_chat_id == chat['id'] else "secondary"
                if st.button(
                        f"💬 {chat['title'][:25]}{'...' if len(chat['title']) > 25 else ''}",
                        key=f"chat_{chat['id']}",
                        use_container_width=True,
                        type=button_type
                ):
                    st.session_state.current_chat_id = chat['id']
                    load_chat_messages(chat['id'])
                    load_chat_documents(chat['id'])
                    st.rerun()

            with col2:
                if st.button("🗑️", key=f"delete_{chat['id']}", help="Delete chat"):
                    delete_chat(chat['id'])
    else:
        st.info("No chats yet. Create your first chat to get started!")

# Main content area
if st.session_state.current_chat_id:
    # Get current chat info
    current_chat = next((chat for chat in st.session_state.chats if chat['id'] == st.session_state.current_chat_id),
                        None)

    if current_chat:
        st.header(f"📝 {current_chat['title']}")

        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Documents", "⚙️ Settings"])

        with tab1:
            # Chat Messages Section
            st.subheader("Messages")

            # Display chat messages
            messages_container = st.container()

            with messages_container:
                if st.session_state.messages:
                    for message in st.session_state.messages:
                        # User message
                        with st.chat_message("user"):
                            st.write(message['message'])
                            st.caption(f"⏰ {message['timestamp']}")

                        # Assistant response
                        with st.chat_message("assistant"):
                            st.write(message['response'])
                else:
                    st.info("No messages yet. Upload some documents and start asking questions!")

            # Query input section
            st.divider()

            # Check if there are documents uploaded
            if st.session_state.chat_documents:
                with st.form("query_form", clear_on_submit=True):
                    query = st.text_area(
                        "Ask a question about your documents:",
                        placeholder="What would you like to know?",
                        height=100
                    )

                    col1, col2 = st.columns([1, 4])
                    with col1:
                        submit_button = st.form_submit_button("Send", type="primary")

                    if submit_button and query.strip():
                        with st.spinner("Generating response..."):
                            # Send query
                            result = send_query(query, st.session_state.current_chat_id)

                            if result:
                                # Reload messages from backend to sync
                                load_chat_messages(st.session_state.current_chat_id)
                                st.rerun()
            else:
                st.warning("📁 Please upload documents first before asking questions!")
                if st.button("Go to Documents Tab", type="secondary"):
                    st.rerun()  # This will keep the documents tab in focus

        with tab2:
            # Documents Management Section
            st.subheader("Document Upload & Management")

            # Document upload section
            st.write("**Upload New Documents**")

            uploaded_files = st.file_uploader(
                "Choose files to upload",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="file_uploader"
            )

            if uploaded_files:
                if st.button("Upload Documents", type="primary"):
                    with st.spinner("Uploading and processing documents..."):
                        result = upload_documents(uploaded_files, st.session_state.current_chat_id)
                        if result:
                            uploaded_count = len(result.get('uploaded', []))
                            skipped_count = len(result.get('skipped', []))

                            if uploaded_count > 0:
                                st.success(f"✅ Successfully uploaded {uploaded_count} documents!")

                            if skipped_count > 0:
                                st.warning(f"⚠️ Skipped {skipped_count} documents:")
                                for skipped in result.get('skipped', []):
                                    st.write(f"• {skipped['filename']}: {skipped['reason']}")

                            # Reload documents
                            load_chat_documents(st.session_state.current_chat_id)
                            st.rerun()

            st.divider()

            # Display uploaded documents
            st.write("**Current Documents**")

            if st.session_state.chat_documents:
                for doc in st.session_state.chat_documents:
                    col1, col2, col3 = st.columns([3, 2, 1])

                    with col1:
                        st.write(f"📄 **{doc['filename']}**")

                    with col2:
                        upload_date = datetime.fromisoformat(doc['upload_date'].replace('Z', '+00:00'))
                        st.write(f"📅 {upload_date.strftime('%Y-%m-%d %H:%M')}")

                    with col3:
                        if st.button("🗑️", key=f"del_doc_{doc['id']}", help="Delete document"):
                            delete_document(doc['id'])
            else:
                st.info("No documents uploaded yet. Upload some documents to start asking questions!")

        with tab3:
            # Settings/Info Section
            st.subheader("Chat Settings & Information")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Chat Information**")
                st.write(f"• **Chat ID**: `{current_chat['id'][:8]}...`")
                created_date = datetime.fromisoformat(current_chat['created_at'].replace('Z', '+00:00'))
                st.write(f"• **Created**: {created_date.strftime('%Y-%m-%d %H:%M')}")
                st.write(f"• **Messages**: {len(st.session_state.messages)}")
                st.write(f"• **Documents**: {len(st.session_state.chat_documents)}")

            with col2:
                st.write("**Actions**")
                if st.button("🔄 Refresh Chat Data", type="secondary"):
                    load_chat_messages(st.session_state.current_chat_id)
                    load_chat_documents(st.session_state.current_chat_id)
                    st.success("Chat data refreshed!")
                    st.rerun()

                if st.button("🗑️ Delete This Chat", type="secondary"):
                    if st.button("⚠️ Confirm Delete", type="secondary"):
                        delete_chat(st.session_state.current_chat_id)

else:
    # Welcome screen when no chat is selected
    st.markdown("""
    ## Welcome to the RAG System! 🎉

    Get started by creating your first chat session.

    ### How it works:
    1. **Create a Chat** 📝 - Click "Create New Chat" in the sidebar
    2. **Upload Documents** 📁 - Add PDF, DOCX, or TXT files to your chat
    3. **Ask Questions** 💬 - Query your documents using natural language
    4. **Get AI Responses** 🤖 - Powered by Ollama Llama3 model

    ### Key Features:
    - ✅ **Chat-specific documents** - Each chat has its own document collection
    - ✅ **Duplicate prevention** - Can't upload the same file twice to a chat  
    - ✅ **Multi-document support** - Upload multiple files at once
    - ✅ **Vector similarity search** - Find the most relevant content
    - ✅ **Persistent history** - All chats and messages are saved
    - ✅ **LlamaIndex integration** - Advanced document processing

    ---

    ### 🚀 Ready to start?

    **Click "Create New Chat" in the sidebar to begin!**
    """)

    # System status check
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔧 System Status")
        if st.button("Check API Connection"):
            try:
                response = requests.get(f"{API_BASE_URL}/")
                if response.status_code == 200:
                    st.success("✅ Backend API is running")
                else:
                    st.error("❌ Backend API is not responding correctly")
            except requests.exceptions.RequestException:
                st.error("❌ Cannot connect to backend API")
                st.info("💡 Make sure your FastAPI backend is running on port 8002")

    with col2:
        st.subheader("📊 Quick Stats")
        if st.session_state.chats:
            st.metric("Total Chats", len(st.session_state.chats))
        else:
            st.metric("Total Chats", 0)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    <p>🤖 RAG System powered by <strong>Ollama Llama3</strong>, <strong>LlamaIndex</strong>, <strong>PostgreSQL</strong> & <strong>Streamlit</strong></p>
    <p>Each chat maintains its own document collection with intelligent deduplication</p>
</div>
""", unsafe_allow_html=True)