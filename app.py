"""
Financial Report Analyzer Chat Bot - Application
A Streamlit-based application for analyzing financial documents using OpenAI and RAG.

Features:
- Upload PDF, CSV, Excel, DOCX, TXT files
- Persistent embedding storage (no regeneration)
- Document management (view, delete)
- Chat with your documents using OpenAI
"""

import os
import sys
import streamlit as st

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from document_processor import (
    process_uploaded_file,
    load_document,
    split_documents,
    SUPPORTED_EXTENSIONS,
)
from embedding_store import EmbeddingStoreManager
from llm_handler import LLMQuestionAnswerer

# ──────────────────── Page Configuration ────────────────────

st.set_page_config(
    page_title="Financial Report Analyzer Chat Bot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────── Constants ────────────────────

UPLOAD_DIR = "uploaded_docs"
VECTOR_STORE_DIR = "vector_store"

# ──────────────────── Session State Initialization ────────────────────

if "embedding_manager" not in st.session_state:
    st.session_state.embedding_manager = EmbeddingStoreManager(
        store_dir=VECTOR_STORE_DIR
    )

if "qa_handler" not in st.session_state:
    st.session_state.qa_handler = LLMQuestionAnswerer()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []

if "api_key_configured" not in st.session_state:
    st.session_state.api_key_configured = bool(os.getenv("OPENAI_API_KEY"))


# ──────────────────── Helper Functions ────────────────────


def check_api_key():
    """Check if OpenAI API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key", "")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.session_state.api_key_configured = True
        return True
    return False


def refresh_document_list():
    """Refresh the list of processed documents from the store."""
    summaries = st.session_state.embedding_manager.get_document_summaries()
    st.session_state.processed_docs = summaries
    return summaries


def format_file_size(size_bytes: int) -> str:
    """Format file size for display."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# ──────────────────── Sidebar ────────────────────

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/financial-growth.png",
        width=60,
    )
    st.title("📊 Financial Report Analyzer Chat Bot")
    st.markdown("---")

    # API Key Configuration
    st.subheader("🔑 API Configuration")
    if not st.session_state.api_key_configured:
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key. It will be used only for this session.",
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state.api_key_configured = True
            st.session_state.embedding_manager = EmbeddingStoreManager(
                store_dir=VECTOR_STORE_DIR
            )
            st.session_state.qa_handler = LLMQuestionAnswerer()
            st.rerun()
    else:
        st.success("✅ API Key configured")
        if st.button("🔄 Change API Key", use_container_width=True):
            st.session_state.api_key_configured = False
            st.session_state.pop("openai_api_key", None)
            st.rerun()

    st.markdown("---")

    # Document Stats
    st.subheader("📚 Document Store")
    docs = refresh_document_list()
    st.metric("Total Documents", len(docs))

    if docs:
        total_chunks = sum(d.get("chunks", 0) for d in docs)
        st.metric("Total Chunks", total_chunks)

    st.markdown("---")
    st.markdown(
        """
    **Supported Formats:**
    - 📄 PDF Documents
    - 📑 Word Documents (.docx)
    - 📊 Excel Files (.xlsx, .xls)
    - 📈 CSV Files
    - 📝 Text Files (.txt)
    """
    )


# ──────────────────── Main Content ────────────────────

# Check API key first
if not st.session_state.api_key_configured:
    st.warning(
        "⚠️ Please configure your OpenAI API key in the sidebar to get started."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.info(
            """
        **How to get an OpenAI API Key:**
        1. Go to [OpenAI Platform](https://platform.openai.com)
        2. Sign up or log in
        3. Navigate to API Keys section
        4. Create a new secret key
        5. Copy and paste it in the sidebar
        """
        )
    with col2:
        st.info(
            """
        **What you can do with this app:**
        - Upload documents (PDF, Excel, CSV, Word, Text)
        - Ask questions about your documents
        - Get AI-powered analysis and insights
        - Persistent document storage
        """
        )

    st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs(
    ["📤 Upload Documents", "📚 Document Manager", "💬 Chat with Documents"]
)

# ════════════════════════════════════════════
# TAB 1: Upload Documents
# ════════════════════════════════════════════

with tab1:
    st.header("📤 Upload Documents")
    st.markdown(
        "Upload PDF, DOCX, Excel, CSV, or text files for AI-powered analysis."
    )

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=list(SUPPORTED_EXTENSIONS.keys()),
        accept_multiple_files=True,
        help=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS.values())}",
    )

    if uploaded_files:
        st.markdown("### Files to Process")
        files_to_process = []

        for uploaded_file in uploaded_files:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"📄 {uploaded_file.name}")
            with col2:
                st.write(format_file_size(uploaded_file.size))
            with col3:
                # Check if already processed
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                file_type = SUPPORTED_EXTENSIONS.get(ext, "Unknown")
                files_to_process.append(
                    {
                        "file": uploaded_file,
                        "name": uploaded_file.name,
                        "size": uploaded_file.size,
                        "type": file_type,
                    }
                )

        if st.button(
            "🚀 Process All Files",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.api_key_configured,
        ):
            progress_bar = st.progress(0, text="Starting processing...")
            status_container = st.empty()

            for i, file_info in enumerate(files_to_process):
                try:
                    status_container.info(
                        f"📄 Processing **{file_info['name']}**..."
                    )

                    # Step 1: Save the uploaded file
                    file_data = process_uploaded_file(file_info["file"], UPLOAD_DIR)
                    progress_bar.progress(
                        (i + 0.3) / len(files_to_process),
                        text=f"📥 Saved {file_info['name']}",
                    )

                    # Step 2: Check if already processed
                    if st.session_state.embedding_manager.is_document_processed(
                        file_data["file_path"]
                    ):
                        st.info(
                            f"✅ {file_info['name']} already processed (embeddings loaded from cache)"
                        )
                        progress_bar.progress(
                            (i + 1) / len(files_to_process),
                            text=f"✅ {file_info['name']} already processed",
                        )
                        continue

                    # Step 3: Load document
                    status_container.info(
                        f"📖 Loading content from **{file_info['name']}**..."
                    )
                    documents = load_document(file_data["file_path"])
                    progress_bar.progress(
                        (i + 0.5) / len(files_to_process),
                        text=f"📖 Loaded {len(documents)} pages from {file_info['name']}",
                    )

                    # Step 4: Split into chunks
                    status_container.info(
                        f"✂️ Splitting **{file_info['name']}** into chunks..."
                    )
                    chunks = split_documents(documents)
                    progress_bar.progress(
                        (i + 0.7) / len(files_to_process),
                        text=f"✂️ Created {len(chunks)} chunks from {file_info['name']}",
                    )

                    # Step 5: Generate embeddings and store
                    status_container.info(
                        f"🧠 Generating embeddings for **{file_info['name']}**..."
                    )
                    metadata = st.session_state.embedding_manager.process_and_store_document(
                        file_data["file_path"],
                        file_info["name"],
                        chunks,
                    )
                    progress_bar.progress(
                        (i + 1) / len(files_to_process),
                        text=f"✅ {file_info['name']} processed sucessfully!",
                    )

                    st.success(
                        f"✅ **{file_info['name']}** processed successfully! "
                        f"({len(chunks)} chunks, {format_file_size(file_info['size'])})"
                    )

                except Exception as e:
                    st.error(f"❌ Error processing {file_info['name']}: {str(e)}")
                    progress_bar.progress(
                        (i + 1) / len(files_to_process),
                        text=f"❌ Failed to process {file_info['name']}",
                    )

            progress_bar.empty()
            status_container.success(
                "✅ All files processed! Go to **Chat with Documents** tab to ask questions."
            )
            refresh_document_list()

    # Display supported formats
    st.markdown("---")
    st.subheader("📋 Supported Document Formats")
    cols = st.columns(3)
    formats = list(SUPPORTED_EXTENSIONS.items())
    for idx, (ext, desc) in enumerate(formats):
        col_idx = idx % 3
        with cols[col_idx]:
            st.markdown(f"- **{ext.upper()}**: {desc}")

# ════════════════════════════════════════════
# TAB 2: Document Manager
# ════════════════════════════════════════════

with tab2:
    st.header("📚 Document Manager")
    st.markdown("View and manage your processed documents.")

    docs = refresh_document_list()

    if not docs:
        st.info(
            "📂 No documents processed yet. Go to the **Upload Documents** tab to add documents."
        )
    else:
        st.markdown(f"**{len(docs)} documents** stored in the knowledge base.")

        for doc in docs:
            with st.expander(
                f"📄 {doc.get('file_name', 'Unknown')} "
                f"({doc.get('chunks', 0)} chunks)"
            ):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**File Name:** {doc.get('file_name', 'N/A')}")
                    st.markdown(f"**File Type:** {doc.get('file_type', 'N/A')}")
                    st.markdown(f"**Chunks:** {doc.get('chunks', 0)}")
                    st.markdown(
                        f"**File Size:** {doc.get('file_size_kb', 0):.1f} KB"
                    )
                    st.markdown(
                        f"**Processed At:** {doc.get('processed_at', 'N/A')}"
                    )

                with col2:
                    if st.button(
                        "🗑️ Delete",
                        key=f"delete_{doc.get('doc_id', '')}",
                        use_container_width=True,
                    ):
                        success = st.session_state.embedding_manager.delete_document(
                            doc.get("doc_id", "")
                        )
                        if success:
                            st.success(f"Deleted {doc.get('file_name', '')}")
                            refresh_document_list()
                            st.rerun()
                        else:
                            st.error("Failed to delete document")

        # Clear all button
        st.markdown("---")
        if st.button(
            "🗑️ Clear All Documents",
            type="secondary",
            use_container_width=True,
        ):
            st.session_state.embedding_manager.clear_all()
            st.session_state.chat_history = []
            refresh_document_list()
            st.success("All documents cleared!")
            st.rerun()

# ════════════════════════════════════════════
# TAB 3: Chat with Documents
# ════════════════════════════════════════════

with tab3:
    st.header("💬 Chat with Your Financial Documents")
    st.markdown(
        "Ask questions about your uploaded financial documents and get AI-powered insights."
    )

    docs = refresh_document_list()

    if not docs:
        st.warning(
            "📂 No documents in the knowledge base. "
            "Please upload and process documents first in the **Upload Documents** tab."
        )
        st.stop()

    # Display processed documents info
    with st.expander("📚 Available Documents", expanded=False):
        for doc in docs:
            st.markdown(
                f"- 📄 **{doc.get('file_name', 'Unknown')}** "
                f"({doc.get('chunks', 0)} chunks, "
                f"{doc.get('file_size_kb', 0):.1f} KB)"
            )

    st.markdown("---")

    # Placeholder for chat history
    chat_placeholder = st.empty()

    # Placeholder for new response (spinner + result) - positioned just above chat input
    response_placeholder = st.container()

    # Chat input at the bottom
    prompt = st.chat_input(
        "Ask a question about your financial documents...",
        key="chat_input",
        disabled=not st.session_state.api_key_configured,
    )

    # Render chat history (above the response area)
    with chat_placeholder.container():
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])
                    if "context" in msg:
                        with st.expander("📖 View source context", expanded=False):
                            st.markdown(msg["context"])

    # Process chat input
    if prompt:
        # Add user message to history FIRST
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Determine if this is a greeting / general question
        GREETINGS = ["hi", "hello", "hey", "how are you", "what can you do", "what can you help", 
                     "how can you help", "what kind of information", "who are you", "good morning",
                     "good afternoon", "good evening", "thanks", "thank you"]
        is_greeting = any(g in prompt.lower().strip() for g in GREETINGS)

        if is_greeting:
            # Handle greeting without document search
            response = "Hello! How can I assist you today?"
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            # Document-based question - show spinner in the response area above input
            with response_placeholder:
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Searching documents and generating response..."):
                        try:
                            search_results = st.session_state.embedding_manager.similarity_search(
                                prompt, k=4
                            )

                            response = st.session_state.qa_handler.generate_response(
                                prompt, search_results
                            )

                            # Only store context if actual document results were retrieved
                            if search_results:
                                context_text = ""
                                for i, (doc, score) in enumerate(search_results, 1):
                                    context_text += f"**Document {i}** (Relevance: {score:.4f})\n"
                                    context_text += f"> {doc.page_content[:500]}...\n\n"
                                st.session_state.chat_history.append(
                                    {
                                        "role": "assistant",
                                        "content": response,
                                        "context": context_text,
                                    }
                                )
                            else:
                                st.session_state.chat_history.append(
                                    {
                                        "role": "assistant",
                                        "content": response,
                                    }
                                )

                        except Exception as e:
                            error_msg = f"❌ Error generating response: {str(e)}"
                            st.session_state.chat_history.append(
                                {
                                    "role": "assistant",
                                    "content": error_msg,
                                }
                            )

        st.rerun()

    # Clear chat button
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("�️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()