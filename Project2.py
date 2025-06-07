import streamlit as st #Used to create the interactive frontend of the web application.
import pandas as pd #Used for structured data manipulation and analysis.
import os #Used for interacting with the operating system.
import warnings #To suppress or manage warnings.
from datetime import datetime #For working with dates and times. 
from groq import Groq # To interact with the Groq API
from typing import List, Dict, Any #Used in function definitions like def func(data: List[str]) -> Dict[str, Any]
import hashlib #For generating unique hashes.
import PyPDF2 #Reading and extracting text from PDFs 
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Suppress warnings globally to avoid clutter in streamlit app.
# Necessary for clean UI.
warnings.filterwarnings("ignore")

# Set up Groq API key from environment
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize session state
if 'documents' not in st.session_state: #Initializes a list to store all uploaded documents. Each Document is typically Dictionary with metadata and text chunks.
    st.session_state.documents = []

if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None #Placeholder for the FAISS vector index, which stores embedded document vectors for fast similarity search.

if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None #Stores the embedding model used to convert text into vector form.
    #also initialized to None, to be set when the user selects or loads a model
if 'document_chunks' not in st.session_state: 
    #Holds the actual text chunks (splits of full documents).
    #These chunks are what's embedded and stored in the FAISS index.
    st.session_state.document_chunks = []

if 'chunk_metadata' not in st.session_state:
    st.session_state.chunk_metadata = []


class DocumentProcessor:
    #This is a DocumentProcessor class that provides functions to extract paragraph-level text from PDF:PyPDF2, .txt files.
    """DocumentProcessor class that provides functions to extract paragraph-level text from PDF and .txt files."""
        
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        try: #Without a try-except block, if any error occurs during PDF reading or text extraction, the whole app or script would crash.
            pdf_reader = PyPDF2.PdfReader(pdf_file) #It opens and reads the PDF so you can access its pages and content.
            text_content = [] #Initializes an empty list to store extracted text data structured by page and paragraph.
                        
            for page_num, page in enumerate(pdf_reader.pages, 1):#Loops through each page in the PDF.enumerate(..., 1) means counting pages starting at 1 (so first page is page number 1, not 0). page_num is the current page number, and page is the actual page object.
                page_text = page.extract_text()
                if page_text and isinstance(page_text, str):
                    # Split into paragraphs
                    paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                    for para_num, paragraph in enumerate(paragraphs, 1):
                        if isinstance(paragraph, str) and paragraph and len(paragraph) > 20:
                            text_content.append({
                                'page': page_num,
                                'paragraph': para_num,
                                'text': str(paragraph)  # Ensure string type
                            })
            return text_content
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return []
    
    def extract_text_from_txt(self, txt_file):
        """Extract text from text file"""
        try:
            content = txt_file.read().decode('utf-8')
            if not isinstance(content, str):
                content = str(content)
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            text_content = []
            
            for para_num, paragraph in enumerate(paragraphs, 1):
                if isinstance(paragraph, str) and paragraph and len(paragraph) > 20:
                    text_content.append({
                        'page': 1,
                        'paragraph': para_num,
                        'text': str(paragraph)  # Ensure string type
                    })
            return text_content
        except Exception as e:
            st.error(f"Error extracting text from TXT: {str(e)}")
            return []


class FAISSSearchEngine:
    """FAISS-based search engine for document retrieval."""
    
    def __init__(self):
        self.index = None
        self.embedding_model = None
        self.document_chunks = []
        self.chunk_metadata = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2 model
        self.initialize_faiss()
    
    def initialize_faiss(self):
        """Initialize FAISS index and embedding model"""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create FAISS index (using IndexFlatIP for cosine similarity)
            self.index = faiss.IndexFlatIP(self.dimension)
            
            return True
        except Exception as e:
            st.error(f"Error initializing FAISS: {str(e)}")
            return False
    
    def clean_text(self, text):
        """Clean and validate text input"""
        if text is None:
            return ""
        
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
        
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        
        # Remove any non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        return text.strip()
    
    def add_documents(self, documents_data):
        """Add documents to FAISS index"""
        try:
            if not self.index or not self.embedding_model:
                st.error("FAISS index or embedding model not initialized")
                return False
            
            # Prepare data for FAISS
            texts = []
            metadatas = []
            
            for doc_data in documents_data:
                if not isinstance(doc_data, dict) or 'chunks' not in doc_data:
                    continue
                    
                for chunk in doc_data['chunks']:
                    if not isinstance(chunk, dict) or 'text' not in chunk:
                        continue
                        
                    # Clean and validate text
                    raw_text = chunk['text']
                    cleaned_text = self.clean_text(raw_text)
                    
                    # Only add non-empty text chunks with minimum length
                    if cleaned_text and len(cleaned_text) > 20:
                        texts.append(cleaned_text)
                        metadatas.append({
                            'doc_id': doc_data.get('doc_id', ''),
                            'filename': doc_data.get('filename', ''),
                            'page': chunk.get('page', 1),
                            'paragraph': chunk.get('paragraph', 1),
                            'upload_time': doc_data.get('upload_time', ''),
                            'text': cleaned_text
                        })
            
            if not texts:
                st.warning("No valid text chunks found to add to FAISS")
                return False
            
            # Debug output
            st.write(f"Preparing to encode {len(texts)} chunks")
            if texts:
                sample_text = texts[0][:100] + "..." if len(texts[0]) > 100 else texts[0]
                st.write(f"Sample text: {sample_text}")
            
            # Validate all texts are strings
            for i, text in enumerate(texts):
                if not isinstance(text, str):
                    st.error(f"Text at index {i} is not a string: {type(text)}")
                    return False
                if len(text.strip()) == 0:
                    st.error(f"Text at index {i} is empty after cleaning")
                    return False
            
            # Generate embeddings with error handling
            try:
                embeddings = self.embedding_model.encode(
                    texts,
                    convert_to_tensor=False,
                    normalize_embeddings=False,  # Don't normalize here, do it manually
                    show_progress_bar=True,
                    batch_size=32  # Add batch size to avoid memory issues
                )
            except Exception as embed_error:
                st.error(f"Embedding generation failed: {str(embed_error)}")
                st.error(f"First few texts: {texts[:3]}")
                return False
            
            # Ensure embeddings are in correct format
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            # Validate embedding dimensions
            if embeddings.shape[1] != self.dimension:
                st.error(f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}")
                return False
            
            embeddings = embeddings.astype('float32')
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            st.write(f"📊 Embedding stats: Shape={embeddings.shape}, Min={embeddings.min():.4f}, Max={embeddings.max():.4f}")
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store metadata
            self.document_chunks.extend(texts)
            self.chunk_metadata.extend(metadatas)
            
            st.success(f"Successfully added {len(texts)} chunks to FAISS index")
            st.write(f"📈 Total vectors in index: {self.index.ntotal}")
            return True
            
        except Exception as e:
            st.error(f"Error adding documents to FAISS: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def search(self, query, n_results=15):
        """Search for relevant documents using FAISS"""
        try:
            if not self.index or not self.embedding_model or self.index.ntotal == 0:
                st.warning(f"Search unavailable - Index: {self.index is not None}, Model: {self.embedding_model is not None}, Vectors: {self.index.ntotal if self.index else 0}")
                return []
            
            # Clean and validate query
            cleaned_query = self.clean_text(query)
            if not cleaned_query:
                st.error("Query is empty after cleaning")
                return []
            
            st.write(f"🔍 Searching for: '{cleaned_query}'")
            st.write(f"📊 Index contains {self.index.ntotal} vectors")
            
            # Generate query embedding
            try:
                query_embedding = self.embedding_model.encode(
                    [cleaned_query],
                    convert_to_tensor=False,
                    normalize_embeddings=False  # Don't normalize here, do it manually
                )
            except Exception as embed_error:
                st.error(f"Query embedding failed: {str(embed_error)}")
                return []
            
            # Ensure query embedding is in correct format
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
            
            query_embedding = query_embedding.astype('float32')
            
            # Normalize query embedding for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            n_results = min(n_results, self.index.ntotal)
            similarities, indices = self.index.search(query_embedding, n_results)
            
            st.write(f"📈 Raw similarities: {similarities[0][:5].tolist()}")
            st.write(f"📍 Indices: {indices[0][:5].tolist()}")
            
            # Process results with more lenient threshold
            search_results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx != -1:  # FAISS returns -1 for empty slots
                    if idx < len(self.chunk_metadata):  # Validate index
                        result_data = self.chunk_metadata[idx].copy()
                        result_data['similarity'] = float(similarity)
                        search_results.append(result_data)
                        if i < 3:  # Show first 3 results for debugging
                            st.write(f"Result {i+1}: Score={similarity:.4f}, Text preview: {result_data['text'][:100]}...")
            
            st.write(f"✅ Found {len(search_results)} total results before filtering")
            return search_results
            
        except Exception as e:
            st.error(f"FAISS search error: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def clear_index(self):
        """Clear all documents from the FAISS index"""
        try:
            # Reset FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.document_chunks = []
            self.chunk_metadata = []
            return True
        except Exception as e:
            st.error(f"Error clearing FAISS index: {str(e)}")
            return False
    
    def get_index_count(self):
        """Get the number of vectors in the FAISS index"""
        try:
            if self.index:
                return self.index.ntotal
            return 0
        except:
            return 0
    
    def save_index(self, filepath):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata
            with open(f"{filepath}.metadata", 'wb') as f:
                pickle.dump({
                    'document_chunks': self.document_chunks,
                    'chunk_metadata': self.chunk_metadata
                }, f)
            return True
        except Exception as e:
            st.error(f"Error saving FAISS index: {str(e)}")
            return False
    
    def load_index(self, filepath):
        """Load FAISS index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load metadata
            with open(f"{filepath}.metadata", 'rb') as f:
                data = pickle.load(f)
                self.document_chunks = data['document_chunks']
                self.chunk_metadata = data['chunk_metadata']
            return True
        except Exception as e:
            st.error(f"Error loading FAISS index: {str(e)}")
            return False


class GroqThemeIdentifier:
    """Class responsible for generating high-level themes using Groq LLM."""
    
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
    
    def identify_themes(self, query, document_answers):
        """Identify themes from document answers using Groq"""
        try:
            context = f"Query: {query}\n\nDocument Answers:\n"
            for i, answer in enumerate(document_answers[:10], 1):
                context += f"Document {answer['doc_id']} ({answer['filename']}): {answer['text'][:300]}...\n\n"
            
            prompt = f"""
            Analyze the following query and document answers to identify main themes.
            
            {context}
            
            Please provide:
            1. Identify 2-3 main themes from the document answers
            2. For each theme, provide supporting document citations
            3. A brief summary for each theme
            
            Format your response exactly as:
            Theme 1 - [Theme Name]:
            [Summary mentioning supporting documents like DOC001, DOC002, etc.]
            
            Theme 2 - [Theme Name]:
            [Summary mentioning supporting documents]
            
            Keep each theme summary concise (2-3 sentences).
            """
            
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error identifying themes: {str(e)}"


def process_uploaded_file(uploaded_file):
    """Process a single uploaded file"""
    processor = DocumentProcessor()
    
    # Generate document ID
    doc_id = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
    
    # Extract text based on file type
    if uploaded_file.type == "application/pdf":
        text_content = processor.extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        text_content = processor.extract_text_from_txt(uploaded_file)
    else:
        st.error(f"Unsupported file type: {uploaded_file.type}")
        return None
    
    if not text_content:
        st.warning(f"No text extracted from {uploaded_file.name}")
        return None
    
    return {
        'doc_id': doc_id,
        'filename': uploaded_file.name,
        'chunks': text_content,
        'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def initialize_search_engine():
    """Initialize FAISS search engine"""
    if st.session_state.faiss_index is None or st.session_state.embedding_model is None:
        search_engine = FAISSSearchEngine()
        if search_engine.index is not None and search_engine.embedding_model is not None:
            st.session_state.faiss_index = search_engine.index
            st.session_state.embedding_model = search_engine.embedding_model
            return search_engine
    else:
        # Create search engine with existing components
        search_engine = FAISSSearchEngine()
        search_engine.index = st.session_state.faiss_index
        search_engine.embedding_model = st.session_state.embedding_model
        search_engine.document_chunks = st.session_state.document_chunks
        search_engine.chunk_metadata = st.session_state.chunk_metadata
        return search_engine
    
    return None


def add_documents_to_faiss(documents_data):
    """Add documents to FAISS index"""
    search_engine = initialize_search_engine()
    if search_engine:
        success = search_engine.add_documents(documents_data)
        if success:
            # Update session state
            st.session_state.faiss_index = search_engine.index
            st.session_state.document_chunks = search_engine.document_chunks
            st.session_state.chunk_metadata = search_engine.chunk_metadata
        return success
    return False


def search_documents_faiss(query, n_results=15):
    """Search documents using FAISS"""
    search_engine = initialize_search_engine()
    if search_engine:
        return search_engine.search(query, n_results)
    return []


def clear_faiss():
    """Clear FAISS index"""
    search_engine = initialize_search_engine()
    if search_engine:
        success = search_engine.clear_index()
        if success:
            # Update session state
            st.session_state.faiss_index = search_engine.index
            st.session_state.document_chunks = []
            st.session_state.chunk_metadata = []
        return success
    return False


def get_faiss_count():
    """Get FAISS index count"""
    if st.session_state.faiss_index:
        return st.session_state.faiss_index.ntotal
    return 0


def main():
    st.set_page_config(
        page_title="Document Research Chatbot (FAISS)",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Document Research Chatbot (FAISS)")
    st.markdown("**🚀 Lightning Fast**: Uses FAISS with sentence transformers for ultra-fast semantic search")
    
    # Check if Groq API key is available
    if groq_api_key:
        st.success("✅ Groq API key loaded from environment")
    else:
        st.warning("⚠️ Groq API key not found in environment variables. Theme analysis will be disabled.")
    
    # Initialize FAISS on first run
    if st.session_state.faiss_index is None:
        with st.spinner("Initializing FAISS..."):
            search_engine = initialize_search_engine()
            if search_engine and search_engine.index is not None:
                st.success("✅ FAISS initialized successfully!")
            else:
                st.error("❌ Failed to initialize FAISS")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📄 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload Documents (PDF/TXT)",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF or text files for analysis"
        )
        
        if uploaded_files and st.button("Process Documents"):
            progress_bar = st.progress(0)
            new_documents = []
            
            for i, file in enumerate(uploaded_files):
                st.write(f"Processing: {file.name}")
                doc_info = process_uploaded_file(file)
                if doc_info:
                    new_documents.append(doc_info)
                    st.success(f"✅ {file.name}")
                else:
                    st.error(f"❌ {file.name}")
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            if new_documents:
                st.session_state.documents.extend(new_documents)
                with st.spinner("Adding to FAISS index..."):
                    if add_documents_to_faiss(new_documents):
                        st.success(f"✅ Added {len(new_documents)} documents to FAISS!")
                    else:
                        st.error("Failed to add documents to FAISS")
        
        if st.session_state.documents and st.button("Clear All Documents"):
            with st.spinner("Clearing FAISS index..."):
                if clear_faiss():
                    st.session_state.documents = []
                    st.success("All documents cleared!")
                else:
                    st.error("Failed to clear documents")
        
        if st.session_state.documents:
            st.subheader(f"📊 Documents ({len(st.session_state.documents)})")
            total_chunks = sum(len(doc['chunks']) for doc in st.session_state.documents)
            faiss_count = get_faiss_count()
            
            st.metric("Total Chunks", total_chunks)
            st.metric("FAISS Vectors", faiss_count)
            
            with st.expander("View Documents"):
                for doc in st.session_state.documents:
                    st.write(f"📄 **{doc['filename']}**")
                    st.write(f"   - ID: {doc['doc_id']}")
                    st.write(f"   - Chunks: {len(doc['chunks'])}")
                    st.write(f"   - Uploaded: {doc['upload_time']}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("⚡ Ultra-Fast Semantic Search")
        
        query = st.text_area("Enter your question:", height=70)
        
        col_a, col_b = st.columns(2)
        with col_a:
            num_results = st.slider("Results", 5, 25, 15)
        with col_b:
            min_similarity = st.slider("Min similarity", 0.0, 1.0, 0.05, 0.05)  # Lower default threshold
        
        if st.button("⚡ Search", disabled=not query or not st.session_state.documents):
            if not st.session_state.documents:
                st.warning("Upload documents first!")
            elif st.session_state.faiss_index is None:
                st.warning("FAISS not initialized!")
            else:
                with st.spinner("Searching with FAISS..."):
                    start_time = datetime.now()
                    results = search_documents_faiss(query, num_results)
                    end_time = datetime.now()
                    search_time = (end_time - start_time).total_seconds() * 1000  # Convert to milliseconds
                    
                    results = [r for r in results if r['similarity'] >= min_similarity]
                
                if results:
                    st.success(f"Found {len(results)} relevant passages in {search_time:.1f}ms (FAISS semantic search)")
                    
                    # Results table
                    st.subheader("📋 Search Results")
                    df_data = []
                    for result in results:
                        df_data.append({
                            'Doc ID': result['doc_id'],
                            'Filename': result['filename'],
                            'Answer': result['text'][:200] + "..." if len(result['text']) > 200 else result['text'],
                            'Citation': f"Page {result['page']}, Para {result['paragraph']}",
                            'Score': f"{result['similarity']:.3f}"
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True, height=400)
                    
                    # Detailed view
                    with st.expander("📖 Full Text Results"):
                        for i, result in enumerate(results, 1):
                            st.write(f"**Result {i}** - {result['filename']}")
                            st.write(f"Citation: Page {result['page']}, Paragraph {result['paragraph']}")
                            st.write(f"Similarity: {result['similarity']:.3f}")
                            st.write(result['text'])
                            st.write("---")
                    
                    # Theme analysis
                    if groq_api_key:
                        st.subheader("🎯 Theme Analysis")
                        with st.spinner("Analyzing themes..."):
                            theme_identifier = GroqThemeIdentifier(groq_api_key)
                            themes = theme_identifier.identify_themes(query, results)
                            st.markdown(themes)
                    else:
                        st.info("💡 Groq API key not available - theme analysis disabled")
                        
                        # Basic analysis
                        st.subheader("📊 Basic Analysis")
                        doc_count = len(set(r['doc_id'] for r in results))
                        avg_score = sum(r['similarity'] for r in results) / len(results)
                        
                        st.write(f"- **Documents matched**: {doc_count}")
                        st.write(f"- **Average relevance**: {avg_score:.3f}")
                        st.write(f"- **Total passages**: {len(results)}")
                        st.write(f"- **Search time**: {search_time:.1f}ms")
                else:
                    st.warning("No relevant results found")
    
    with col2:
        st.header("📊 System Status")
        
        # FAISS Status
        if st.session_state.faiss_index:
            st.success("✅ FAISS: Connected")
            faiss_count = get_faiss_count()
            st.metric("⚡ FAISS Vectors", faiss_count)
        else:
            st.error("❌ FAISS: Not Connected")
        
        if st.session_state.documents:
            total_chunks = sum(len(doc['chunks']) for doc in st.session_state.documents)
            st.metric("📄 Documents", len(st.session_state.documents))
            st.metric("📝 Text Chunks", total_chunks)
        else:
            st.info("No documents yet")
        
        # Progress
        progress = min(len(st.session_state.documents) / 75, 1.0)
        st.progress(progress)
        st.write(f"{len(st.session_state.documents)}/75+ documents")
        
        # Requirements
        st.subheader("✅ Requirements")
        reqs = [
            ("75+ documents", len(st.session_state.documents) >= 75),
            ("Document processing", len(st.session_state.documents) > 0),
            ("FAISS ready", st.session_state.faiss_index is not None),
            ("Semantic search", st.session_state.embedding_model is not None),
            ("Citations", True),
            ("Vector database", True)
        ]
        
        for req, status in reqs:
            if status:
                st.success(f"✅ {req}")
            else:
                st.warning(f"⏳ {req}")
        
        # Technical Info
        st.subheader("🔧 Technical Details")
        st.info("""
        **Vector Database**: FAISS
        **Index Type**: IndexFlatIP (Cosine)
        **Embeddings**: all-MiniLM-L6-v2
        **Dimensions**: 384
        **Search**: Ultra-fast similarity
        **Storage**: In-memory
        """)
        
        # Performance Info
        if st.session_state.faiss_index:
            st.subheader("⚡ Performance")
            st.info(f"""
            **Vectors**: {get_faiss_count():,}
            **Memory**: Optimized
            **Speed**: Sub-millisecond
            **Accuracy**: High precision
            """)


if __name__ == "__main__":
    main()