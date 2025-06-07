import streamlit as st #Used to create the interactive frontend of the web application.
import pandas as pd #Used for structured data manipulation and analysis.
import os #Used for interacting with the operating system.
import warnings #To suppress or manage warnings.
from datetime import datetime #For working with dates and times. 
from groq import Groq # To interact with the Groq API
from typing import List, Dict, Any #Used in function definitions like def func(data: List[str]) -> Dict[str, Any]
import hashlib #For generating unique hashes.
import PyPDF2 #Reading and extracting text from PDFs 
from sklearn.feature_extraction.text import TfidfVectorizer #Vectorizing documents for similarity comparison
from sklearn.metrics.pairwise import cosine_similarity #Finding the most relevant document responses.

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

if 'vectorizer' not in st.session_state: #Holds the TF-IDF vectorizer instance after fitting it on the document chunks.
    st.session_state.vectorizer = None #So that the trained vectorizer can be reused for querying without retraining every time.

if 'document_vectors' not in st.session_state: #Stores the vectorized form of all document chunks.
    st.session_state.document_vectors = None #These are the TF-IDF vectors created using the vectorizer. Needed to compute cosine similarity between user queries and the stored chunks.

if 'document_chunks' not in st.session_state: #Stores a flat list of all text chunks from all uploaded documents.
    st.session_state.document_chunks = [] #Helps with search, display, and filtering the results after a query.


class DocumentProcessor: #This is a DocumentProcessor class that provides functions to extract paragraph-level text from PDF:PyPDF2, .txt files.
        
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        try: #Without a try-except block, if any error occurs during PDF reading or text extraction, the whole app or script would crash.
            pdf_reader = PyPDF2.PdfReader(pdf_file) #It opens and reads the PDF so you can access its pages and content.
            text_content = [] #Initializes an empty list to store extracted text data structured by page and paragraph.
            
            for page_num, page in enumerate(pdf_reader.pages, 1): #Loops through each page in the PDF.enumerate(..., 1) means counting pages starting at 1 (so first page is page number 1, not 0). page_num is the current page number, and page is the actual page object.
                page_text = page.extract_text() #Extracts the text content from the current page
                if page_text.strip(): #Checks if the extracted text is not empty or just whitespace 
                    # Split into paragraphs
                    paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()] #Keeps only paragraphs that are not empty after stripping
                    for para_num, paragraph in enumerate(paragraphs, 1): 
                        if paragraph and len(paragraph) > 20: #Filters out any empty paragraphs or paragraphs shorter than 21 characters, to keep meaningful content only.
                            text_content.append({ #Adds a dictionary to text_content list containing: page, paragraph, text.
                                'page': page_num,
                                'paragraph': para_num,
                                'text': paragraph
                            })
            return text_content
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return [] #In case of error, you return an empty list ([]), so the rest of your code can continue without problems.
    
    def extract_text_from_txt(self, txt_file):  # Same thing as previous function just for text file
        """Extract text from text file"""
        try:
            content = txt_file.read().decode('utf-8')
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            text_content = []
            
            for para_num, paragraph in enumerate(paragraphs, 1):
                if paragraph and len(paragraph) > 20:
                    text_content.append({
                        'page': 1,
                        'paragraph': para_num,
                        'text': paragraph
                    })
            return text_content
        except Exception as e:
            st.error(f"Error extracting text from TXT: {str(e)}")
            return []

class TFIDFSearchEngine: #This class will be responsible for searching documents using the TF-IDF technique.
    def __init__(self):
        self.vectorizer = TfidfVectorizer( #nitializes a TfidfVectorizer instance from scikit-learn, which converts text data into TF-IDF vectors.
            max_features=5000, #Limits the number of features (unique terms/phrases) to 5000 to reduce dimensionality and improve performance.
            stop_words='english', #Automatically removes common English stop words like “the”, “is”, “and” to focus on meaningful words.
            ngram_range=(1, 2),#Considers both single words (unigrams) and pairs of words (bigrams) as features.
            max_df=0.8, #Ignores terms that appear in more than 80% of documents, as they are likely too common.
            min_df=2 #Ignores terms that appear in fewer than 2 documents, to remove rare words.
        )
        self.document_vectors = None
        self.document_chunks = []
    
    # This method accepts a list of document data (each document having multiple chunks).
    # It extracts all the chunk texts and associated metadata into two parallel lists: all_texts and all_metadata.
    # This prepares the text and metadata for further processing, like vectorization or indexing in a search engine.
    def add_documents(self, documents_data):
        """Add documents to the search index"""
        # Initialize empty lists to store all text chunks and their metadata
        all_texts = []
        all_metadata = []
        
        # Loop over each document's data in the input list
        for doc_data in documents_data:
            # Each document contains multiple chunks of text
            for chunk in doc_data['chunks']:
                # Append the text from each chunk to all_texts list
                all_texts.append(chunk['text'])
                # Append a metadata dictionary for each chunk with info about.
                all_metadata.append({
                    'doc_id': doc_data['doc_id'],
                    'filename': doc_data['filename'],
                    'page': chunk['page'],
                    'paragraph': chunk['paragraph'],
                    'text': chunk['text']
                })
        
        if all_texts:
            # If the list all_texts is not empty (i.e., there is some text to process)
            # Use the TF-IDF vectorizer to fit and transform the collected texts into vectors
            # This creates a sparse matrix where each row corresponds to a document chunk vector
            self.document_vectors = self.vectorizer.fit_transform(all_texts)
            # Store the metadata corresponding to each text chunk
            self.document_chunks = all_metadata
            # Return True to indicate that documents were successfully added and processed
            return True
        return False
    

    def search(self, query, n_results=15):
        """Search for relevant documents"""
        # If no document vectors exist yet, return empty list immediately
        if self.document_vectors is None:
            return []
        
        try:
            # Convert the query string into a TF-IDF vector using the trained vectorizer
            query_vector = self.vectorizer.transform([query])
            # Compute cosine similarity between query vector and all document vectors
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
            
            # Find indices of top n_results highest similarity scores (sorted descending)
            top_indices = similarities.argsort()[-n_results:][::-1]
            
            results = []
            for idx in top_indices:
                # Only include results with similarity greater than 0.1 (threshold)
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    # Copy metadata of the matched chunk to avoid modifying original
                    chunk_data = self.document_chunks[idx].copy()
                    # Add similarity score to the metadata dictionary
                    chunk_data['similarity'] = similarities[idx]
                    # Add this chunk’s metadata and similarity to results list
                    results.append(chunk_data)
            # Return the list of top matching document chunks with their similarity scores
            return results
        except Exception as e:
            # In case of any error during search, show error and return empty list
            st.error(f"Search error: {str(e)}")
            return []

class GroqThemeIdentifier: #This class is responsible for generating high-level themes from a set of document answers, using a Groq-hosted LLM (like LLaMA 3).
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key) #It creates a Groq client instance using the provided API key for further communication with Groq's models.
    
    def identify_themes(self, query, document_answers): 
        #This method is used to extract meaningful themes from the document_answers given a query
        """Identify themes from document answers using Groq"""
        try:
            #Starts building a context string that includes the query and some relevant document chunks to provide background to the LLM.
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
            #Sends the prompt to the Groq LLaMA 3 (8B) mode
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error identifying themes: {str(e)}"

#To extract structured text chunks from a user-uploaded PDF or TXT file and return a dictionary with metadata.
def process_uploaded_file(uploaded_file):
    """Process a single uploaded file"""
    #Instantiates the DocumentProcessor class (you defined earlier) to access the PDF or TXT extraction methods.
    processor = DocumentProcessor()
    
    # Generate document ID
    # Generates a unique ID for the document based on its filename.
    # Helps in tracking documents internally.
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

#This function re-creates the TF-IDF vector index whenever new documents are added.
# Useful for making sure the search engine reflects the latest uploaded content.
def rebuild_search_index():
    """Rebuild the TF-IDF search index"""
    if not st.session_state.documents:
        return False
    
    search_engine = TFIDFSearchEngine()
    if search_engine.add_documents(st.session_state.documents):
        #Stores everything in st.session_state so other parts of the app (like search) can access:
        # The vectorizer (to encode queries)
        # The actual document vectors
        # The chunk-level metadata
        st.session_state.vectorizer = search_engine.vectorizer
        st.session_state.document_vectors = search_engine.document_vectors
        st.session_state.document_chunks = search_engine.document_chunks
        return True
    return False

def search_documents(query, n_results=15):
    """Search documents using TF-IDF"""
    if st.session_state.vectorizer is None or st.session_state.document_vectors is None:
        return []
    
    try:
        query_vector = st.session_state.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, st.session_state.document_vectors).flatten()
        
        top_indices = similarities.argsort()[-n_results:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                chunk_data = st.session_state.document_chunks[idx].copy()
                chunk_data['similarity'] = similarities[idx]
                results.append(chunk_data)
        
        return results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def main():
    st.set_page_config(
        page_title="Document Research Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Document Research Chatbot")
    st.markdown("**🚀 Fast & Simple**: Uses TF-IDF for search")
    
    # Check if Groq API key is available
    if groq_api_key:
        st.success("✅ Groq API key loaded from environment")
    else:
        st.warning("⚠️ Groq API key not found in environment variables. Theme analysis will be disabled.")
    
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
                with st.spinner("Building search index..."):
                    if rebuild_search_index():
                        st.success(f"✅ Processed {len(new_documents)} documents!")
                    else:
                        st.error("Failed to build search index")
        
        if st.session_state.documents and st.button("Clear All Documents"):
            st.session_state.documents = []
            st.session_state.vectorizer = None
            st.session_state.document_vectors = None
            st.session_state.document_chunks = []
            st.success("All documents cleared!")
        
        if st.session_state.documents:
            st.subheader(f"📊 Documents ({len(st.session_state.documents)})")
            total_chunks = sum(len(doc['chunks']) for doc in st.session_state.documents)
            st.metric("Total Chunks", total_chunks)
            
            with st.expander("View Documents"):
                for doc in st.session_state.documents:
                    st.write(f"📄 **{doc['filename']}**")
                    st.write(f"   - ID: {doc['doc_id']}")
                    st.write(f"   - Chunks: {len(doc['chunks'])}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Search Documents")
        
        query = st.text_area("Enter your question:", height=70)
        
        col_a, col_b = st.columns(2)
        with col_a:
            num_results = st.slider("Results", 5, 25, 15)
        with col_b:
            min_similarity = st.slider("Min similarity", 0.0, 1.0, 0.1, 0.05)
        
        if st.button("🔍 Search", disabled=not query or not st.session_state.documents):
            if not st.session_state.documents:
                st.warning("Upload documents first!")
            elif st.session_state.vectorizer is None:
                st.warning("Please rebuild search index!")
            else:
                with st.spinner("Searching..."):
                    results = search_documents(query, num_results)
                    results = [r for r in results if r['similarity'] >= min_similarity]
                
                if results:
                    st.success(f"Found {len(results)} relevant passages")
                    
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
                else:
                    st.warning("No relevant results found")
    
    with col2:
        st.header("📊 System Status")
        
        if st.session_state.documents:
            total_chunks = sum(len(doc['chunks']) for doc in st.session_state.documents)
            st.metric("📄 Documents", len(st.session_state.documents))
            st.metric("📝 Text Chunks", total_chunks)
            
            if st.session_state.vectorizer is not None:
                st.metric("🔧 Status", "Ready")
            else:
                st.metric("🔧 Status", "Need Index")
                if st.button("🔨 Build Search Index"):
                    with st.spinner("Building..."):
                        if rebuild_search_index():
                            st.success("Index built!")
                            st.rerun()
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
            ("Search ready", st.session_state.vectorizer is not None),
            ("Citations", True),
            ("Free tools", True)
        ]
        
        for req, status in reqs:
            if status:
                st.success(f"✅ {req}")
            else:
                st.warning(f"⏳ {req}")

if __name__ == "__main__":
    main()