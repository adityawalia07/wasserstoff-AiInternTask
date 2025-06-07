# 📚 Document Research Chatbot

A powerful Streamlit-based document analysis and search application that uses TF-IDF vectorization for fast document retrieval and Groq's LLaMA 3 model for intelligent theme analysis.

## ✨ Features

- **Multi-format Support**: Upload and process PDF and TXT files
- **Fast Search**: TF-IDF vectorization for quick document retrieval
- **Smart Chunking**: Automatic paragraph-level text extraction
- **Similarity Scoring**: Cosine similarity-based relevance ranking
- **Theme Analysis**: AI-powered theme identification using Groq's LLaMA 3
- **Interactive UI**: Clean, responsive Streamlit interface
- **Document Management**: Upload, process, and manage multiple documents
- **Detailed Citations**: Page and paragraph-level source citations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Groq API key (optional, for theme analysis)

### Installation

1. **Clone or download the application files**

2. **Install required dependencies**:
```bash
pip install streamlit pandas scikit-learn PyPDF2 groq python-dotenv
```

3. **Set up environment variables** (optional):
Create a `.env` file in the project directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

4. **Run the application**:
```bash
streamlit run app.py
```

## 📋 Requirements

```
streamlit
pandas
scikit-learn
PyPDF2
groq
python-dotenv
```

## 🏗️ Architecture

### Core Components

#### 1. DocumentProcessor Class
- **Purpose**: Extract text from PDF and TXT files
- **Methods**:
  - `extract_text_from_pdf()`: Processes PDF files using PyPDF2
  - `extract_text_from_txt()`: Handles plain text files
- **Features**: 
  - Paragraph-level chunking
  - Page tracking
  - Error handling

#### 2. TFIDFSearchEngine Class
- **Purpose**: Implement document search using TF-IDF vectorization
- **Configuration**:
  - Max features: 5,000
  - N-gram range: 1-2 (unigrams and bigrams)
  - Stop words: English
  - Document frequency: 2-80%
- **Methods**:
  - `add_documents()`: Index documents for search
  - `search()`: Find relevant passages using cosine similarity

#### 3. GroqThemeIdentifier Class
- **Purpose**: Generate intelligent theme analysis using LLaMA 3
- **Features**:
  - Context-aware analysis
  - Document citation integration
  - Structured theme extraction

## 🎯 Usage Guide

### Document Upload
1. Use the sidebar file uploader
2. Select PDF or TXT files (multiple files supported)
3. Click "Process Documents" to extract and index content
4. Wait for the search index to build

### Searching Documents
1. Enter your research question in the main text area
2. Adjust search parameters:
   - **Results**: Number of passages to return (5-25)
   - **Min Similarity**: Relevance threshold (0.0-1.0)
3. Click "🔍 Search" to find relevant passages

### Understanding Results
- **Search Results Table**: Quick overview with scores
- **Full Text Results**: Complete passage content with citations
- **Theme Analysis**: AI-generated themes and patterns (requires Groq API)

## 🔧 Configuration

### TF-IDF Parameters
```python
TfidfVectorizer(
    max_features=5000,      # Maximum vocabulary size
    stop_words='english',   # Remove common English words
    ngram_range=(1, 2),    # Use unigrams and bigrams
    max_df=0.8,            # Ignore terms in >80% of docs
    min_df=2               # Ignore terms in <2 docs
)
```

### Search Parameters
- **Similarity Threshold**: 0.1 (adjustable via UI)
- **Default Results**: 15 passages
- **Text Chunk Minimum**: 20 characters

## 📊 System Requirements

### Minimum Requirements
- **Documents**: 1+ (recommended: 75+)
- **Memory**: 2GB RAM
- **Storage**: Varies by document size

### Performance Optimization
- Vectorization is performed once per document set
- Session state maintains search index
- Efficient sparse matrix operations

## 🔑 API Integration

### Groq Setup
1. Sign up at [Groq](https://groq.com)
2. Generate an API key
3. Add to `.env` file or environment variables
4. Theme analysis will be automatically enabled

### Without Groq API
- Application works fully without API key
- Basic analysis statistics provided instead
- All search functionality remains available

## 📁 File Structure

```
document-chatbot/
├── app.py              # Main application file
├── .env               # Environment variables (optional)
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🐛 Troubleshooting

### Common Issues

**"No text extracted from file"**
- Ensure PDF is not image-based or encrypted
- Check file format compatibility
- Verify file is not corrupted

**"Search error" messages**
- Rebuild search index using sidebar button
- Check if documents are properly processed
- Verify vectorizer initialization

**Theme analysis not working**
- Confirm Groq API key is set correctly
- Check internet connection
- Verify API key permissions

### Error Handling
- Comprehensive try-catch blocks
- User-friendly error messages
- Graceful degradation when services unavailable

## 🔒 Security Notes

- API keys stored in environment variables
- No data persistence beyond session
- Local processing of documents
- No external data transmission (except Groq API)

## 🚀 Performance Tips

1. **Document Optimization**:
   - Clean, well-structured documents work best
   - Remove unnecessary formatting
   - Ensure good paragraph separation

2. **Search Optimization**:
   - Use specific, descriptive queries
   - Adjust similarity threshold for precision/recall balance
   - Experiment with result count

3. **System Performance**:
   - Process documents in batches for large collections
   - Clear documents when switching research topics
   - Monitor memory usage with large document sets

## 📈 Future Enhancements

- **Planned Features**:
  - Support for DOCX files
  - Advanced filtering options
  - Export functionality
  - Batch query processing
  - Vector database integration


