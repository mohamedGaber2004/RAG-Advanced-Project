# RAG Advanced Project

An advanced **Retrieval-Augmented Generation (RAG)** agent powered by LangGraph, Groq LLM, and Pinecone vector database. This intelligent system combines document retrieval, web search capabilities, and multi-modal language processing to provide accurate, context-aware answers.

## 🌟 Features

- **Multi-Source Information Retrieval**: Combines knowledge from uploaded documents (RAG) and real-time web search
- **Intelligent Routing**: Smart decision-making to route queries to RAG, web search, or direct answering
- **LangGraph Agent**: State-based workflow management with memory persistence
- **Pinecone Vector Database**: Fast semantic search over document embeddings
- **Web Search Integration**: Real-time information retrieval via Tavily API
- **Groq LLM**: High-performance language models for routing, judgment, and answer generation
- **RESTful API**: FastAPI backend for seamless integration
- **Document Upload**: Support for PDF documents with automatic chunking and embedding
- **Session Management**: Per-session conversation tracking with LangGraph checkpoints

## 🛠️ Technology Stack

### Core Technologies
- **LangGraph**: Agentic workflow orchestration
- **LangChain**: LLM framework and integrations
- **Groq**: High-speed language model inference
- **Pinecone**: Vector database for semantic search
- **FastAPI**: Modern web framework for APIs
- **Streamlit**: UI framework (optional)

### Embeddings & Processing
- **HuggingFace Embeddings**: Text vectorization
- **Ollama Embeddings**: Local embedding models
- **RecursiveCharacterTextSplitter**: Intelligent document chunking
- **PyPDF & unstructured**: Document parsing

### External Services
- **Tavily Search API**: Web search capabilities
- **Groq Cloud**: Language model API

## 📋 Prerequisites

- Python 3.13 or higher
- API Keys:
  - `GROQ_API_KEY`: From [Groq Console](https://console.groq.com)
  - `TAVILY_API_KEY`: From [Tavily API](https://tavily.com)
  - `PINECONE_API_KEY`: From [Pinecone Console](https://app.pinecone.io)
  - (Optional) `LANGSMITH_API_KEY`: For LangSmith monitoring

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RAG-Advanced-Project
   ```

2. **Create a Python virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key
   TAVILY_API_KEY=your_tavily_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   LANGSMITH_API_KEY=your_langsmith_api_key  # Optional
   ```

## 📂 Project Structure

```
RAG-Advanced-Project/
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
├── .env                   # Environment variables (create this)
└── backend/
    ├── main.py            # FastAPI application
    ├── agent.py           # LangGraph agent logic
    ├── Config.py          # Configuration and API keys
    └── vector_store.py    # Pinecone integration
```

## 🔧 Configuration

### [Config.py](backend/Config.py)

The configuration file loads API keys from environment variables:

```python
GROQ_API_KEY         # Groq API key
TAVILY_API_KEY       # Tavily search API key
PINECONE_API_KEY     # Pinecone database key
PINECONE_ENVIRONMENT # AWS region (default: us-east-1)
PINECONE_INDEX_NAME  # Vector index name (default: rag-index)
```

## 🤖 Agent Architecture

The agent follows a sophisticated decision-making pipeline:

1. **Router**: Decides whether to use RAG, web search, or answer directly
2. **RAG Pipeline**: Retrieves relevant documents from Pinecone
3. **Web Search**: Fetches real-time information via Tavily
4. **Judge**: Evaluates if retrieved information is sufficient
5. **Answerer**: Generates final response based on available context

### Agent State

```python
AgentState = {
    messages: List[BaseMessage],      # Conversation history
    route: Literal['rag','web','answer','end'],  # Current routing decision
    rag: str,                         # RAG retrieval results
    web: str,                         # Web search results
    web_search_enabled: bool          # Toggle web search capability
}
```

## 🔌 API Endpoints

### 1. Upload Document
```http
POST /upload-document/
Content-Type: multipart/form-data

Request:
- file: PDF file

Response:
{
  "message": "PDF successfully uploaded and indexed.",
  "filename": "document.pdf",
  "processed_chunks": 42
}
```

### 2. Chat with Agent
```http
POST /chat/
Content-Type: application/json

Request:
{
  "session_id": "user-session-123",
  "query": "What is RAG?",
  "enable_web_search": true
}

Response:
{
  "response": "RAG (Retrieval-Augmented Generation) is...",
  "trace_events": [
    {
      "step": 1,
      "node_name": "router",
      "description": "Routing decision",
      "event_type": "decision",
      "details": {"route": "rag"}
    }
  ]
}
```

## 🏃 Running the Application

### Backend API Server
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

### Main Entry Point
```bash
python main.py
```

### Streamlit UI (Optional)
```bash
streamlit run frontend/app.py  # If Streamlit UI is available
```

## 📚 Document Management

### Adding Documents to RAG

**Via API:**
1. Upload PDF files using the `/upload-document/` endpoint
2. Automatic chunking using `RecursiveCharacterTextSplitter` (1000 chars with 200 char overlap)
3. Embeddings generated using Ollama's nomic-embed-text model
4. Vectors stored in Pinecone for fast retrieval

**Programmatically:**
```python
from backend.vector_store import add_document_to_vectorstore

text_content = "Your document text here..."
add_document_to_vectorstore(text_content)
```

## 🔍 Vector Store Details

- **Index Name**: `rag-index`
- **Embedding Model**: `nomic-embed-text` (384-dimensional)
- **Dimension**: 384
- **Distance Metric**: Cosine similarity
- **Deployment**: Pinecone Serverless on AWS (us-east-1)
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters

## 🌐 Web Search Integration

The agent uses **Tavily Search API** for real-time information:
- Up to 3 results per search
- Topic: General knowledge
- Automatically triggered based on routing decision
- Results integrated into agent context

## 💡 Usage Example

```python
from backend.agent import rag_agent
from langchain_core.messages import HumanMessage

# Create query
message = HumanMessage(content="Explain quantum computing")

# Run agent
result = rag_agent.invoke(
    {"messages": [message], "web_search_enabled": True},
    {"configurable": {"thread_id": "user-123"}}
)

print(result["messages"][-1].content)
```

## 📊 Trace Events

The API returns detailed trace events for debugging and monitoring:

```python
{
  "step": int,              # Sequential step number
  "node_name": str,         # Agent node name (router, rag_retriever, etc.)
  "description": str,       # Human-readable description
  "event_type": str,        # Type: decision, retrieval, processing, etc.
  "details": dict           # Additional metadata
}
```

## 🔐 Security Considerations

- Never commit `.env` files containing API keys
- Use environment variables for all sensitive information
- Validate and sanitize user inputs
- Consider rate limiting in production
- Use HTTPS in production deployments

## 🐛 Troubleshooting

### Pinecone Connection Issues
- Verify `PINECONE_API_KEY` is correct
- Check network connectivity to Pinecone servers
- Ensure index exists or auto-create is enabled

### Groq API Errors
- Verify `GROQ_API_KEY` is valid and has available quota
- Check rate limits and retry logic

### Web Search Not Working
- Verify `TAVILY_API_KEY` is active
- Check API rate limits

### Embedding Model Issues
- Ensure Ollama is running if using local embeddings
- Verify HuggingFace model availability

## 📝 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Groq language model API key | ✅ |
| `TAVILY_API_KEY` | Tavily web search API key | ✅ |
| `PINECONE_API_KEY` | Pinecone database API key | ✅ |
| `LANGSMITH_API_KEY` | LangSmith monitoring (optional) | ❌ |
| `PINECONE_ENVIRONMENT` | AWS region for Pinecone | ❌ |

## 🚀 Performance Optimization

- **Caching**: Implement response caching for similar queries
- **Batching**: Process multiple documents in batch for efficiency
- **Index Optimization**: Regularly update vector index for better retrieval
- **Rate Limiting**: Implement per-user rate limits in production
- **Connection Pooling**: Use connection pools for database operations

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

See [LICENSE](LICENSE) file for details.

## 📞 Support

For issues, questions, or suggestions:
1. Check existing documentation
2. Review troubleshooting section
3. Open an issue with detailed information

## 🔗 Related Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Groq Documentation](https://console.groq.com/docs)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Tavily Search API](https://tavily.com/)