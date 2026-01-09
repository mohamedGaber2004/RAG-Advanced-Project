import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY=os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = "us-east-1"
PINECONE_IDNEX_NAME = "rag-index"