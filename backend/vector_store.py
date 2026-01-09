import os
from pinecone import Pinecone , ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HugginFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

from Config import PINECONE_API_KEY

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

pc = Pinecone(api_key = PINECONE_API_KEY)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
INDEX_NAME = 'rag-index'

def get_retriever():
    """Initializes and returns the pinecone vectorestore retriever"""
    if INDEX_NAME not in pc.list_indexes().names():
        print("Creating new index")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws',region='us-east-1')
        )
        print("Created pinecone index")
    
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME,embedding = embeddings)
    return vectorstore.as_retriever()



def add_document(text_cont:str):
    """
    Docstring for add_document
    
    :param text_doc: Description
    :type text_doc: str
    """
    if not text_cont : 
        raise ValueError("Document Content Cannot be embty!")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )


    documets = text_splitter.create_documents([text_cont])
    print("Splitting into chunks Done.")

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME,embedding = embeddings)

    vectorstore.add_documents(documets)

    print("Successfully added chunks to pinecone VS")