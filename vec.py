from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma 
from langchain_core.documents import Document
import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import uuid

# Existing code for initializing the vector store
df = pd.read_csv("FATF40.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        # Create more contextualized documents by adding more metadata
        document = Document(
            page_content=row["Content"],
            metadata={"title": row["Title"], "source": "FATF40", "row_id": i},
            id=str(i)
        )
        
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="fatf_40",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance for diverse results
    search_kwargs={"k": 5, "fetch_k": 10}  # Fetch more, return fewer but more diverse
)

# New functions for PDF processing and vector store updating

def process_pdf(pdf_file):
    """Extract text from PDF and split into chunks"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Extract text from PDF
        pdf_reader = PdfReader(tmp_path)
        text = ""
        
        # This is the correct way to iterate through pages in PyPDF2
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        
        # Get PDF filename from the uploaded file if available
        try:
            pdf_name = pdf_file.name
        except:
            pdf_name = "uploaded_pdf"
        
        # Split the extracted text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Create document objects
        documents = []
        for i, chunk in enumerate(chunks):
            # Generate unique ID using UUID to avoid collisions with existing docs
            doc_id = f"pdf_{uuid.uuid4().hex[:8]}_{i}"
            
            # Create a Document with appropriate metadata
            document = Document(
                page_content=chunk,
                metadata={
                    "title": f"{pdf_name} - Section {i+1}",
                    "source": "user_pdf",
                    "pdf_name": pdf_name,
                    "chunk_id": i
                },
                id=doc_id
            )
            documents.append(document)
        
        return documents
    
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)

def add_pdf_to_vectorstore(pdf_file):
    """
    Process a PDF file and add its content to the vector store
    
    Args:
        pdf_file: A file-like object containing PDF data
        
    Returns:
        int: Number of document chunks added to the vector store
    """
    # Process the PDF into Document objects
    documents = process_pdf(pdf_file)
    
    # Generate IDs for the documents
    ids = [doc.id for doc in documents]
    
    # Add the documents to the vector store
    vector_store.add_documents(documents=documents, ids=ids)
    
    # Return the number of documents added
    return len(documents)

def filter_retriever_by_source(source_type):
    """
    Update the retriever to filter by source type
    
    Args:
        source_type: String indicating the source to filter by ('FATF40', 'user_pdf', or None for both)
    """
    global retriever
    
    if source_type is None:
        # Remove filter to search all documents
        if "filter" in retriever.search_kwargs:
            retriever.search_kwargs.pop("filter")
    else:
        # Add filter to search only specified source
        retriever.search_kwargs["filter"] = {"source": source_type}
    
    return retriever