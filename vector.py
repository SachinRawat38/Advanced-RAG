from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma 
from langchain_core.documents import Document
import os
import pandas as pd

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