import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import re
import pandas as pd

# Import your existing vector retrieval system plus the new PDF functions
from vec import retriever, df, vector_store, add_pdf_to_vectorstore, filter_retriever_by_source

# Set page configuration
st.set_page_config(
    page_title="FATF40 Recommendations Assistant",
    page_icon="💼",
    layout="centered"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .stApp {
        max-width: 1000px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e3f2fd;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the Ollama LLM
@st.cache_resource
def get_llm():
    return OllamaLLM(model='llama3.2')

model = get_llm()

# Define the prompt template (modified to handle both FATF40 and PDF content)
template = """
You are an expert assistant that provides accurate, concise answers based on provided documentation.

Below are relevant sections from the documentation:

{reviews}

Question: {question}

Instructions:
1. Provide a clear, direct answer based solely on the content provided
2. If information is partial or unclear in the provided context, acknowledge this
3. Include specific recommendation numbers or section references when applicable
4. Keep your answers concise and focused
5. Do not make up information or extrapolate beyond what's in the provided context
6. If the answer comes from FATF40 recommendations, mention the specific recommendation number
7. If the answer comes from a user-uploaded PDF, mention this in your response

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

if "knowledge_source" not in st.session_state:
    st.session_state.knowledge_source = "FATF40"  # Default to FATF40

def get_relevant_context(question, default_k=5):
    """Get relevant context with adaptive retrieval strategy"""
    # Try to identify if the question is about a specific recommendation when in FATF40 mode
    if st.session_state.knowledge_source == "FATF40" or st.session_state.knowledge_source == "Both":
        rec_match = re.search(r'recommendation\s+(\d+)', question.lower())
        
        if rec_match:
            rec_num = int(rec_match.group(1))
            # If asking about specific recommendation, try to find it in the dataframe
            matches = df[df['Title'].str.contains(f'Recommendation {rec_num}', case=False, na=False) | 
                         df['Content'].str.contains(f'Recommendation {rec_num}', case=False, na=False)]
            
            if not matches.empty:
                # We found a specific match, create a document from it
                from langchain_core.documents import Document
                specific_docs = [Document(page_content=row["Content"], 
                                        metadata={"title": row["Title"], "source": "FATF40"})
                               for _, row in matches.iterrows()]
                
                # Still get some general context
                general_docs = retriever.invoke(question)
                
                # Combine, prioritizing the specific recommendation
                return specific_docs + general_docs
    
    # Default retrieval strategy
    return retriever.invoke(question)

def format_reviews(reviews):
    """Format reviews into more readable sections"""
    formatted = ""
    for i, doc in enumerate(reviews):
        # Get source information for context
        source = doc.metadata.get("source", "Unknown")
        title = doc.metadata.get("title", f"Section {i+1}")
        
        # Format with source information
        if source == "FATF40":
            formatted += f"--- {title} ---\n{doc.page_content}\n\n"
        elif source == "user_pdf":
            pdf_name = doc.metadata.get("pdf_name", "User PDF")
            formatted += f"--- {title} (From: {pdf_name}) ---\n{doc.page_content}\n\n"
        else:
            formatted += f"--- {title} ---\n{doc.page_content}\n\n"
    
    return formatted

# App title and description
st.title("FATF40 Recommendations Assistant")
st.markdown("""
This application helps you explore the Financial Action Task Force (FATF) 40 Recommendations
by answering your questions based on the official documentation. You can also upload your own 
PDF documents to get answers from them.
""")

# Sidebar with PDF upload and configuration options
with st.sidebar:
    st.subheader("Document Settings")
    
    # PDF Upload Section
    st.subheader("Upload Your Document")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Process the PDF and add to vector store
            num_chunks = add_pdf_to_vectorstore(uploaded_file)
            st.session_state.pdf_uploaded = True
            st.success(f"PDF processed! Added {num_chunks} document chunks to the knowledge base.")
    
    # Knowledge source selection (only show options if PDF is uploaded)
    if st.session_state.pdf_uploaded:
        st.subheader("Knowledge Source")
        source_options = ["FATF40", "Uploaded PDF", "Both"]
        selected_source = st.radio(
            "Select knowledge source for answers:",
            source_options
        )
        
        # Update retriever based on selection
        if selected_source == "FATF40":
            st.session_state.knowledge_source = "FATF40"
            filter_retriever_by_source("FATF40")
        elif selected_source == "Uploaded PDF":
            st.session_state.knowledge_source = "PDF"
            filter_retriever_by_source("user_pdf")
        else:  # Both
            st.session_state.knowledge_source = "Both"
            filter_retriever_by_source(None)  # No filter
    
    st.subheader("About")
    st.markdown("""
    This assistant utilizes Retrieval-Augmented Generation (RAG) to instantly retrieve precise
    financial data from documents or uploaded files, then generates expert-level responses using 
    Llama 3.2. It simplifies complex financial tasks—including compliance, reporting, bookkeeping, 
    and strategic analysis—to help you streamline operations and make data-driven decisions with confidence.
    """)
    
    # Add a reset chat button
    if st.button("Reset Chat", key="reset"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about FATF40 recommendations or your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with a spinner while processing
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Update the source indicator based on current knowledge source
        source_indicator = ""
        if st.session_state.knowledge_source == "FATF40":
            source_indicator = "FATF40 recommendations"
        elif st.session_state.knowledge_source == "PDF":
            source_indicator = "your uploaded document"
        else:
            source_indicator = "all available documents"
            
        with st.spinner(f"Searching {source_indicator}..."):
            # Get context with adaptive retrieval strategy
            reviews = get_relevant_context(prompt)
            
            # Process reviews into a more structured format
            formatted_reviews = format_reviews(reviews)
            
            # Check if the reviews seem relevant
            if not reviews or len(reviews) == 0:
                response = f"I'm sorry, but I couldn't find relevant information in {source_indicator} to answer your question."
            else:
                # Get response from the LLM
                response = chain.invoke({"reviews": formatted_reviews, "question": prompt})
            
            # Display the response
            message_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})