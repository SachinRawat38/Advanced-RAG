import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import re
import pandas as pd

# Import your existing vector retrieval system
from vector import retriever, df

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

# Define the prompt template (same as in your original code)
template = """
You are an expert in Financial Action Task Force (FATF) recommendations and anti-money laundering standards. Your task is to provide accurate, concise answers based on the FATF40 recommendations.

Below are relevant sections from the FATF40 recommendations document:

{reviews}
TWQ 
Instructions:
1. Provide a clear, direct answer based solely on the FATF40 content provided
2. If information is partial or unclear in the provided context, acknowledge this
3. Include specific recommendation numbers when applicable
4. Keep your answers concise and focused
5. Do not make up information or extrapolate beyond what's in the provided context

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def get_relevant_context(question, default_k=5):
    """Get relevant context with adaptive retrieval strategy"""
    # Try to identify if the question is about a specific recommendation
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
        title = doc.metadata.get("title", f"Section {i+1}")
        formatted += f"--- {title} ---\n{doc.page_content}\n\n"
    return formatted

# App title and description
st.title("FATF40 Recommendations Assistant")
st.markdown("""
This application helps you explore the Financial Action Task Force (FATF) 40 Recommendations
by answering your questions based on the official documentation.
""")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
## Chat inputs user compliance query 
if prompt := st.chat_input("Ask about FATF40 recommendations..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response with a spinner while processing
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Searching and analyzing FATF recommendations..."):
            # Get context with adaptive retrieval strategy
            ## process query - extract key entities - general docs + specific docs 
            reviews = get_relevant_context(prompt)
            
            # Process reviews into a more structured format
            formatted_reviews = format_reviews(reviews)
            
            # Check if the reviews seem relevant
            if not reviews or len(reviews) == 0:
                response = "I'm sorry, but I couldn't find relevant information in the FATF40 recommendations to answer your question."
            else:
                # Get response from the LLM
                response = chain.invoke({"reviews": formatted_reviews, "question": prompt})
            
            # Display the response
            message_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with info
with st.sidebar:
    st.subheader("About")
    st.markdown("""
    This assistant uses RAG (Retrieval-Augmented Generation) to provide accurate 
    information about the FATF 40 Recommendations. It first retrieves relevant sections 
    from the FATF40 documentation and then uses Llama 3.2 to generate a response.
    """)
    
    
    # Add a reset chat button
    if st.button("Reset Chat", key="reset"):
        st.session_state.messages = []
        st.rerun()