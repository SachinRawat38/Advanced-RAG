# FATF40 Recommendations Assistant

A Streamlit-powered chatbot that helps users explore and understand the Financial Action Task Force (FATF) 40 Recommendations through natural language queries.

![FATF40 Assistant Screenshot](https://img.shields.io/badge/FATF40-Assistant-blue)

## 📋 Overview

This application provides an interactive way to explore the FATF 40 Recommendations, which are international standards for combating money laundering, terrorist financing, and other related threats to the integrity of the international financial system. Users can ask questions in natural language, and the application will retrieve relevant information from the official FATF documentation and provide concise, accurate answers.

## 🚀 Features

- **Natural Language Queries**: Ask questions about FATF recommendations in plain English
- **Contextual Understanding**: The system can identify questions about specific recommendations
- **Retrieval-Augmented Generation (RAG)**: Combines vector search retrieval with LLM generation
- **Responsive UI**: Clean chat interface with message history
- **Local Execution**: Uses Ollama for running large language models locally

## 🛠️ Technology Stack

- **Streamlit**: For the web interface
- **LangChain**: For creating the RAG pipeline
- **Ollama**: For running the LLM locally (Llama 3.2)
- **ChromaDB**: As the vector database for document retrieval
- **Pandas**: For data manipulation

## 📦 Prerequisites

- Python 3.8+
- Ollama installed and running locally with Llama 3.2 model
- At least 16GB RAM recommended for running the LLM

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fatf40-assistant.git
   cd fatf40-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Ollama is installed and the required models are downloaded:
   ```bash
   # Install Ollama (if not already installed)
   # Visit https://ollama.com/ for installation instructions
   
   # Pull the required models
   ollama pull llama3.2
   ollama pull mxbai-embed-large
   ```

4. Ensure you have the FATF40.csv file in the root directory of the project.

## 🚀 Usage

1. Start the Streamlit application:
   ```bash
   streamlit run chatbot.py
   ```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Start asking questions about FATF recommendations in the chat input field

## 💾 Data

The application uses a CSV file (`FATF40.csv`) that contains the FATF 40 Recommendations broken down into structured content. The file should have at least the following columns:
- `Title`: The title or identifier of the section
- `Content`: The actual text content of the recommendation or explanatory material

## 🔄 How It Works

1. **Vector Database Creation**: On first run, the system processes the FATF40.csv file, creates embeddings for each section using mxbai-embed-large, and stores them in a ChromaDB vector database.

2. **Query Processing**: When a user asks a question:
   - The system analyzes the query for mentions of specific recommendation numbers
   - It retrieves relevant context using a combination of specific matching and semantic search
   - The retrieved context is formatted into a structured prompt

3. **Response Generation**: The formatted prompt is sent to the Llama 3.2 model, which generates a concise, accurate response based on the provided context

4. **UI Presentation**: The response is displayed in the chat interface and added to the conversation history

## ⚙️ Configuration

The main configuration parameters can be found at the top of the `chatbot.py` file, including:
- LLM model selection
- Number of documents to retrieve (k)
- Prompt template

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgements

- Financial Action Task Force (FATF) for their crucial work in establishing international standards for combating financial crimes
- The Llama, LangChain, and Streamlit communities for their excellent open-source tools
