# RAG_AppWith_Chat_History

A simple Retrieval-Augmented Generation (RAG) application with conversation history, built using LangChain, Google Gemini (via langchain-google-genai), ChromaDB, and HuggingFace embeddings. This project demonstrates how to combine document retrieval, LLMs, and chat history for context-aware question answering.

## Features
- **RAG Pipeline**: Retrieve relevant documents and generate answers using Gemini LLM.
- **Chroma Vector Store**: Store and search document embeddings efficiently.
- **HuggingFace Embeddings**: Use `sentence-transformers/all-MiniLM-L6-v2` for document and query embeddings.
- **Chat History**: Maintain and utilize conversation history for more context-aware responses.
- **Jupyter Notebooks**: Interactive exploration and demonstration in `app.ipynb` and `full_rag_pipeline.ipynb`.
- **Streamlit Interface**: User-friendly web interface for document upload and chat interaction.
- **Multi-format Document Support**: Process TXT, PDF, and DOCX files.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.

## Setup
1. **Clone the repository**
   ```sh
   git clone <your-repo-url>
   cd RAG_AppWith_Chat_History
   ```
2. **Create Virtual Environment**
    ```sh
    python -m venv .venv
    ```
    activate it through-
    ```sh
    .venv\Scripts\activate  # On Windows
    source .venv/bin/activate  # On macOS/Linux
    ```
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **Set up environment variables**
   - Create a `.env` file in the project root with the following content:
     ```env
     GOOGLE_API_KEY=your_google_gemini_api_key
     HF_TOKEN=your_huggingface_token  # if required for private models
     ```

## Usage

### Jupyter Notebooks
- Open `app.ipynb` or `full_rag_pipeline.ipynb` in Jupyter or VS Code.
- Run the cells step by step to initialize the RAG pipeline, load documents, and interact with the chatbot.
- Example queries:
  - "What is the capital of Telangana?"
  - "Can we say all gold is glitter?"

### Streamlit App
Run the Streamlit app with:

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501 by default.

#### How to Use the Streamlit App

1. **Upload Documents**: Use the sidebar to upload one or more documents (TXT, PDF, or DOCX).
2. **Process Documents**: Click the "Process Documents" button to extract text and create embeddings.
3. **Ask Questions**: Type your questions in the chat input at the bottom of the page.
4. **View Responses**: The chatbot will respond based on the content of your uploaded documents.
5. **Clear Chat History**: Use the "Clear Chat History" button in the sidebar to start a new conversation.

## File Structure
- `app.ipynb` — Simple RAG app with conversation history demo.
- `full_rag_pipeline.ipynb` — Full RAG pipeline demonstration.
- `app.py` — Streamlit application combining RAG and chat history.
- `requirements.txt` — Python dependencies.

## Notes
- Ensure your Google Gemini API key is valid and has access to the Gemini model.
- **IMP**: For best results, restart the Jupyter kernel after updating `.env`.
- The Streamlit app processes entire documents at once, which may not be optimal for very large documents.
- The quality of responses depends on the quality of the uploaded documents and the relevance of their content to the questions asked.
