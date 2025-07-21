# RAG_AppWith_Chat_History

A simple Retrieval-Augmented Generation (RAG) application with conversation history, built using LangChain, Google Gemini (via langchain-google-genai), ChromaDB, and HuggingFace embeddings. This project demonstrates how to combine document retrieval, LLMs, and chat history for context-aware question answering.

## Features
- **RAG Pipeline**: Retrieve relevant documents and generate answers using Gemini LLM.
- **Chroma Vector Store**: Store and search document embeddings efficiently.
- **HuggingFace Embeddings**: Use `sentence-transformers/all-MiniLM-L6-v2` for document and query embeddings.
- **Chat History**: Maintain and utilize conversation history for more context-aware responses.
- **Jupyter Notebooks**: Interactive exploration and demonstration in `app.ipynb` and `full_rag_pipeline.ipynb`.

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
    .venv\Scripts\activate
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
- Open `app.ipynb` or `full_rag_pipeline.ipynb` in Jupyter or VS Code.
- Run the cells step by step to initialize the RAG pipeline, load documents, and interact with the chatbot.
- Example queries:
  - "What is the capital of Telangana?"
  - "Can we say all gold is glitter?"

## File Structure
- `app.ipynb` — Simple RAG app with conversation history demo.
- `full_rag_pipeline.ipynb` — Full RAG pipeline demonstration.
- `requirements.txt` — Python dependencies.

## Notes
- Ensure your Google Gemini API key is valid and has access to the Gemini model.
- **IMP**: For best results, restart the Jupyter kernel after updating `.env`.
