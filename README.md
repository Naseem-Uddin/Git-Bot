# 🤖 Git Bot – GitHub Repository Assistant

Git Bot is a powerful assistant designed to help developers dive into unfamiliar codebases faster. By leveraging Retrieval-Augmented Generation (RAG), the app allows users to chat directly with any public GitHub repository to understand its logic, structure, and functionality.

## 📋 Table of Contents

### [Features](https://github.com/Naseem-Uddin/rag_codebase?tab=readme-ov-file#-features)

### [How It Works](https://github.com/Naseem-Uddin/rag_codebase?tab=readme-ov-file#-how-it-works)

### [Tech Stack](https://github.com/Naseem-Uddin/rag_codebase?tab=readme-ov-file#%EF%B8%8F-tech-stack)

### [Getting Started](https://github.com/Naseem-Uddin/rag_codebase?tab=readme-ov-file#-getting-started)

### [Usage](https://github.com/Naseem-Uddin/rag_codebase/tree/main?tab=readme-ov-file#-usage)

## ✨ Features

URL Ingestion: Simply paste a public GitHub URL to begin analysis.

Automated Pipeline: Automatic cloning, parsing, and embedding generation.

Natural Language Querying: Ask questions like "Where is the authentication logic?" or "How are API errors handled?"

Context-Aware Answers: Responses are grounded in the specific code snippets retrieved from the repository.

## 🔄 How It Works

Clone: The app clones the target repository to a local temporary directory.

Process: It filters for supported source code files and splits them into manageable chunks.

Embed: Using Hugging Face models, it converts code chunks into vector embeddings.

Store: Vectors are upserted into a Pinecone index for high-speed similarity searches.

Retrieve & Generate: When a user queries, the system finds the most relevant code and passes it to the Groq-hosted LLM to generate an answer.

## 🛠️ Tech Stack

- Interface: Streamlit
- Embeddings: Hugging Face Sentence Transformers
- Vector Database: Pinecone
- LLM Inference: Groq API
- Framework: LangChain

## 🚀 Getting Started

**Prerequisites**

- Python 3.9+
- A Pinecone API Key
- A Groq API Key

**Installation**

1. Clone this project:

`git clone [https://github.com/your-username/git-bot.git](https://github.com/your-username/git-bot.git)
cd git-bot`


2. Install dependencies:

`pip install -r requirements.txt`


3. Create a .env file in the root directory:

`PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key`


4. Run the App:

`streamlit run app.py`


## 📖 Usage

Enter the URL of any public GitHub repository.

Wait for the status indicator to show that the repository has been indexed.

Start asking questions in the chat interface!

This project was built to explore practical developer tools powered by LLMs and to make large codebases easier to navigate and understand.
