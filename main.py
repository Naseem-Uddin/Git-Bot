import streamlit as st
import numpy as np
import random
import time
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, Index
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import tempfile
from github import Github, Repository
from git import Repo
from pathlib import Path
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore # type: ignore

def clone_repository(repo_url):
    if not repo_url:  # Add validation for empty URL
        raise ValueError("Please provide a GitHub repository URL")
        
    repo_name = repo_url.split("/")[-1]  # Extract repository name from URL
    temp_dir = tempfile.mkdtemp()
    repo_path = os.path.join(temp_dir, repo_name)
    Repo.clone_from(repo_url, str(repo_path))
    return str(repo_path)

# Initialize session state for GitHub URL if not already present
if "github_url" not in st.session_state:
    st.session_state["github_url"] = ""

# Create input field for GitHub URL
github_url = st.text_input(
    "Enter GitHub Repository URL",
    value=st.session_state["github_url"],
    placeholder="https://github.com/username/repository",
    help="Enter the full HTTPS URL of the GitHub repository you want to analyze"
)

# Update session state when URL changes
if github_url != st.session_state["github_url"]:
    st.session_state["github_url"] = github_url

# Clone repository and save path if URL is provided
path = None
if github_url:
    path = clone_repository(github_url)

SUPPORTED_EXTENSIONS = {'.py', '.js', '.tsx', '.jsx', '.ipynb', '.java',
                         '.cpp', '.ts', '.go', '.rs', '.vue', '.swift', '.c', '.h'}

IGNORED_DIRS = {'node_modules', 'venv', 'env', 'dist', 'build', '.git',
                '__pycache__', '.next', '.vscode', 'vendor'}

def get_file_content(file_path, repo_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Get relative path from repo root
        rel_path = os.path.relpath(file_path, repo_path)

        return {
            "name": rel_path,
            "content": content
        }
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def get_main_files_content(repo_path: str):
    files_content = []

    try:
        for root, _, files in os.walk(repo_path):
            # Skip if current directory is in ignored directories
            if any(ignored_dir in root for ignored_dir in IGNORED_DIRS):
                continue

            # Process each file in current directory
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[1] in SUPPORTED_EXTENSIONS:
                    file_content = get_file_content(file_path, repo_path)
                    if file_content:
                        files_content.append(file_content)

    except Exception as e:
        print(f"Error reading repository: {str(e)}")

    return files_content

file_content = get_main_files_content(path)

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

# Set the PINECONE_API_KEY as an environment variable
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
os.environ['PINECONE_API_KEY'] = pinecone_api_key

# Initialize Pinecone
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"],)

# Connect to your Pinecone index
pinecone_index = pc.Index("codebase-rag")

vectorstore = PineconeVectorStore(index_name="codebase-rag", embedding=HuggingFaceEmbeddings())

documents = []

for file in file_content:
    doc = Document(
        page_content=f"{file['name']}\n{file['content']}",
        metadata={"source": file['name']}
    )

    documents.append(doc)


vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=HuggingFaceEmbeddings(),
    index_name="codebase-rag",
    namespace= github_url
)

def perform_rag(query):
    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(vector=raw_query_embedding.tolist(), top_k=3, include_metadata=True, namespace= github_url)

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    # Modify the prompt below as need to improve the response quality
    system_prompt = f"""You are a Senior Software Engineer with 30 years of experience, specializing in TypeScript, Python, and JavaScript. 
    You constantly refer back to the codebase and the context surrounding a question to provide accurate and helpful responses.
    Keep in mind that the user may not understand the highest level concepts, so provide detailed explanations for all of his questions/queries.
    Always consider all of the context provided when forming a response.
    """

    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return llm_response.choices[0].message.content








st.title("Git Bot")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"])

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "llama-3.1-70b-versatile"

#Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything about the codebase..."):
    # display user message on site
    with st.chat_message("user"):
        st.markdown(prompt)

    # add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # display bot message on site
    with st.chat_message("assistant"):
        # Use perform_rag instead of direct LLM call
        response = perform_rag(prompt)
        st.write(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})