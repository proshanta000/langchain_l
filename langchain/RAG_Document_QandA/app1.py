## Importing libraries
import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

## loading the env data
load_dotenv()

# Loading the Groq Api key
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv('GROQ_API_KEY')

# Models name
LLAMA= "llama-3.1-8b-instant"
NOMIC_EMBED = "nomic-embed-text"



##   <----- Step 1 ---------->

# Loading model
model = ChatGroq(model=LLAMA, groq_api_key=groq_api_key)

# Prompt template
rag_prompt= ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question:{input}
    """
)



##   <----- Step 2 ---------->


# creating a function for vector embedding
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embedding = OllamaEmbeddings(model=NOMIC_EMBED) # Embeddings model
        st.session_state.loader = PyPDFDirectoryLoader("Research_paper") # Data ingestion
        st.session_state.docs = st.session_state.loader.load() # Documents loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # Text splitter 
        st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50]) # Final Documents For the vectors
        st.session_state.vectors = FAISS.from_documents(st.session_state.documents, st.session_state.embedding) # Vectorrization


##   <----- Step 3 ---------->


# Create a button for Documents embeddings
button = st.button("Documents Embedding")
if button:
    create_vector_embedding()
    st.write("vector database is ready.")

# Input box
user_prompt = st.text_input("Enter your query from the research paper.")



##   <----- Step 4 ---------->

# Finalize the app
if user_prompt:
    document_chain = create_stuff_documents_chain(model, rag_prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    
    start = time.process_time()
    response =retriever_chain.invoke({'input': user_prompt})
    print(f"Response Time: {time.process_time()-start}")

    st.write(response['answer'])

    # With a streamlit expander
    with  st.expander("Document similarity search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write('===========================')



