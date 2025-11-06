import os
import streamlit as st # Streamlit is used to create the web interface for the application
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma # Chroma is the vector store (database) used to efficiently search document embeddings
# Corrected import from ChatMessage to ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory # Used to store and manage the history of the conversation
from langchain_core.chat_history import BaseChatMessageHistory # Base class for chat history management
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Used to define the structure and content of the prompts sent to the LLM, including where history and context go
from langchain_groq import ChatGroq # Groq's integration for using their fast LLM models (e.g., Llama 3)
from langchain_core.runnables.history import RunnableWithMessageHistory # Wraps the chain to automatically manage and pass the chat history between turns
from langchain_huggingface import HuggingFaceEmbeddings # Used to convert text (documents and query) into numerical vectors (embeddings) for similarity search
from langchain_text_splitters import RecursiveCharacterTextSplitter # Used to break large documents into smaller, manageable chunks
from langchain_community.document_loaders import PyPDFLoader # Used to load text content from PDF files

from dotenv import load_dotenv # Used to load environment variables (like API keys) from a .env file
load_dotenv()

# --- Configuration Constants ---
# Models name
GROQ_LLAMA = "llama-3.1-8b-instant" # The specific Groq LLM model to be used for generation and query rewriting
HF_EMBED = "all-MiniLM-L6-v2" # The specific HuggingFace model used to create vector embeddings

# Ensure the environment variable is set for HuggingFace (if needed, though it's typically for model access)
# If HuggingFaceEmbeddings doesn't need this token for public models, this line can be removed.
# os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED) # Initialize the embedding model

# --- Streamlit UI Setup ---
# Set up streamlit
st.title("Conversation RAG with PDF uploads and chat history") # Sets the title of the Streamlit application
st.write("Upload PDF's and chat with their content") # Description for the user

## Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password") # Secure input for the required API key

# Check if groq api is provided
if api_key:
    # Initialize the model only if the key is provided
    model = ChatGroq(groq_api_key=api_key, model_name=GROQ_LLAMA) # Initializes the Groq LLM

    # Chat Interface
    session_id = st.text_input("Session ID", value="Default_Session") # Allows user to define a session ID for history tracking

    # statefully manage chat history
    if 'store' not in st.session_state:
        st.session_state.store = {} # Dictionary to hold chat histories, keyed by session ID

    # Streamlit recommends handling single file uploads this way
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=False) # File upload widget

    # Process uploaded PDF's
    if uploaded_file:
        documents = []
        # Since we use accept_multiple_files=False, uploaded_file is a single object
        
        temppdf = f"./temp_{uploaded_file.name}" # Creates a temporary local file path to save the uploaded PDF
        
        try:
            # FIX: Use .read() instead of .getvalues() to get the file contents as bytes
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.read()) # Writes the uploaded file content (bytes) to the temporary file
            
            # Load the document
            loader = PyPDFLoader(temppdf) # Initializes the PDF loader using the temporary file path
            docs = loader.load() # Loads the text content from the PDF
            documents.extend(docs) # Adds the loaded documents to the list
            
            # Clean up the temporary file immediately after loading
            os.remove(temppdf) # Deletes the temporary file to keep the environment clean
        except Exception as e:
            st.error(f"Error processing file: {e}") # Display error if file loading fails
            st.stop() # Stop execution if file processing fails

        # --- Indexing (Splitting and Embedding) ---
        # split and creat embeddings for the documents
        # Initializes the text splitter for breaking down large documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500) 
        splits = text_splitter.split_documents(documents) # Splits the loaded documents into smaller chunks
        # Creates a Chroma vector store instance from the document chunks and embeddings
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings) 
        retriever = vectorstore.as_retriever() # Creates a retriever object, which is responsible for searching the vector store

        # --- RAG Chain Construction: Step 1 (History-Aware Retriever) ---
        # Contextualize Question Prompt
        # Prompt instructing the LLM to rewrite a conversational query into a standalone search query
        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question, "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is." # Instructs the LLM on its role
        )
        # Combines the system instruction, chat history placeholder, and user input placeholder
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"), # Placeholder for the previous conversation history
                ("human", "{input}"), # Placeholder for the current user question
            ]
        )

        # Creates the history-aware retriever chain. This chain takes history + question, rewrites the question, 
        # and then uses the rewritten question to fetch relevant documents.
        history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

        # --- RAG Chain Construction: Step 2 (Answer Generation) ---
        # Answer Question Prompt
        # System prompt instructing the LLM on how to generate the final answer using the retrieved context
        system_prompt =(
            "You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer"
            "the question. If you don't know, say that you don't know."
            "Use three sentences maximum and keep the answer concise." # Constraints on the LLM's response style
            "\n\n"
            "{context}" # Placeholder where the retrieved documents will be 'stuffed'
        )

        # Final prompt template for the LLM to generate the answer
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"), # Includes history for conversational continuity in the answer
                ("human", "{input}"), # Includes the original user question
            ]
        )

        # Creates a chain that takes documents and 'stuffs' them into the qa_prompt for the LLM to generate an answer
        question_answer_chain = create_stuff_documents_chain(model, qa_prompt) 
        # Combines the two main parts: history-aware retrieval and question answering
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # --- History Management Function ---
        def get_session_history(session: str) -> BaseChatMessageHistory:
            """Function to retrieve or create the chat history for a session ID."""
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory() # Create new history object if session is new
            return st.session_state.store[session]
        
        # --- Conversational Wrapper ---
        # Configures the entire RAG chain to manage conversation history automatically
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input", # Specifies the key for the user's current input
            # FIXED: Standardized history_messages_key to "chat_history" to match the prompts
            history_messages_key="chat_history", # Specifies the key for the conversation history
            output_messages_key="answer" # Specifies the key for the final generated answer
        )

        # --- Streamlit Chat UI Logic ---
        # Streamlit Chat Input and Display
        if "messages" not in st.session_state:
            st.session_state.messages = [] # Initialize a list to hold messages for UI display

        # Display existing messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Your question:") # Chat input box
        
        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Store user message for UI redisplay
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Invoke the chain
            with st.spinner("Thinking..."):
                # Runs the full RAG chain with the user input and session ID
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": session_id} # Passes the session ID to the history manager function
                    }
                )

            assistant_response = response["answer"] # Extracts the final answer from the chain output

            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            
            # Store assistant message for UI redisplay
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
else:
    st.warning("Please enter your GROQ API key") # Warning displayed until API key is entered