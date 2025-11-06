# Import libraries
import os
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from dotenv import load_dotenv
load_dotenv()


# ------- Configuration constants ------

# Model name
GROQ_LLAMA = "llama-3.1-8b-instant"
HF_EMBED = "all-MiniLM-L6-v2"


embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED)


# ---- Streamlit UI Setup ----

st.title("Conversation RAG with PDF uploads and chat history")
st.write("Uploads your PDF file")


## Input the groq API key 
api_key = st.text_input("Please Enter Your GROQ API Key:", type="password")

# Check if groq api is provided
if api_key:
    #Initialize the model only if the key is given
    model = ChatGroq(groq_api_key=api_key, model_name=GROQ_LLAMA)


    # Chat Interface
    session_id = st.text_input("Session ID", value="Default_Session")

    if 'store' not in st.session_state:
        st.session_state.store = {}
    

    # Streamlit recomended single file handling way
    uploaded_files = st.file_uploader("Choose Your PDF File", type="pdf", accept_multiple_files=False)


    # Process Uploaded PDF's
    if uploaded_files:
        documents =[]

        temppdf = f"./temp_{uploaded_files.name}"

        try:
            with open(temppdf, "wb") as file:
                file.write(uploaded_files.read())

            # Load the document
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

            # Clean the temprary files immediately after loading
            os.remove(temppdf)

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.stop()


        # --- Indexing (splitting and Embedding) ----
        # split and embedding  for the documents 

        # Initializes the text splitter for breaking down large documents 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Creats chroma vector isttance from the document chunks and embedding
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()


        # --- RAG chain constraction: Step 1 (History-Aware Retriever)

        # Contextualize Question Prompt
        # Prompt instructing the LLM to rewrite a conversational query into a standalone search query
        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question,"
            "Which might reference context in the chat history,"
            "formulated a sentandalon question which can be understood"
            "without the chat history. Do Not answer the question,"
            "Just reformulate it if needed and otherwise return it as is."
        )
        # Combines the system instruction, chat history, placeholder, and user input placeholder
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        
        # Creates the history-aware retriever chain. This chain takes history + question, rewrites the question, 
        # and then uses the rewritten question to fetch relevant documents.

        history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

        # --- RAG Chain Construction: Step 2 (Answer Generation) ---
        # Answer Question Prompt
        # System prompt instructing the LLM on how to generate the final answer using the retrieved context

        system_prompt = (
            "You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer"
            "the question. If you don't know, say that you don't know."
            "Use three sentences maximum and keep the answer concise." # Constraints on the LLM's response style
            "\n\n"
            "{context}" # Placeholder where the retrieved documents will be 'stuffed'
        )

        # Final Prompt template for the LLM to generate the answer
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

        # ---- History Management Function ---
        def get_session_history(session:str) -> BaseChatMessageHistory:

            """Function to retrieve or create the chat history for a session ID."""

            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            
            return st.session_state.store[session]
        
        # --- Conversational Wrapper ---
        # Configures the entire RAG chain to manage conversation history automatically
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # --- Streamlit Chat UI Logic ---
        # Streamlit Chat Input and Display

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display existing messing
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
            
        user_input = st.chat_input("Your Question:")

        if user_input:
            #Display user message
            with st.chat_message("user"):
                st.markdown(user_input)

            # Store user message fro UI redisplay
            st.session_state.messages.append({"role": "user", "content": user_input})
        
            # Invoke the chain 
            with st.spinner("Thinking....."):
                # RUn the full RAG with the user input  and session id
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": session_id}
                    }
                )
            assistant_response = response["answer"]

            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            
            #store assistant message for UI redisplay
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})

else:
    st.warning("Please Enter Your GROQ API Key")




