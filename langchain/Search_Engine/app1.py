# Import necessary libraries
import os
import streamlit as st # Streamlit is used for creating the web application UI
from langchain_groq import ChatGroq # LangChain integration for the Groq chat models
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper # Wrappers for accessing Arxiv and Wikipedia APIs
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults # LangChain tools for agent to use
from langchain.agents import initialize_agent, AgentType # Functions for creating and defining the LangChain agent
from langchain.callbacks import StreamlitCallbackHandler # A callback handler to stream agent's thoughts and actions to Streamlit UI


from dotenv import load_dotenv
load_dotenv() # Load environment variables (like API keys) from a .env file

# --- Tool Setup ---

# Initialize the Arxiv API wrapper with specific settings
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
# Create the Arxiv tool that the agent can execute
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Initialize the Wikipedia API wrapper with specific settings
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
# Create the Wikipedia tool
wiki= WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Create the DuckDuckGo search tool
search = DuckDuckGoSearchResults(name="Search")

# --- Streamlit UI Setup ---

# Set the title of the Streamlit application
st.title("🗨️Chat with 🔎 Search")

# Display a brief description and link in the main body
"""
In this example, we're using "StreamlitcallbackHandler" to display the thoughts and actions of agent in an interactive Streamlit app.
Try more LangChain Streamlit Agent at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# Sidebar for settings
st.sidebar.title("Settings")

# Input field in the sidebar for the user to enter their GROQ API Key
api_key = st.sidebar.text_input("Please Enter your GROQ API Key:", type="password")

# --- Chat Interface Logic ---

# Initialize chat history in Streamlit's session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role" : "assistant", "content" : "Hi, I'm a chatbot who can search on web. How may I help you?"}
    ]

# Display all previous messages in the chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input from the chat input box
# 'prompt:=' assigns the input value to the 'prompt' variable and evaluates the condition
if prompt:=st.chat_input(placeholder="what is Machine learning?"):
    # Append the user's prompt to the chat history
    st.session_state.messages.append({"role" : "user", "content" : prompt})
    # Display the user's message in the chat
    st.chat_message("user").write(prompt)

    # --- Agent Initialization on User Input ---

    # Instantiate the ChatGroq model with the provided API key and model name, enabling streaming
    model= ChatGroq(groq_api_key=api_key, model="llama-3.1-8b-instant", streaming=True)
    # Define the list of tools the agent can use
    tools = [search, arxiv, wiki]

    # Initialize the LangChain agent
    search_agent = initialize_agent(
        tools, # The tools the agent has access to
        model, # The language model (LLM) to drive the agent's decisions
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # The type of agent (ReAct style)
        handling_parsing_error=True # A setting to handle potential parsing issues in the LLM's output
    )

    # Display the assistant's response within a chat message container
    with st.chat_message("assistant"):
        # Initialize the StreamlitCallbackHandler to stream the agent's process (thoughts, actions)
        # It uses the current container and does not expand new thoughts automatically
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # Run the agent with the chat history and the Streamlit callback
        # NOTE: Passing all messages here, but the agent will primarily focus on the latest prompt.
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        
        # Append the assistant's final response to the chat history
        # NOTE: There is a typo here in the original code ('assistgant') which you should correct if possible.
        st.session_state.messages.append({"role" : "assistant", "content" : response})
        
        # Write the final response to the Streamlit UI
        st.write(response)


