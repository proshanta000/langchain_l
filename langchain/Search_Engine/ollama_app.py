import os
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

from dotenv import load_dotenv

load_dotenv()

# --- Tool Setup ---
# Arxiv Tools and wikipedia tools
# Increased doc_content_chars_max to give the model more context for better answers
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Named the DuckDuckGo tool "Search" as it appears in the error message
search = DuckDuckGoSearchRun(name="Search")

# --- Streamlit UI Setup ---
st.title("🗨️Chat with 🔎 Search")
"""
In this example, we're using "StreamlitcallbackHandler" to display the thoughts and actions of the agent in an interactive Streamlit app.
Try more LangChain Streamlit Agent at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# Model name
OLLAMA_MODEL = "llama3.2:1b"

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you? "}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="what is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Re-initialize model and tools inside the prompt block
    model = ChatOllama(model=OLLAMA_MODEL, streaming=True)
    tools = [search, arxiv, wiki]

    # FIX 1: Corrected the typo in the parameter name
    # Changed 'handling_parsing_error' to 'handle_parsing_errors'
    # This enables the agent to recover if the LLM outputs the wrong format.
    search_agent = initialize_agent(
        tools,
        model,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # FIX 2: Corrected the input to the agent
        # Agents expect a single string query, not the list of messages.
        # We pass the most recent user 'prompt' string instead of st.session_state.messages.
        response = search_agent.run(prompt, callbacks=[st_cb])
        
        st.session_state.messages.append({'role': "assistant", "content": response})
        st.write(response)
