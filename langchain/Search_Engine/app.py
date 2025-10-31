import os
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

from dotenv import load_dotenv

load_dotenv()

# Arxiv Tools and wikipedia tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)


search = DuckDuckGoSearchRun(name="Search")


st.title("🗨️Chat with 🔎 Search")
"""
In this example, we're using "StreamlitcallbackHandler" to display the throughts and actions of agent in an interactive Streamlit app.
Try more LangChain Streamlit Agint at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# Sidebar for settings
st.sidebar.title("Settings")
api_key= st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role": "assistant", "content":"Hi, I'm a chatbot who can search the web. How can I help you? "}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="what is machine learning?"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    model= ChatGroq(groq_api_key=api_key, model="llama-3.1-8b-instant", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent= initialize_agent(tools, model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_error=True)


    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response= search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({'role':"assistant", "content":response})
        st.write(response)
