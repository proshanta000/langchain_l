import os
import streamlit as st
from dotenv import load_dotenv

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()



os.environ["LANGCHAIN_API_KEY"]  = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

## Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked"),
        ("user", "Question: {question}")
    ]
)

## streamlit freamework
st.title("Langchain Demo With LLAMA3.2")
input_text= st.text_input("What question you have in mind?")


## Ollama Llama2 model
llm = OllamaLLM(model="llama3.2")

output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))