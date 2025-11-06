import math
import numexpr
import streamlit as st
from pydantic import BaseModel, Field
import uuid # Added for generating a unique thread ID

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_core.tools import tool, StructuredTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import create_agent # The LangGraph wrapper utility
# Import AIMessage for converting stored dictionaries back to LangChain objects
from langchain_core.messages import HumanMessage, AIMessage 
from langgraph.checkpoint.memory import MemorySaver

# --- 1. Tool Definitions ---

@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression safely using numexpr. 
    Use this tool for all math-related questions (e.g., 4/3 * 3.14 * 5**3, 25**2 - 15**2)."""
    try:
        # Define allowed functions for security
        local_dict = {"pi": math.pi, "e": math.e}
        result = numexpr.evaluate(
            expression, global_dict={}, local_dict=local_dict
        )
        return str(result)
    except Exception as e:
        return f"Calculation Error: {e}"

# Define Pydantic input schema for StructuredTool
class WikipediaInput(BaseModel):
    """Input for the Wikipedia tool."""
    query: str = Field(description="The search query or topic to look up on Wikipedia.")

# Initializing the Wikipedia tool
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = StructuredTool(
    name='Wikipedia',
    func=wikipedia_wrapper.run,
    description="A tool for search the internet to find various factual information and data.",
    args_schema=WikipediaInput
)

# The agent itself handles  and selecting tools. 
ALL_TOOLS = [calculate, wikipedia_tool] 

# --- 2. Streamlit UI Setup ---

st.set_page_config(page_title="LangChain: Text To Math Problem Solver And Data Search Assistant", page_icon="🦜")
st.title("🦜 LangChain: Math Solver and Data Search Assistant")

# Key is read as an empty string on first load
groq_api_key = st.sidebar.text_input("GROQ API key", type="password")


if not groq_api_key:
    st.info("Please add your GROQ API Key to continue.")
    st.stop()

# Initialize LLM and Agent Configuration
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# **SYSTEM INSTRUCTIONS**
SYSTEM_INSTRUCTIONS = "You are a helpful assistant specialized in solving mathematical, factual, and reasoning problems. Use the 'calculate' tool only for numerical computations and the 'Wikipedia' tool only for factual searches. For all **pure reasoning, logic, and definition-based questions (like 'odd man out' or 'word analogies')**, provide the answer directly by comparing the terms and clearly stating the logical relationship, without using any tool. Always provide a detailed, point-wise explanation for complex questions."


# Optional: Add a checkpointer for state management (memory/history)
memory = MemorySaver()

agent_executor = create_agent(
    model=llm, 
    tools=ALL_TOOLS, # Now only includes external utility tools
    system_prompt=SYSTEM_INSTRUCTIONS,
    # Adding the checkpointer for persistent state/history
    checkpointer=memory,
)

# --- 3. Streamlit Chat Interface ---

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role": "assistant", "content": "Hello! I can solve math problems and search for data. How can I help you today?"}
    ]
    # FIX: Initialize a unique thread ID for the checkpointer to maintain history
    st.session_state["thread_id"] = str(uuid.uuid4())

# FIX: The display loop now safely accesses the 'role' and 'content' 
# assuming all messages in session state are dictionaries.
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Text input area with a default question for easy testing
question = st.text_area(
    "Enter Your Question:", 
    value="First, calculate 25**2 - 15**2. Then, what is the capital of Peru?", 
    height=100
)

if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response...."):
            # 1. Store and display user message
            # Store the user message as a dictionary, not a HumanMessage object.
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            # ADDITION: Explicitly label the area where the calculations/thoughts will appear.
            st.subheader("Agent's Thought Process and Tool Use:")
            # Setup callback handler. expand_new_thoughts=True ensures tool calls and outputs are visible.
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True) 
            
            # Correct Agent Invocation
            try:
                # Convert the list of message dictionaries back into a list of 
                # LangChain Message objects (HumanMessage/AIMessage) for agent input.
                langchain_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))

                response = agent_executor.invoke(
                    {"messages": langchain_messages}, # Pass the converted list of messages
                    config={
                        "callbacks": [st_cb],
                        #  Pass the required thread_id to the checkpointer
                        "configurable": {"thread_id": st.session_state["thread_id"]} 
                    }             
                )

                # Extract Final Content
                final_response_content = response["messages"][-1].content
                
                # 2. Update session state and display agent response
                # Store the assistant response as a dictionary.
                st.session_state.messages.append({"role": "assistant", "content": final_response_content})
                st.write("### Final Response:")
                st.success(final_response_content)
            
            except Exception as e:
                st.error(f"An error occurred during agent execution: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}"})

    else:
        st.warning("Please enter the question.")
    


