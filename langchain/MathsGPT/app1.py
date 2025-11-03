import math
import numexpr # Still imported, but no longer used in the calculator function for max compatibility
import streamlit as st
from pydantic import BaseModel, Field
import uuid 

# Langchain imports
from langchain_groq import ChatGroq
from langchain_core.tools import tool, StructuredTool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver


# ---- 1. Tool Definitions -----

@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression safely. 
    Use this tool for all numerical computations (e.g., 4/3 * 3.14 * 5**3, 100/tan(30)). 
    Note: Trigonometric functions (tan) and the degree conversion function (radians) are available. 
    Use 'tan(radians(degrees))' for correct calculation."""
    
    try:
        # Define allowed functions/constants for security
        # We are using Python's standard math functions
        # EXPANDED: Added sin, cos, sqrt, pow, log, log10, and factorial
        local_dict = {
            "pi": math.pi, 
            "e": math.e, 
            # Trigonometry & Conversion
            "tan": math.tan, 
            "radians": math.radians,
            "sin": math.sin, 
            "cos": math.cos,
            # Exponents & Roots
            "sqrt": math.sqrt, 
            "pow": math.pow,
            # Logarithms
            "log": math.log, 
            "log10": math.log10,
            # Other
            "factorial": math.factorial
        }
        
        # FIX: Switched to Python's built-in eval with a highly restricted scope 
        # (no built-ins) for better compatibility with LLM-generated expressions,
        # while keeping the environment safe.
        result = eval(expression, {"__builtins__": None}, local_dict)
        
        return str(result)
    
    except Exception as e:
        return f"Calculation Error: {e}"
    
# Define Pydantic input schema for structuredTool
class WikipediaInput(BaseModel):
    """Input for the wikipedia tool."""
    query: str = Field(description= "The search query or topic to look up on wikipedia")

# Initializing the wikipedia tool
wikipedia_wrapper= WikipediaAPIWrapper()
wikipedia_tool = StructuredTool(
    name='Wikipedia',
    func=wikipedia_wrapper.run,
    description="A tool for search the internet to find various factual information and data",
    args_schema=WikipediaInput
) 

# The agent itself handles and selecting tools.
ALL_TOOLS= [calculator, wikipedia_tool]

# --- 2. Streamlit UI Setup ---

st.set_page_config(page_title="LangChain: Text To Math Problem Solver And Data Search Assistant", page_icon="🦜")
st.title("🦜 LangChain: Math Solver and Data Search Assistant")

# Key is read as an empty string on first load
groq_api_key = st.sidebar.text_input("GROQ API key", type="password")

if not groq_api_key:
    st.info("Please Enter your GROQ API Key to continue.")
    st.stop()
    
# Initialize LLM Agent Configuratrion
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# ** SYSTEM INSTRUCTIONs**
# FIX: Explicitly instruct the agent to use 'tan()' and 'radians()' without the 'math.' prefix.
SYSTEM_INSTRUCTIONS = "You are a helpful assistant specialized in solving mathematical, factual, and reasoning problems. Use the 'calculator' tool only for numerical computations. When using trigonometric functions, use the available functions 'tan()' and 'radians()' directly, without the 'math.' prefix (e.g., use tan(radians(30))). Use the 'Wikipedia' tool only for factual searches. For all **pure reasoning, logic, and definition-based questions (like 'odd man out' or 'word analogies')**, provide the answer directly by comparing the terms and clearly stating the logical relationship, without using any tool. Always provide a detailed, point-wise explanation for complex questions."

# Optional: Add a checkpointer for state management (memory/history)
memory = MemorySaver()

# Corrected variable name: agent_executor
agent_executor = create_agent(
    model=llm, 
    tools=ALL_TOOLS,
    system_prompt=SYSTEM_INSTRUCTIONS,
    # Adding the checkpointer for persistent state/history
    checkpointer=memory,
)

# --- 3. Streamlit chat Interface ---

# FIX A: Check for the plural key "messages"
if "messages" not in st.session_state:
    st.session_state["messages"] =[
        {"role" : "assistant", "content" :"Hello! I can solve math problems and search for data. How can I help you today?" }
    ]
    # Initialize a unique thread ID for the checkpointer to maintain history
    st.session_state["thread_id"] = str(uuid.uuid4())

# This display loop now safely access the 'role' and  'content'
# assuming all messages in session state are dictionaries.
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Text input area with a default question for easy testing
question = st.text_area(
    "Enter Your Question:",
    value="First, calculate 25**2 - 15**2. Then, what is the capital of Peru?", 
    height=150
)

if st.button("Find my answer"):
    if question:
        with st.spinner("Generating response...."):
            # 1. Store and display user message
            st.session_state.messages.append({"role" : "user", "content" : question})
            st.chat_message("user").write(question)

            # ADDITION: Explicitly label the area where the calculations/thoughts will appear.
            st.subheader("Agent's Thought Process and Tool Use:")
            # Setup callback handler. expand_new_thoughts=True ensures the tool calls and output are visible.
            st_cb= StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)

            # Correct Agent Invoction
            try:
                # Convert list of message dictionaries back into a list of
                # LangChain Message objects (HumanMessage/AIMessage) for agent input.
                langchain_messages = []
                for msg in st.session_state.messages:
                    if msg['role'] == "user":
                        langchain_messages.append(HumanMessage(content=msg['content']))

                    elif msg['role'] == 'assistant':
                        langchain_messages.append(AIMessage(content=msg["content"]))
                
                # Use correct executor name and correct input key ("messages")
                response = agent_executor.invoke(
                    {"messages" : langchain_messages},
                    config={
                        "callbacks":[st_cb],
                        # Pass the required thread_id to the checkpoints
                        "configurable": {"thread_id" : st.session_state["thread_id"]}
                    }
                )

                # Extract Final content
                final_response_content = response["messages"][-1].content

                # 2. Updated session state and display agent response
                # Store the assistant as a dictionary.
                st.session_state.messages.append({"role" : "assistant", "content" : final_response_content})
                st.write("### Final Response:")
                st.success(final_response_content)
            
            
            except Exception as e:
                st.error(f"An error occured during agent execution: {e}")
                st.session_state.messages.append({"role" : "assistant", "content" : f"An error occured: {e}"})
                
    else:
        st.warning("please enter the question.")
