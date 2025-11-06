#Import libraries
import streamlit as st
from pathlib import Path
from sqlalchemy import create_engine
import sqlite3

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Langchain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 Langchain: Chat with SQL DB")

# --- UI sidebar setup ---
LOCALDB="USE_LOCALDB"
MYSQL = "USE_MYSQL"


radio_opt = ["use sqlite3 Database- student.db", "Connect to you SQL Database"]

selected_db = st.sidebar.radio(label="Choose DB which you want to chat", options=radio_opt)

# Intialize variables to avoid errors if radio_opt[1] isn't selected
mysql_host, mysql_user, mysql_password, mysql_db = None, None, None, None

if radio_opt.index(selected_db) == 1:
    db_uri = MYSQL
    st.sidebar.markdown('---')
    st.sidebar.markdown("### MySQL Connection Details")

    # --- UPDATED DEPENDENCY AND CONNECTION CHECK NOTE ---
    st.sidebar.warning(
        "**Dependency Check:** For MySQL, ensure 'mysql-connector-python' is installed: "
        "`pip install mysql-connector-python`."
    )
    st.sidebar.info(
        "**Connection Check:** If you see 'Can't connect to MySQL server', ensure the server is running "
        "and accessible from where this app is hosted (e.g., your firewall allows external connections if needed)."
    )
    # ----------------------------------------------------

    mysql_host = st.sidebar.text_input("Provide MySQL Host (e.g., localhostm IP).")
    mysql_user = st.sidebar.text_input("MySQL User.")
    mysql_password= st.sidebar.text_input("MySQL Password.", type="password")
    mysql_db = st.sidebar.text_input("MySQL database name")

else:
    db_uri= LOCALDB

# api_key = st.sidebar.text_input(label="GROQ API Key", type="password")

# --- Validation and Execution Stop (Crucial for Streamlit flow) ---

# 1. stop if connection details are incomplte
if db_uri == MYSQL and not (mysql_host and mysql_user and mysql_password and mysql_db):
    st.info("Please provide all MySQL connection details.")
    st.stop()

# 2. Stop if API key is missing (Prevents the previous GroqError)
#if not api_key:
    #st.info("Please add the **GROQ API key** in the sidebar to proceed.")
    #st.stop()

# --- Database Configuration (Only runs if necessary inputs are provided) ---

@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        # Note: 'student.db' must exist in the same directory as app.py
        dbfilepath = (Path(__file__).parent / "student.db").absolute()

        # Usingr ead-only mode for safty
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)

        # Using sqlite:/// requires a creator function to point to the file
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    
    elif db_uri == MYSQL:
        # This line requires the 'mysql-connector-python' package to be installed
        # The connection string format is: mysql+mysqlconnector://user:password@host/database
        return SQLDatabase(create_engine(
            f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
        ))
    
    return None # Should not happen if logic is correct

try:
    if db_uri == MYSQL:
        db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
    else:
        db = configure_db(db_uri)
    
    if not db:
        st.error("Failed to configure database.")
        st.stop

    # --- LLM and Agent Initialization (Only runs if key is present) ---
    llm = ChatOllama(model="llama3.2:1b", streaming=True)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    # Increasing power by upgrading the model and switching to the more stable,
    # text-based reasoning agent (zero-shot-react-description) which works better with Groq.
    sql_agent_executor = create_sql_agent(
        llm= llm,
        toolkit=toolkit,
        verbose=True,
        # agent_type="openai-tools", # Removed due to Groq API 'Failed to call a function' error
        handle_parsing_errors=True

    )

except Exception as e:
     # This block now catches the DatabaseError (2003) and provides context
    st.error(f"Error during DB or Agent initialization: {e}")
    if "Can't connect to MySQL server" in str(e):
        st.error(
            "**Action Required (MySQL Connection):** The database connection failed. "
            "Please verify that the MySQL server is running, the host and port are correct, "
            "and that there is no firewall blocking port 3306."
        )
    st.stop()

# --- Chat History and Input ---

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm ready to query your sql database. what question do you have?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask anything from the database.")

if  user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        # The StreamlitCallbackHandler allows LangChain to stream output to the container
        st_cb = StreamlitCallbackHandler(st.container())
        try:
            # We use invoke/stream in modern LangChain, but run() is used here for simplicity
            # in older AgentExecutor chains.
            response= sql_agent_executor.run(user_query, callbacks=[st_cb])

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

        except Exception as e:
            error_message = f"An unexpected error occured during agent excuteion: {e}"

            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I ran into an error: {e}"})









