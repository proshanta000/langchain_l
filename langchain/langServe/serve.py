from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables (like GROQ_API_KEY) from .env file
load_dotenv()



groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq Chat Model with a stable model name
model = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)


# 1. Create prompt template
system_template = "Translate the following into {language}"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])



# 2. Define Output Parser
parser = StrOutputParser()

# 3. Create LangChain Expression Language (LCEL) Chain
chain = prompt_template | model | parser

# 4. App definition
app = FastAPI(
    title="LangChain Translation Server",
    version="1.0",
    description="A simple API server using LangChain runnable interfaces for translation."
)

# 5. Adding chain routes to the FastAPI app
# This makes the chain available at /chain/ and /chain/playground/
add_routes(
    app,
    chain,
    path="/chain"
)

# 6. Run the Uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)