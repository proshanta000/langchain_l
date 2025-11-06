import validators, streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
# Correcting the import path for load_summarize_chain (using the community path)
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader, WebBaseLoader

## ------ Streamlit UI App---

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website.", page_icon="🦜")
st.title("🦜LangChain: Summarize Text From YT or Website.")
st.subheader("Summarize URL")

# --- 1. Get the GROQ API Key and URL ---
with st.sidebar:
    # Key is read as an empty string on first load
    groq_api_key = st.text_input("GROQ API key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

# NOTE: The 'llm' definition must be moved inside the button block!
# Removed: llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

prompt_template ="""
Provide a summary of the follwing content in 300 words.
content:{text}
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)

if st.button("Summarize The Content From YT or Website"):
    ## Validation all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please Provide the information to get started.")
    
    elif not validators.url(generic_url):
        st.error("please enter a valid URL. It can be YT or website url")

    else:
        try:
            with st.spinner("Processing content..."):
                
                # Initialize LLM HERE, where groq_api_key is available ---
                llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

                # 2. Loading the website or YT video data
                if "youtube.com" in generic_url:
                    # NOTE: load() is called directly on the loader for YT
                    loader = YoutubeLoader.from_youtube_url(generic_url,  language="es") #add_video_info=True,
                    docs = loader.load()
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url], 
                        ssl_verify=False, 
                        headers = {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
                            "Accept-Encoding": "gzip, deflate, br",
                            "Accept-Language": "en-US,en;q=0.9",
                            "Connection": "keep-alive"
                        })
                    docs = loader.load()
                    
                # 3. Chain for summarization
                # Using the corrected load_summarize_chain import from earlier
                chain = load_summarize_chain(
                    llm=llm,
                    chain_type="stuff",
                    prompt=prompt
                )

                # 4. Invoke the chain
                output = chain.invoke({"input_documents": docs})
                
                # The output from invoke is a dict for a chain, extract the output_text
                st.success(output['output_text'])
                

        except Exception as e:
            # Handle potential errors during API call or data loading
            st.exception(f"An error occurred: {e}")