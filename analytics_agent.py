# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama.llms import OllamaLLM
from langchain.agents.agent_types import AgentType
import streamlit as st


# page config
st.set_page_config(page_title="Streamlit Analytics Agent", page_icon="💡")


# sidebar
st.sidebar.header("File Upload")

# files supported
file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel
}

# submit button state
def clear_submit():
    """
    clear the submit button state
    """
    st.session_state["submit"] = False

# load data
@st.cache_data(ttl='2h')
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error("Unsupported file format. {ext}")
        return None

# file uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload your data or use the default dataset",
    type=list(file_formats.keys()),
    help="accepts: " + ", ".join(file_formats.keys()),
    on_change=clear_submit)


# set page title
st.write("""
         # Streamlit Analytics Agent
         Analyze your own data using LLMs and automated EDA tools.

         Built with [LangChain](https://python.langchain.com/en/latest/index.html), [Ollama](https://ollama.com/), and [Streamlit](https://streamlit.io/).
         
         ----
         """)

# # data
# df = pd.read_csv("student_habits_performance.csv")

if not uploaded_file:
    st.warning("This app uses LangChain's `PythonAstREPLTool`. Please use caution when sharing this app.")

if uploaded_file:
    df = load_data(uploaded_file)

    # initialize model
    llm = OllamaLLM(model="llama3.2")

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar='💡' if message["role"] == 'assistant' else '🗣️'):
            st.write(message["content"])

    with st.chat_message('assistant', avatar='💡'):
        got_it = st.write("Got it! You uploaded a file with filename `{}`. Ask me anything about the data.".format(uploaded_file.name))
        st.session_state.messages.append({"role": "assistant", "content": got_it})


    # display user message in chat message container
    if prompt:= st.chat_input(placeholder='Write your prompt here...or type "/r" to relaunch.'):

        # strip prompt of any potentially harmful html/js injections
        prompt = prompt.replace("<", "&lt;").replace(">", "&gt;")

        # add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # display message
        with st.chat_message('user', avatar='🗣️'):
            st.write(prompt)

        # set up agent
        agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        allow_dangerous_code=True
        )


        # respond
        with st.spinner("Thinking..."):
            response = agent.invoke(prompt).get("output")
        
            with st.chat_message('assistant', avatar='💡'):
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

