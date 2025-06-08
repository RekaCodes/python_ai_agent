# import libraries
import pandas as pd
import numpy as np

from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama.llms import OllamaLLM
from langchain.agents.agent_types import AgentType
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st


# set page title
st.title("Student Habits and Performance Analysis")
st.write("Analyze student habits and performance using natural language queries.")


 # load data
df = pd.read_csv("student_habits_performance.csv")


# initialize model and agent
model = OllamaLLM(model="llama3.2")
agent = create_pandas_dataframe_agent(
    model,
    df,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    allow_dangerous_code=True
)

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar='💡' if message["role"] == 'assistant' else '🗣️'):
        st.markdown(message["content"])


# accept user input
prompt = st.chat_input(placeholder='Write your prompt here...or type "/r" to relaunch.')


# display user message in chat message container
if prompt:
    # strip prompt of any potentially harmful html/js injections
    prompt = prompt.replace("<", "&lt;").replace(">", "&gt;")
    # display message
    with st.chat_message('user', avatar='🗣️'):
        st.markdown(prompt)
    # add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


# respond
response = agent.invoke(prompt)
with st.chat_message('assistant', avatar='💡'):
    st.markdown(response)
st.session_state.messages.append({"role": "assistant", "content": response})









# # initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # display chat messages from history
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # accept user input
# if prompt := st.chat_input("Ask your question about student habits and performance:"):
#     # display user message in chat message container
#     st.chat_message("user").markdown(prompt)
#     # add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})


# # stream response