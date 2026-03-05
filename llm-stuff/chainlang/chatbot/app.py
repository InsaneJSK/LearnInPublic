from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["groq_api_key"] = os.getenv("groq_api_key")
os.environ["LANGCHAIN_TRACING_V2"]='true'
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question: {question}")
    ]
)

## streamlit functionality

st.title('Langchain Demo with Groq API')
input_text=st.text_input("Search the topic you want")

## OpenAI LLM
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["groq_api_key"],
    model="llama-3.1-8b-instant"
)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
