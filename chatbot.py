import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.langchain import LangchainLLM
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pandasai.prompts.base import AbstractPrompt as BasePrompt  # fixed
from jinja2 import Template
import os

load_dotenv()

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")

class CustomPrompt(BasePrompt):
    template: str = ""  # required by AbstractPrompt

    def __init__(self, **kwargs):
        self.props = kwargs
        self._resolved_prompt = None
        self._jinja_template = Template("""
        You are a Python data analysis expert.
        Analyze the DataFrame to answer the following question:
        Question: {{ question }}
        DataFrame Columns: {{ columns }}
        Respond with only the Python code wrapped in triple backticks. No explanation.
        """)

    def to_string(self) -> str:
        self._resolved_prompt = self._jinja_template.render(**self.props)
        return self._resolved_prompt
def chat_with_csv(df, query):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    langchain_llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.2,
        stream=False,
    )

    llm = LangchainLLM(langchain_llm)

    custom_prompt = CustomPrompt(
        question=query,
        columns=list(df.columns)
    )

    pandas_ai = SmartDataframe(df, config={
        "llm": llm,
        "custom_prompt": custom_prompt,
    })

    return pandas_ai.chat(query)

# === Streamlit UI ===
st.set_page_config(layout='wide')
st.title("Assistant Chatbot : ")

input_csvs = st.sidebar.file_uploader(
    "Upload your CSV files",
    type=['csv'],
    accept_multiple_files=True
)

if input_csvs:
    selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
    selected_index = [file.name for file in input_csvs].index(selected_file)

    data = pd.read_csv(input_csvs[selected_index])
    st.success("CSV uploaded successfully")
    st.dataframe(data.head(3), use_container_width=True)

    input_text = st.text_area("Enter your query")
    if input_text and st.button("Chat with csv"):
        st.info("Your Query: " + input_text)
        try:
            result = chat_with_csv(data, input_text)
            st.success(result)
        except Exception as e:
            st.error(f"Error: {e}")
