import streamlit as st
from utils import *
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import logging

logging.basicConfig(filename='query_transformation.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()  # take environment variables from .env.

model_id = "llama2:latest"
index_name = "pubmedbert-sentence-transformer-400"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="💬 PubMed ChatBot")
st.title('💬 PubMed ChatBot')



def _parse(text):
    return text.strip("**")


def generate_response(input_text):
  ## TODO: replace this with the other template style with | from openai
  rag = rag_pipeline(model_id, index_name,use_openai=True)
  
  ## Rewrite the input text
  rewrite_prompt = hub.pull("langchain-ai/rewrite")
  rewrite_llm = ChatOpenAI(temperature = 0, openai_api_key = OPENAI_API_KEY)

  rewriter = rewrite_prompt | rewrite_llm | StrOutputParser() | _parse
  rewritten_input_text = rewriter.invoke({"x": input_text})

  logging.info(f"Original input: {input_text}. Transformed input: {rewritten_input_text}")

  answer = rag(rewritten_input_text)
  return answer['result']

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    msg = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

# OLD streamlit code
# with st.form('my_form'):
#   text = st.text_area('Ask your question:')
#   submitted = st.form_submit_button('Submit')
#   if submitted:
#     generate_response(text)