import streamlit as st
from utils import rag_pipeline

model_id = "meta-llama/Llama-2-7b-chat-hf"
index_name = "qa_project_pubmedbert-50"

st.set_page_config(page_title="💬 PubMed ChatBot")
st.title('💬 PubMed ChatBot')

def generate_response(input_text):
  rag = rag_pipeline(model_id, index_name)
  answer = rag(input_text)
  st.info(answer['result'])

with st.form('my_form'):
  text = st.text_area('Ask your question:')
  submitted = st.form_submit_button('Submit')
  if submitted:
    generate_response(text)