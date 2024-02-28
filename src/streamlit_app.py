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
import time

st.set_page_config(page_title="💬 PubMed ChatBot")

logging.basicConfig(filename='query_transformation.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")

model_id = "llama2:latest"
index_name = "pubmedbert-sentence-transformer-400"
embedding_model = "NeuML/pubmedbert-base-embeddings"
device = 'cuda:0' 

@st.cache_resource
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name = embedding_model,
        model_kwargs = {'device': device},
        encode_kwargs = {'device': device}
    )
    return embeddings

embeddings = get_embeddings()

@st.cache_resource
def get_vector_search():    
    elastic_vector_search = ElasticsearchStore(
        es_cloud_id = ELASTIC_CLOUD_ID,
        index_name = index_name,
        embedding = embeddings,
        es_api_key = ELASTIC_API_KEY
    )
    return elastic_vector_search

elastic_vector_search = get_vector_search()

st.title('💬 PubMed ChatBot')
st.write("This is a chatbot that can answer questions related to PubMed articles. The default response type is targeted towards users with intermediate to advanced knowledge in the field of biomedicine. The initial buffering may take around 5 minutes.")

@st.cache_resource
def load_ensemble_retriever(index_name,_elastic_vector_search):
    text_splitter = get_splitter_per_index(index_name)
    retriever = create_ensemble_retriever(_elastic_vector_search, text_splitter, neuro_weight=0.5)
    return retriever

## buffer ensemble retriever for consecutive uses
ensemble_retriever = load_ensemble_retriever(index_name,elastic_vector_search)


def _parse(text):
    return text.strip("**")

@st.cache_resource
def get_llm():
    llm = ChatOpenAI(temperature=0,openai_api_key=OPENAI_API_KEY)
    return llm

llm = get_llm()

@st.cache_resource
def get_retrieval_chain():
    rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        verbose=True,
        retriever = ensemble_retriever,
        chain_type_kwargs={
            "verbose": True },
    )
    return rag

rag = get_retrieval_chain()

@st.cache_resource
def get_rewrite_prompt():
    rewrite_prompt = hub.pull("langchain-ai/rewrite")
    return rewrite_prompt

rewrite_prompt = get_rewrite_prompt()

def generate_response(input_text):
    global rag
    ## Rewrite the input text
    start_time = time.time()
    rewrite_llm = llm  
    rewriter = rewrite_prompt | rewrite_llm | StrOutputParser() | _parse
    rewritten_input_text = rewriter.invoke({"x": input_text})   
    logging.info(f"Original input: {input_text}. Transformed input: {rewritten_input_text}")    
    answer = rag(rewritten_input_text)
    end_time = time.time()
    logging.info(f"Time taken to generate response: {end_time - start_time} seconds")
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
