import streamlit as st
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import ElasticsearchStore
import transformers
from transformers import LlamaTokenizer
import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
HUGGINGFACE_USERNAME = os.getenv('HUGGINGFACE_USERNAME')
HUGGINGFACE_DATASET_NAME = os.getenv('HUGGINGFACE_DATASET_NAME')

ELASTIC_CLOUD_ID = os.getenv('ELASTIC_CLOUD_ID')
ELASTIC_API_KEY = os.getenv('ELASTIC_API_KEY')


@st.cache_resource
def get_retriever():
    index_name = 'qa_project_pubmedbert-50' # previously index="test_pubmed_split"
    elastic_vector_search = ElasticsearchStore(
        es_cloud_id = ELASTIC_CLOUD_ID,
        index_name = index_name,
        embedding = embeddings,
        es_api_key = ELASTIC_API_KEY,
    )
    return elastic_vector_search.as_retriever(search_kwargs={"k":3})


@st.cache_resource
def create_chain():
    model_id = 'meta-llama/Llama-2-7b-chat-hf'
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=os.environ.get('HUGGINGFACE_TOKEN')
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        device_map='auto',
        token=os.environ.get('HUGGINGFACE_TOKEN')
    )
    model.eval()# we only use the model for inference

    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", use_auth_token=HUGGINGFACE_TOKEN)
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        temperature=0.01,
        max_new_tokens=512,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )

    llm = HuggingFacePipeline(pipeline=generate_text)

    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        verbose=True,
        retriever = get_retriever(),
        chain_type_kwargs={
            "verbose": True 
        },
    )
    return rag_pipeline


st.set_page_config(
    page_title="Pubmed Chatbox"
)

st.header("Pubmed Chatbox!")


