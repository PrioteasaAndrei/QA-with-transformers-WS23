import streamlit as st
from utils import *
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import format_document
import logging
import time
from langchain_community.llms import Ollama
from langchain_community.document_transformers.embeddings_redundant_filter import *
from langchain.chains import LLMChain

st.set_page_config(page_title="💬 PubMed ChatBot")

logging.basicConfig(filename='query_transformation.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()  # take environment variables from .env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")

model_id = "openai"
index_name = "pubmedbert-sentence-transformer-100"
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
def get_llm(model_id="openai"):
    if model_id == "openai":
        llm = ChatOpenAI(temperature=0,openai_api_key=OPENAI_API_KEY)
    elif model_id == "llama2":
        llm = Ollama(model=model_id)
    else:
        raise ValueError(f"model_id {model_id} not supported")
    
    return llm

llm = get_llm(model_id)

@st.cache_resource
def get_retrieval_chain(chain_type='unique_docs'):

    if chain_type == "unique_docs":
        # Chain
        qa_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
        return qa_chain

    rag = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        verbose=True,
        retriever = ensemble_retriever,
        chain_type_kwargs={
            "verbose": True },
    )
    return rag

rag = get_retrieval_chain("retrieval qa")

@st.cache_resource
def get_rewrite_prompt():
    rewrite_prompt = hub.pull("langchain-ai/rewrite")
    return rewrite_prompt

rewrite_prompt = get_rewrite_prompt()

@st.cache_resource
def get_reduntant_filter():
    reduntant_filter = EmbeddingsRedundantFilter(embeddings=embeddings, threshold=0.9) ## default is cosine similarity
    return reduntant_filter

reduntant_filter = get_reduntant_filter()

@st.cache_resource
def get_qa_prompt():
    QA_PROMPT = PromptTemplate(
        input_variables=["query", "contexts"],
        template="""You are a helpful assistant who answers user queries using the
        contexts provided. If the question cannot be answered using the information
        provided say "I don't know". Limit your answers to maximum 100 words.

        Contexts:
        {contexts}

        Question: {query}""",
    )
    return QA_PROMPT

QA_PROMPT = get_qa_prompt()

qa_chain = get_retrieval_chain("unique_docs")


def generate_response(input_text):
    global rag
    ## Rewrite the input text
    start_time = time.time()
    rewrite_llm = llm  
    rewriter = rewrite_prompt | rewrite_llm | StrOutputParser() | _parse
    rewritten_input_text = rewriter.invoke({"x": input_text})   
    logging.info(f"Original input: {input_text}. Transformed input: {rewritten_input_text}")    
    # answer = rag(rewritten_input_text)

    unfiltered_docs = ensemble_retriever.invoke(rewritten_input_text,30)
    unformatted_docs = reduntant_filter.transform_documents(unfiltered_docs)
    docs = [x.to_document() for x in unformatted_docs]

    
    out = qa_chain(
        inputs={
            "query": rewritten_input_text,
            "contexts": "\n---\n".join([d.page_content for d in docs])
        }
    )
    answer = out["text"]

    end_time = time.time()
    logging.info(f"Time taken to generate response: {end_time - start_time} seconds")
    # return answer['result']
    return answer

def generate_response_with_sources(input_text):
    # rag = rag_pipeline(model_id, index_name,use_openai=True,retriever_type='ensemble')
    llm = Ollama(model=model_id)
    
    ## Rewrite the input text
    rewrite_prompt = hub.pull("langchain-ai/rewrite")
    rewrite_llm = ChatOpenAI(temperature = 0, openai_api_key = OPENAI_API_KEY)  
    rewriter = rewrite_prompt | rewrite_llm | StrOutputParser() | _parse
    rewritten_input_text = rewriter.invoke({"x": input_text})   
    logging.info(f"Original input: {input_text}. Transformed input: {rewritten_input_text}")


    ANSWER_PROMPT = ChatPromptTemplate.from_template(
        """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be as verbose and educational in your response as possible. 
        Each passage has a SOURCE which is the title of the document. When answering, cite source name of the passages you are answering from below the answer, on a new line, with a prefix of "SOURCE:".


        context: {context}
        Question: "{question}"
        Answer:
        """
    )

    DOCUMENT_PROMPT = PromptTemplate.from_template(
        """
        ---
        SOURCE: {title}
        {page_content}
        ---
        """
    )
    def _combine_documents(docs, document_prompt=DOCUMENT_PROMPT, document_separator="\n\n"):
        doc_strings = [format_document(doc, document_prompt) for doc in docs]
        return document_separator.join(doc_strings)

    retriever = elastic_vector_search.as_retriever(search_kwargs={"k":3})

    _context = {
        "context": retriever | _combine_documents,
        "question": RunnablePassthrough(),
    }

    chain = _context | ANSWER_PROMPT | llm | StrOutputParser()

    answer = chain.invoke(rewritten_input_text)
    return answer


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # msg = generate_response(prompt)
    msg = generate_response_with_sources(prompt)
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
