import os
import pandas as pd
from tqdm import tqdm
import csv
from dotenv import load_dotenv
from datasets import load_dataset
from datasets import Dataset, Sequence, Value
from elasticsearch import Elasticsearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import ElasticsearchStore
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import SpacyTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain.chains import RetrievalQA
from transformers import BitsAndBytesConfig
from torch import cuda, bfloat16
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers
from langchain.llms import HuggingFacePipeline
import torch
from langchain_openai import ChatOpenAI
import torch
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from importlib import reload

from langchain_community.llms import Ollama

def prepare_llm(auth_token, model_id = "llama2:latest", use_openai=False, **kwargs):
    '''
    Initializes an LLM either through Ollama or through OpenAI:

    PARAMETERS:
    model_id - name of Ollama model

    RETURNS:
    llm
    '''
    OPENAI_API_KEY = kwargs.get('OPENAI_API_KEY', None)

    if use_openai: 
        llm = ChatOpenAI(temperature=0,openai_api_key=OPENAI_API_KEY)
        return llm


    bitsAndBites_config = BitsAndBytesConfig(load_in_4bit = True, 
                                            bnb_4bit_compute_dtype = bfloat16, 
                                            bnb_4bit_use_double_quant = True)

    model_config = AutoConfig.from_pretrained(model_id, use_auth_token = auth_token)

    ## TODO: add back bits and bites
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # quantization_config = bitsAndBites_config,
        trust_remote_code = True,
        torch_dtype=torch.bfloat16, 
        config = model_config,
        device_map = 'auto',
        token = auth_token,
        # attn_implementation="flash_attention_2",
    )

    ##supported natively by llama
    # model = model.to_bettertransformer()

    model.eval()# we only use the model for inference
    print(f"Model loaded ")

    
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token = auth_token)

    generate_text = transformers.pipeline(
        model = model, 
        tokenizer = tokenizer,
        return_full_text = True,
        task = 'text-generation',
        temperature = 0.01,
        max_new_tokens = 200,  # max number of tokens to generate in the output
        repetition_penalty = 1.1,  # without this output begins repeating,
        # device= 'cuda:0'
    )

    llm = HuggingFacePipeline(pipeline = generate_text)

    return llm

def _parse(text):
    return text.strip("**")


def rag_pipeline(model_id: str, 
                 index_name: str,
                 use_openai: bool = False,):
    '''
    Initializes a RAG pipeline.

    PARAMETERS:
    model_id - name of generative model
    index_name - name of document index on Elastic Cloud 

    RETURNS:
    Chain for question-answering against an index.
    '''
    
    load_dotenv()
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
    ELASTIC_CLOUD_ID = os.getenv('ELASTIC_CLOUD_ID')
    ELASTIC_API_KEY = os.getenv('ELASTIC_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    embedding_model = "NeuML/pubmedbert-base-embeddings"
    device = 'cuda:0' 

    embeddings = HuggingFaceEmbeddings(
        model_name = embedding_model,
        model_kwargs = {'device': device},
        encode_kwargs = {'device': device}
    )

    elastic_vector_search = ElasticsearchStore(
        es_cloud_id = ELASTIC_CLOUD_ID,
        index_name = index_name,
        embedding = embeddings,
        es_api_key = ELASTIC_API_KEY
    )
    ## TODO: replace with ensemble retriever
    retriever = elastic_vector_search.as_retriever(search_kwargs={"k":3})

    llm = prepare_llm(HUGGINGFACE_TOKEN,model_id,use_openai,OPENAI_API_KEY=OPENAI_API_KEY)
    
    print("Preparing RAG pipeline...")
    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        verbose=True,
        retriever = retriever,
        chain_type_kwargs={
            "verbose": True },
    )
    print("RAG pipeline ready.")
    return rag_pipeline

def run_config(elastic_vector_search : ElasticsearchStore,
               use_ensemble_retriever: bool,
               verbose: bool = True,
               config_name: str = 'default_config',
               save: bool = False,
               **kwargs):
    '''
    Runs a configuration of the RAG pipeline and returns the RAGAS evaluation metrics.
    Assume that the index is already created and the documents are already indexed.
    params:
    elastic_vector_search: ElasticsearchStore
    use_ensemble_retriever: bool
    verbose: bool
    config_name: str
    **kwargs:
        index_name: str
        evaluation_dataset_path: str
        HUGGINGFACE_TOKEN: str
        HUGGINGFACE_DATASET_NAME: str
        text_splitter: TextSplitter
        llm: HuggingFacePipeline
        max_retrieved_docs: int
    '''
    index_name = kwargs.get('index_name', None)
    evaluation_dataset_path = kwargs.get('evaluation_dataset_path', None)
    HUGGINGFACE_TOKEN = kwargs.get('HUGGINGFACE_TOKEN', None)
    HUGGINGFACE_DATASET_NAME = kwargs.get('HUGGINGFACE_DATASET_NAME', None)
    QA_VALIDATION_TOKEN = kwargs.get('QA_VALIDATION_TOKEN', None)
    text_splitter = kwargs.get('text_splitter', None)
    llm = kwargs.get('llm', None)
    save_path = kwargs.get('save_path', None)
    max_retrieved_docs = kwargs.get('max_retrieved_docs', 3)
    query_transformation_strategy = kwargs.get('query_transformation_strategy', 'default')
    OPENAI_API_KEY = kwargs.get('OPENAI_API_KEY', None)

    rewrite_prompt = hub.pull("langchain-ai/rewrite")
    rewrite_llm = ChatOpenAI(temperature=0,openai_api_key=OPENAI_API_KEY)

    rewriter = rewrite_prompt | rewrite_llm | StrOutputParser() | _parse
    

    if index_name is None or evaluation_dataset_path is None or HUGGINGFACE_TOKEN is None or HUGGINGFACE_DATASET_NAME is None or llm is None:
        raise ValueError("Missing parameters")
    
    if not elastic_vector_search.client.indices.exists(index=index_name):
        raise ValueError(f"Index {index_name} does not exist")

    if verbose:
        print(elastic_vector_search.client.info())

    retriever = elastic_vector_search

    ## define RAG pipeline

    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        verbose=verbose,
        retriever = elastic_vector_search.as_retriever(search_kwargs={"k":max_retrieved_docs}), ##TODO: increase later to see influence
        chain_type_kwargs={
            "verbose": verbose },

    )

    if verbose:
        print("Successfully loaded RAG pipeline")

    ## build ensemble retriever if needed

    if use_ensemble_retriever:
        ## seems like the only way is to hold the documents in memory: https://github.com/langchain-ai/langchain/discussions/10619

        if verbose:
            print("Building ensemble retriever")

        ## load and split the data
        loader = HuggingFaceDatasetLoader(HUGGINGFACE_DATASET_NAME,use_auth_token=HUGGINGFACE_TOKEN,page_content_column='abstract')
        data = loader.load()

        split_data = text_splitter.split_documents(data)

        bm25_retriever = BM25Retriever.from_documents(split_data)
        neuro_retriever = retriever.as_retriever()

        bm25_retriever_weight = 0.5

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, neuro_retriever], weights=[bm25_retriever_weight, 1 - bm25_retriever_weight]
        )
        retriever = ensemble_retriever

    ## Load the evaluation dataset
    eval_dataset = load_dataset(evaluation_dataset_path,token=QA_VALIDATION_TOKEN)['train']

    if verbose:
        print("Successfully loaded evaluation dataset")

    ## iterate with tqdm over dataset

    answers = []
    
    ## TODO: vecotrize this
    for example in tqdm(eval_dataset,desc="generate RAG answers"):
        query = example['question']
        ragas_answer_gpt = example['ground_truth'][0]
        ragas_context = example['ground_truth_context'][0]
        
        top_k = 5

        # returned_docs = call_similartiy(retriever,query,top_k)

        ## rewrite query
        if query_transformation_strategy == "read-write-retrieve":
            query = rewriter.invoke({"x": query})

        answer = rag_pipeline(query)
        answers.append(answer)

    
    if save:
        field_names = ['query','result']

        # Write the list of dictionaries to a CSV file
        with open(save_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)

            # Write the header
            writer.writeheader()

            # Write the data
            writer.writerows(answers)

    ## list[{'query': str,'result':str}]
    return answers

def call_similartiy(retriever,query, top_k=5,is_ensemble=False):
    '''
    Call the similarity method of the retriever
    '''
    if is_ensemble:
        return retriever.invoke(query,top_k)
    else:
        return retriever.get_relevant_documents(query,top_k)

def testset_to_validation(save=False,**kwargs):

    QA_VALIDATION_DATASET = kwargs.get('QA_VALIDATION_DATASET', None)
    QA_VALIDATION_TOKEN = kwargs.get('QA_VALIDATION_TOKEN', None)
    save_path_answers = kwargs.get('save_path', None)
    save_path_result = kwargs.get('save_path_result', None)

    eval_dataset = load_dataset(QA_VALIDATION_DATASET,token=QA_VALIDATION_TOKEN)['train']

    df = pd.read_csv(save_path_answers)

    ## join them on the query vs question

    result_df = pd.merge(df, eval_dataset.to_pandas(), left_on='query', right_on='question', how='inner')

    result_df = result_df.drop(columns=['query','question_type','episode_done'])
    ## first parse the ground_truth and ground_truth context by \n
    columns_mapping = {'question': 'question', 'result': 'answer', 'ground_truth': 'ground_truths', 'ground_truth_context':'contexts'}
    result_df = result_df.rename(columns=columns_mapping)
    
    result_df['contexts'] = result_df['contexts'].apply(lambda x: [x])
    result_df['ground_truths'] = result_df['ground_truths'].apply(lambda x: [x])

    if save:
        result_df.to_csv(save_path_result,index=False)

    result_df_dataset = Dataset.from_pandas(result_df)
        
    return result_df_dataset



def index_docs(data, text_splitter, index_name, **kwargs):
        
    ELASTIC_CLOUD_ID = kwargs.get('ELASTIC_CLOUD_ID', None)
    ELASTIC_API_KEY = kwargs.get('ELASTIC_API_KEY', None)
    embeddings = kwargs.get('embeddings', None)

    split_data = text_splitter.split_documents(data)

    if ELASTIC_CLOUD_ID and ELASTIC_API_KEY:
        db = ElasticsearchStore.from_documents(
            split_data,
            embeddings,
            es_cloud_id=ELASTIC_CLOUD_ID,
            index_name=index_name,
            es_api_key=ELASTIC_API_KEY,
            distance_strategy="COSINE",
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(
                hybrid=True,
            )

        )

    db.client.indices.refresh(index=index_name)

    return db, split_data

def create_ensemble_retriever(db,split_data):
    '''
    use invoke method for the ensemble retriever
    '''
    bm25_retriever = BM25Retriever.from_documents(split_data)
    neuro_retriever = db.as_retriever()

    bm25_retriever_weight = 0.5

    ensemble_retriever = EnsembleRetriever(
          retrievers=[bm25_retriever, neuro_retriever], weights=[bm25_retriever_weight, 1 - bm25_retriever_weight]
    )

    return ensemble_retriever