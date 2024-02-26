29 December 2023 18:49:32 - Andrei

-built an interface to establish a connection with the database
-successfully posted some pub-med articles in a simple form to check that it works

TODOs: 


02 January 2024 13:19:38

PROBLEM: *we need exact match not STS for pubmed* -> do hybrid search

- used the proposed pubmed-bert model for the encodings and pushed all the data to the database
-performed a simple QA query to see if it works
- used a splitting strategy for chunks as seen in the last tutorial

TODOs:
- create different bodies for hybrid search


13 January 2024 12:50:04

- added metadata to the huggingface dataset
- parsed published date from journal
- added metadata to index
- switch to biomedical bert
-updated the model to https://huggingface.co/NeuML/pubmedbert-base-embeddings


TODOs:
-check this out for valdiation: https://huggingface.co/datasets/pubmed_qa
- try to fix the other chunking mechanism
-hybrid search


14 January 2024 18:20:47


forgot to write stuff here

16 January 2024 21:46:07

TODOS:
- generate answer to store in index using gpt 3.5 see last tutorial


17 January 2024 14:45:35


https://opensearch.org/blog/semantic-science-benchmarks/
https://opensearch.org/docs/latest/query-dsl/compound/hybrid

18 January 2024 13:26:09

Starting the migration to LangChain
Considering switching to ElasticSearch as LangChain has better support for that


28 January 2024 17:28:59

-began migration to elastich search and langchain, deployed on a free trial cloud
-implemented a mock up hybrid search using an ensemble retriever

TODO:

- figure out how to create a custom index with many vector fields (e.g for generated questions, title) and how to use an ensemble retriever for all of them and combine results
- how to do elastic search query from langchain
- Hypothetical Questions with llama 2 and put it in a new index, then combine results (PS llama 2 7B takes too much for inference)


31 January 2024 13:42:18

Todays plan:
- finish the rag architecture in its simplest form
- do some evaluation at the end like in the last 

02 February 2024 15:58:03

-same error locally seems to be related to the embedding but I literally have no idea whats going on.

03 February 2024 18:25:42

Error was fixed. It was due to some weird API call.

Problems with IR evaluation:
- no way to know which are the relevant documents for one query, other than manual inspection
- retrieved document may be correct but paragraph may be shifted to the left or right

For now lets just evaluate that the IR returns the relevant documents, regardless of the correct paragraph.

-also need to do query augumentation

What I did today:
- i was thinking we can use the results retrieved directly from pubmed using their api as a baseline of which documents are relavant or not for a given query, but it seems that pubmed uses only some keyword searches and not vector searches. In that sense our returned document are more relevant as they use synonims and context etc. Spent about 1.5 h on this

-investigated different evaluation mechanisms for when labeled data is not present. Just ended up with the conclusion that there just arent any. Still need to investigate more but no result has been found.

-created a new index with 50 token chunks into elastic_search

- together with @Mara completed the RAG and debugged something that took around 3h total

06 February 2024 22:49:34

-switching initial data retrieval system using the langchain API
-using RAGAs for the validation and synthethic test data generation


07 February 2024 22:25:12

Approach: using small chunking size (50) and feed directly to generator or use big chunking size (400) -> summary -> generator
For this we need a complet system in order to report the metrics

07 February 2024 09:46

Here is a list of things we could try out:

- Inference time is slow, but we could optimize it through the following methods:
    - Use FlashAttention-2 for the model generating the answer. Not all models support it, llama does.
    - Use BetterTransformer.
    - Use a quantization library like bitsandbytes.
    Check https://huggingface.co/docs/transformers/perf_infer_gpu_one#bitsandbytes.

- Add reference to the source documents either through the prompt for the generative model (see https://github.com/langchain-ai/langchain/blob/master/cookbook/qa_citations.ipynb) or by matching parts of generated response to the original text chuncks.

- Try several generative models (mostly open source):
    - meta-llama/Llama-2-13b-chat-hf
    - meta-llama/Llama-2-7b-chat-hf
    - mistralai/Mistral-7B-v0.1 (Mistral-7B-v0.1 is Mistral AI’s first Large Language Model (LLM))
    - mistralai/Mixtral-8x7B-v0.1 (Mistral AI’s second Large Language Model (LLM))
    - Rocketknight1/falcon-rw-1b 
    - gpt-3.5-turbo
    - gpt-4 (no longer open source, but probably covered by subscription)

- Query transformation

- Play with different chunck sizes and text splitters.

07 February 2024 22:25:12

Approach: using small chunking size (50) and feed directly to generator or use big chunking size (400) -> summary -> generator
For this we need a complet system in order to report the metrics


10 February 2024 16:03:30

Added bits and bytes and to_bettertransformer to optimize inference time. Inference time is now about 30s / query. Run the evaluation queries through the RAG with the 400 chunking index on the new dataset and computed the RAGAs metrics. Defined a pipeline for running a configuration and getting the ragas metrics.

Evaluation dataset is available in QA_VALIDATION_DATASET using QA_VALIDATION_TOKEN. See evaluation_IR to see how to apply metric from RAGas on it.

15 February 2024 22:45:35

TODOS:
-run different configs
    -for IR (chunking, top k, embedding, retriever (base vs ensemble) )
    -for generation (summary, model used, prompt)
-query transformation (research https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/
    https://blog.langchain.dev/query-transformations/
    ) / context enrichment
    

25 February 2024 20:41:07

probably wont have time and resources to run different configs, maybe at least for query transformation
https://python.langchain.com/docs/templates/

106 mins still not done indexing the 400 dataset on cpu

TODO: 
-implement read-write-retrieve query transformation usint OpenAI chat in utils.py


26 February 2024 16:50:28
read-write-retrieve query transformation.
Inference time is really slow: 10 mins per inference with 2B params
results are aweful


Tried a mock-up for openAI gpt 3.5 chain and seems to work fine. Still need to evaluate.

evaluation_dataset_openai.json - manually extracted pairs of question/answer + generated answer from gpt 3.5 chain

TODO: 
-replace generation model with GPT 3.5 turbo
-manually check evaluation dataset made initially by Mara to see gpt 3.5. answers. It halucinates a lot and is not very explicit
(note that some are good and the original paragraphs do not correspond with what the retreiver retrieves).