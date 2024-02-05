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

