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
- do some evaluation at the end like in the last assigment