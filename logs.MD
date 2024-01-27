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