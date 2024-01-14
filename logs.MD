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
- build top 5 recall function for evaluation dataset
- try to fix the other chunking mechanism
-hybrid search