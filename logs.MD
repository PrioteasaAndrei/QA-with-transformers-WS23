29 December 2023 18:49:32 - Andrei

-built an interface to establish a connection with the database
-successfully posted some pub-med articles in a simple form to check that it works

TODOs: 
-lowercase everything in the dataset X
-check aproaches to STS based on paragraphs or whole paragraph or CLS token or splitting the abstract in chunks using voronoi cells


02 January 2024 13:19:38

PROBLEM: *we need exact match not STS for pubmed* -> do hybrid search

- used the proposed pubmed-bert model for the encodings and pushed all the data to the database
-performed a simple QA query to see if it works
- used a splitting strategy for chunks as seen in the last tutorial

TODOs:
- see Name Entity Recognition approaches
- add entire dataset to huggingface (not just the two columns)
- add metadata to serach fields (e.g search by year)
- create different bodies for hybrid search