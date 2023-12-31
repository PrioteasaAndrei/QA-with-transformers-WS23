29 December 2023 18:49:32 - Andrei

-built an interface to establish a connection with the database
-successfully posted some pub-med articles in a simple form to check that it works

TODOs: 
-create vector embeddings and update the index function
-lowercase everything in the dataset
-check aproaches to STS based on paragraphs or whole paragraph or CLS token or splitting the abstract in chunks using voronoi cells
-remove \n and replace them with space (may have already been done in the parsing of the dataset)