first make validation to be able to compare models

no need for overcomplicating with replacing named entities from a knowledge base. RAG should do the expanding by default because it does closed book anwering so it has some info in its params

create handcrafted set of {question: paragraph with answer from article}. May also use chat-gpt: feed abstract to chat gpt and generate question from it etc.

use perplexity for generated answer to check that they are choerent
use recall or top-5 recall to measure the quality of the end-to-end system

all experiments should be written down in a form or another (maybe a big word at the end) to keep track of the progress. we get points for that

for simple questions (e.g. :what is the author of this paper ) do a simple comparison to evaluate the results

other data to index:
- whole text + whole text embedding
-chunks + chunks embedding
- metadata
- questions generated with rag / gpt for that specific chunk  / document