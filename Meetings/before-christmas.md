18 December 2023

# QA-with-transformers-WS23
This is the final project for the Course NLP with Transformers of Univeristy of Heidelberg, Msc. Data and Computer Science, WS23/24

Model: https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract     -----   https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
QA dataset (possible): https://pubmedqa.github.io/

Pretrain for QA.

DB: Opensearch

Questions:
  - Where to run OpenSearch? Asure or AWS free trials and run it there
  - What is the point of paragraphing the document if the average abstract size (in tokens) is 220?
  - Are we allowed to use other data such as metadata, document content etc?

First Task: DB
  1. Find appropiate way to represent documents (\n, paragraphs, find right size for the documents etc.) - trial and error
  2. Do information retrieval based on semantic search that retrieves the right paragraphs at first. (extraction QA)


First Mileston: 11th January
  - need data
  - opensearch working (extractive QA)

Final Deadline: 4th March

**************

Dataset: https://huggingface.co/datasets/prio7777777/pubmed-demo

**************

General Ideas:

In doc structure hold both the vector embedding, but also the title and abstract as text to be analyzed (see Part 2 of the last tutorial).
