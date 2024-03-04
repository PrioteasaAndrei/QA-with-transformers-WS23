# Pubmed QA System

A user-friendly QA system designed to answer questions on medical data concerning intelligence from Pubmed abstracts.  

**Team Members**
- Cristi Andrei Prioteasa (cristi.prioteasa@stud.uni-heidelberg.de) - 4740844, Master Data and Computer Science, Heidelberg University.
- Mara-Eliana Popescu (mara-eliana.popescu@stud.uni-heidelberg.de) - 4166979, Master of Data and Computer Science, Heidelberg University.
- Alper Dağgez (ro312@stud.uni-heidelberg.de) - 4732226, Master of Data and Computer Science, Heidelberg University.

**Previous Team Member:** Tarik Mistura Arcoverde Cavalcanti

**Advisor:**
John Ziegler (ziegler@informatik.uni-heidelberg.de)

**Anti-plagiarism Confirmation:** We declare that what has been written in this work has been written by the members of Group 7 and that, with the exception of quotations, no part has been copied from scientific publications, the Internet or from research works already presented in the academic field.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ul>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#related-work">Related Work</a></li>
    <li><a href="#methods">Methods</a></li>
    <li><a href="#experimental-setup-and-results">Experimental Setup and Results</a></li>
    <li><a href="#conclusions-and-future-work">Conclusions and Future Work</a></li>
    <li><a href="#contribution-table">Contribution table</a></li>
    <li><a href="#references">References</a></li>
  </ul>
</details>

## Important Note
**The current configuration uses the free trial of Elastic Cloud for hosting the indexes. The trial expires on the 12th of March, so setting up a new cloud is possible but we need a heads up of about 12hrs to reupload the indexes to the new cloud.**

## Getting Started
Make sure you have [Anaconda](https://www.anaconda.com/download) installed.
Use the following commands to create the proper environment, activate it and start the app:

```
conda env create -f environment.yml
conda activate nlp
streamlit run src/streamlit_app.py
```
You can pass the following flags to the `streamlit run` command:
- `model_id` selects the generative model. The value can be `openai` or `llama2`.
- `index_name` selects the index on Elastic Cloud. The following values can be passed:
  - `pubmedbert-sentence-transformer-100`
  - `pubmedbert-sentence-transformer-200`
  - `pubmedbert-sentence-transformer-400`
  - `pubmedbert-recursive-character-400-overlap-50`
- `sourcing` tells the system if sources should be included (then set it to `True`) or not (set it to `False`).
- `use_ensemble` tells the system if the ensemble_retriever should be used (then set it to `True`) or not (set it to `False`).

Default command from above is equivalent to:
```
streamlit run .\src\streamlit_app.py --model_id openai --index_name pubmedbert-sentence-transformer-100  --sourcing False --use_ensemble False
```

Make sure port 8501 is not used by another process. \
Local URL: http://localhost:8501 \
Network URL: http://147.142.152.35:8501

**Note**: the [Ensemble Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble) takes about 5 minutes to load some of the data in memory. This is because, by default, the langchain BM25 retriever can only hold the documents in memory and writing them to disk for retrieval is not trivial.\
After posing a question expect an inference time of 1-5 minutes.

## Introduction
With the recent advancements of Large Language Models (LLMs) interest in domain-specific applications powered by LLMs has increased. In particular, language models can assist experts in the medical domain most commonly by answering questions against a knowlegde base.

In this project, we implement a question answering system that takes in any queries regarding Pubmed abstracts related to the keyword "intelligence" from 2013 until 2023 and generates an answer in natural language.
Our system is based on a Retrieval Augmented Generation (RAG) pipeline, combining text retrieval from a vector database with the generative capabilities of a state-of-the-art LLM.
Furthermore, we evaluate the performance of our system by using a custom designed dataset.

The [Methods](#methods) section gives an extensive account of our approach covering especially data acquisition and preprocessing, embedding and storing of the corresponding vectors, document retrieval from a given query, answer generation and much more.

## Related Work

In application domains like biomedicine, past work has shown that using in-domain text can provide additional gains over general-domain language models. In particular, [Gu, Tinn, Cheng, et al., (2021)](https://arxiv.org/pdf/2007.15779.pdf) have shown that domain-specific pretraining serves as a solid foundation for a wide range of biomedical
NLP tasks, leading to state-of-the-art results. Their [BiomedBERT](https://microsoft.github.io/BLURB/models.html) models were pretrained exclusively on medical data from Pubmed and currently score high in the [Biomedical Language Understanding and Reasoning Benchmark](https://microsoft.github.io/BLURB/). The BiomedBERT base models are not suitable for complex QA tasks and on data published after the models were last trained. 
However, [PubMedBERT Embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings), finetuned on BiomedBERT to produce high quality embeddings, has drawn our attention as being particularly suitable as a foundation for our QA system.

The relatively small number of documents in our dataset and the absence of a proper format for the question answering task indicated that a QA system based on a Retrieval Augmented Generation pipeline would be most effective. Furthermore, previous work has proven that incorporating external knowledge (i.e., non-parametric knowledge) with internal knowledge (i.e., parametric knowledge) can alleviate hallucination.
To the best of our knowledge, no QA system based on Retrieval Augmented Generation (RAG), makes use of the PubMedBERT Embeddings in the text retrieval component to generate answers to queries on Pubmed data. 

Many existing RAG based systems use a retrieve-then-read strategy ([Lewis et al., (2020)](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html), [Karpukhin et al., (2020)](https://aclanthology.org/2020.emnlp-main.550/)), where relevant documents are first selected and then passed on to a large language model as external contexts to generate an answer. The significant limitation of such systems is the inevitable "gap between the input text and the knowledge that is really needed to query". To solve this issue, we adopted the rewrite-retrieve-read strategy introduced by [Ma et al., (2023)](https://arxiv.org/pdf/2305.14283.pdf?ref=blog.langchain.dev) . This new module uses and additional rewrite step in which the initial query is transformed before the retrieval and generation step. 

Evaluating RAG architectures is challenging because there are several aspects to
consider such as the ability of the retrieval system to identify relevant and focused context passages or the quality of the generation. At the same time, evaluating the correctness of generated answers is difficult in specialized fields and in the absence of ground-truth answers. To evaluate and ensure the quality of our QA system, we use the RAGAS evaluation framework, introduced by [Es et al., (2023)](https://arxiv.org/pdf/2309.15217.pdf), which leverages LLMs to create "ground-truth" answers and evaluates several components of the entire system.

## Methods
Our pipeline consists of the following stages:
<ol>
  <li><a href="#data-acquisition">Data Acquisition</a></li>
  <li><a href="#chunking-and-vectorization">Chunking and Vectorization</a></li>
  <li><a href="#information-retrieval">Information Retrieval</a></li>
  <li><a href="#query-transformation">Query Transformation</a></li>
  <li><a href="#response-generation">Response Generation</a></li>
  <li><a href="#reference-sources">Reference Sources</a></li>
  <li><a href="#evaluation-methods">Evaluation Methods</a></li>
  <li><a href="#user-interface">User Interface</a></li>
</ol>

### Data Acquisition
As already mentioned in the introduction, the data we gathered consists of abstracts of Pubmed articles concerning intelligence, published between 2013 and 2023 [[Link](https://pubmed.ncbi.nlm.nih.gov/?term=intelligence+%5BTitle%2Fabstract%5D&filter=simsearch1.fha&filter=years.2014-2024&sort=date)].

We used the [Bio.Entrez](https://biopython.org/docs/1.75/api/Bio.Entrez.html) Python package to access the Pubmed database and scrape the targeted abstracts together with relevant metadata consisting of the title of the article, the date of publication and the list of authors.

The main challenge encountered at this stage was the fact that the underlying public NCBI Entrez API limits the number of retrieved articles to 10000. If you follow the link above you can see that over 60000 articles match our criteria. To overcome this limitation we used multiple API calls, each of them retrieving articles from different periods of time. We ensured that the prescribed periods of time do not yield more than 10000 articles. We exhaust the whole range from 2013 to 2023 without having overlapping between the different API calls. Overall, we managed to gather 69698 abstracts with the corresponding metadata.

We stored the data as a private HuggingFace dataset with the following data fields:
- `title`,
- `abstract`,
- `publication_date`: in the format DD/MM/YYYY,
- `id`: unique identifier for each article,
- `authors`: list of names.

The file [download_pubmed.py](src/download_pubmed.py) contains the whole pipeline for scraping the data and creating a local dataset in JSON format with the same data fields as our HuggingFace dataset.

### Chunking and Vectorization

![Image](images/embedding-model.png)

To be able to perform both lexical and semantic search on the gathered data, we chunked each abstract, embedded each chunk and stored all embeddings in an index in Elasticsearch hosted on Elastic Cloud.

Our system uses the [PubMedBERT Embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings) to vectorize the text chunks. This embedding model is particularly suitable to our task for various reasons:
- The model, based on [MSR BiomedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) already pretrained on full Pubmed articles, was further finetuned using Pubmed title-abstract pairs.
- The embedding space is a 768 dimensional dense vector space, having a generous embedding dimensionality.

We experimented with a couple of chunking strategies employing the Langchain Text Splitters. Besides selecting the text splitter, we tried out different chunk sizes to find a good fit. The embedding model expects as input a sequence no longer than 512 tokens, thus limiting the chunk size. The following table shows all the combinations we used:

<!-- Use ✅ or  ❌ -->
| Text Splitter\Chunk Size | 50 | 100 | 200 | 400 |
|--------------------------|----|-----|-----|-----|
| [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter) |❌ | ❌ | ❌ | ✅ |
| [SentenceTransformersTokenTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/split_by_token) | ✅ | ✅ | ✅ | ✅ |

For easy access to the vector database, irrespective of the machine on which our QA system is run we created an Elasticsearch instance on Elastic Cloud. Other reasons for choosing Elasticsearch were its compatibility with Langchain and the costs for maintaining the vector database.

Each index has the same underlying retrieval strategy, namely a [Hybrid ApproxRetrievalStrategy](https://python.langchain.com/docs/integrations/vectorstores/elasticsearch#approxretrievalstrategy), which uses a combination of approximate semantic search and keyword based search.
 
### Information Retrieval

The information retrieval component of our system is based on an [Ensemble Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble) combining the sparse retriever [BM25](https://python.langchain.com/docs/integrations/retrievers/bm25) with the hybrid retriever of our Elasticsearch index. The results of both retrievers are reranked using [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).
Since the hybrid retrieval strategy of our Elasticsearch index already combines lexical and semantic search, the resulting ensemble retriever emphasizes the lexical search, which is a preferred strategy in the medical field where certain keywords cannot be semantically approximated through other words.

Given an arbitrary query, our system retrieves the first $k=30$ documents that would be relevant for answer generation and passes them on to the generative component. 
We still need to experiment with the hyperparameter $k$.

One peculiarity we noticed while evaluating this component of our system is that duplicates appear among the retrieved documents. This behaviour becomes worse upon increasing $k$. To solve this problem, we used an [EmbeddingsRedundantFilter](https://api.python.langchain.com/en/latest/document_transformers/langchain_community.document_transformers.embeddings_redundant_filter.EmbeddingsRedundantFilter.html), which identifies similar documents and filters out redundant content.

### Query Transformation
The quality of the retrieved documents is also highly influenced by the quality of the query. Two problems that come up regarding the queries are that user questions may be poorly worded for retrieval or that the user questions may contain misleading or irrelevant information.

To address these problems, we adopted the [Rewrite-Retrieve-Read](https://arxiv.org/pdf/2305.14283.pdf?ref=blog.langchain.dev) approach to transform the user's query. To put it in a nutshell, our system first prompts a LLM to rewrite the query and then uses the rewritten query to retrieve relevant documents.

### Response Generation

Currently, our system relies on OpenAI's model `gpt-3.5-turbo` to generate the answer using the rewritten query and the retrieved documents. In our experiments we also employed Meta's `llama2` model with 7B parameters and 4-bit quantization, provided through [Ollama](https://ollama.com/). In the end we settled for the former model to be able to host our QA system on [Streamlit Community Cloud](https://streamlit.io/cloud). If the user wants to test our system in a local environment, it is possible to switch from `gpt-3.5-turbo` to `llama2`. The two LLM's are comparable in terms of performance.

We faced certain problems at this stage of our project like resource limitations on our local machines or on Google Colab, nonavailability of certain packages or softwares on Windows systems (bitsandbytes, until recently Ollama as well), slow inference times both locally and on Google Colab before we started working with Ollama and OpenAI's API.

### Reference Sources 

Since the response generation relies on multiple documents, we deemed it necessary that our system accurately back-references the used sources. To this end we extended the prompt template to ask the generative LLM to cite the retrieved documents using the stored metadata. More specifically, each answer will contain a list of sources similar to what we see in following example:

![Image](images/source_tracing.png)

### Evaluation Methods

We used [Ragas](https://docs.ragas.io/en/stable/) to evaluate our entire QA system. First we created a synthetic test dataset with Ragas containing 67 rows with the following data fields:
- `question` (generated with the default LLM of Ragas)
- `context` (containing the documents returned by our retriever)
- `answer` (the ground-truth answer generated with the default LLM of Ragas)
- `question_type`
- `episode_done`

Next, we run all 67 questions from the test dataset through our system, add the generated answers under a new data field to the test dataset and evaluate the following metrics on the resulting dataset:

- `context precision` measures the signal-to-noise ratio of the retrieved context ([definition](https://docs.ragas.io/en/stable/concepts/metrics/context_precision.html)).
- `context recall` measures if all the relevant information required to answer the question was retrieved ([definition](https://docs.ragas.io/en/stable/concepts/metrics/context_recall.html)).
- `answer relevancy` measures how relevant the generated answer is to the question ([definition](https://docs.ragas.io/en/stable/concepts/metrics/answer_relevance.html)).

More on the evaluation results and the exact configurations we evaluated is described in the section [Experimental Setup and Results](#experimental-setup-and-results).

### User Interface

To make our QA system more user-friendly we added a user interface made with [streamlit](https://streamlit.io/). One major advantage of using streamlit is the [caching machanism](https://docs.streamlit.io/library/advanced-features/caching), which makes the system run much faster and helps with persisting objects across reruns.
By using caching of the embedding model, the vectorstore, the document retriever and the generative model, we were able to decrease the answer time significantly.

## Experimental Setup and Results


RAGAs validation metrics for Sentence Transformer 400 chunking, no ensemble retriever, read-write-retrieve query transformation, Chat GPT 3.5. Turbo for both the rewritter and generation part, pubmedbert embeddings.

We obtained the best results (on our synthetic validation dataset) using the following configuration: 

| **Model**                    | GPT 3.5 Turbo                   |
|------------------------------|--------------------------------|
| **Chunking size**                 | 400 (50 overlap)                |
| **Splitter**                 | Sentence Transformer|
| **Ensemble Retriever**       | None |
| **Query Transformation**    | Read-Write-Retrieve            |
| **Rewriter and Generation**  | GPT 3.5 Turbo (Both Parts)              |
| **Embeddings**               | PubMedBERT                     |

Our results are as follows:

| **Metric**            | **Value**                    |
|------------------------|------------------------------|
| **Answer Relevancy**   | 0.89           |
| **Context Precision**  | 0.92           |
| **Context Recall**     | 0.92           |
| **Faithfulness**       | 0.82           |

For a description of the used metrics see [RAGAS](https://docs.ragas.io/en/latest/concepts/metrics/index.html).

When running the previous configuration with a different index (**Chunking size 100, overlap 10**) we obtain the following results:


| **Metric**            | **Value**                    |
|------------------------|------------------------------|
| **Answer Relevancy**   | 0.85           |
| **Context Precision**  | 0.92           |
| **Context Recall**     | 0.92           |

Evaluation on different chunking mechanisms:

| Index (spliiter-chunk size)                                  | Answer Relevancy | Context Precision | Context Recall |
|-------------------------------------------------|------------------|---------------------|-----------------|
| pubmedbert-sentence-transformer-50              | 0.87             | 0.93                | 0.93            |
| pubmedbert-sentence-transformer-400             | 0.88             | 0.93                | 0.93            |
| pubmedbert-recursive-character-400-overlap-50   | 0.91             | 0.93                | 0.93            |


### Ensemble retriever influence
We analyze the influence of the weight of the ensemble retriever (weight of BM25 retriever) in terms of Context Precision, which is defined as the signal to noise ratio of retrieved context. As we are using PubMed data and we want precise answers we will prioritize the Context Precision over Context Recall.

| Context Precision     | BM25 Retriever Weight   |
|------------------------|-------------------------|
| 0.8800                 | 0                       |
| 0.9200                 | 0.3                     |
| 0.9200                 | 0.5                     |
| 0.7899                 | 0.7                     |
| 0.7564                 | 1                       |

### Inference time

| Hardware Specifications          | Inference Time  |
|-----------------------------------|------------------|
| GPU: RTX 3050 Ti (4 GB VRAM)      | 3 mins 53 seconds|
| RAM: 16 GB                         |                  |
| CPU: Intel Core i7 (10th Gen)     |                  |
| Max Retrieved Docs: 50             |                  |
| Used Tokens per query: 8295                  |                  |


## Conclusions and Future Work

We conclude the following aspects about our work:
- we have sucessfully built an end-to-end chat-bot using a RAG architecture on the pubmed articles which contain the keyword `intelligence` and answers questions with a focus on precision rather than recall (targeted behaviour for sensitive content such as medical articles) 
- we implemented a query transformation mechanism (Rewrite-Retrieve-Read) and investigated different approaches such as Multi Querying and Self Querying (not implemented in the final version)
- we implemented target sourcing, where statements are cited with the article from which they are extracted / generated
- we implemented a graphical interface using `streamlit` for easy interaction with the system
- we have created an evaluation method for our pipeline, using `RAGAS` to generate a synthetic dataset using `GPT 3.5 Turbo` as a critic and generator. We include in our synthetic dataset different subtypes of questions such as simple, reasoning, multi-context and conditional.
- using our synthethic dataset we investigated the influence of different hyperparameters as follows:

        - the influence of using an ensemble retriever and the influence  of the weight of the BM25 retriever
        
        - using different chunking sizes and text splitting mechanisms. We concluded that the best chunking mechanism is a Sentence Transformer with a chunking size of 100 and an overlap of 10

        - we compared different prompting strategy, dealing with length of generated text and conciseneses

        - we compared two models for our generation component, llama2 (via Ollama) and GPT 3.5 Turbo (via OpenAI API). We concluded that GPT 3.5 Turbo is vastly superior to the llama2. 

These results are available in the chunking_configuration and excel_configurations directors.

During these experimtents a `max_retrieveal_limit` for the documents is set at *30*. We exported the synthetic generated testing set on [Huggingface](https://huggingface.co/datasets/prio7777777/pubmed_validation_chunking_400).

In the current state the project has the following limitations, which will be adressed in future work:
- querying after a specific date range is not bound to work (tests have shown use cases when the correct articles from that date range are returned and others when not)
- does not support chat history (new queries do not take into account old queries)
- the BM25 component of the ensemble retriever is hold in memory and is quite inneficient
- sometimes the generator uses information from its parameters that cannot be directly inferred from the given articles the retrievel passest to the prompt
- sourcing is not bound to work (in some tests only one article is cited in the generated answer and we cannot tell whether only that document has been used to answer the question or the sourcing is failing)

For future work we propose the following improvements:
- replacing the query augumentation mechanism with a MultiQuery retriever or a Step Back prompting retriever ([article](https://blog.langchain.dev/query-transformations/))
- investigating the efficiency of a Self Query retriever in combination with a reranking / fusion mechanism
- implementing history for our chat-bot
- extend the testing dataset and incorporate more types of question e.g. with date filters or author filters (*which was here limited because of the lack of computing resources*)


## Contribution table

A lot of the features where implemented during Peer Programing sessions, so it's hard to assign individual responsability to one team member. Please see infrastructure directory and **langchain-test** branch for detailled explanation of infrastructure aspects. **langchain-test** branch has the correct history of commits done for infrastructure. When they cherrypicked to main, commit times have been lost. Reason of keeping them into separate branch is also explained there.

| Feature | Member(s) |
|--------------------------|----|
| Graphical Interface (streamlit app + caching + argparser) |[Mara](README.md#L7)|
| IR component (querying, ensemble retriever, indexing and chunking) |[Andrei](README.md#L6)|
| Generation Component |[Mara](README.md#L7) + [Andrei](README.md#L6)|
| Query component (augumentation/transformation + sourcing) |[Mara](README.md#L7) + [Andrei](README.md#L6)|
| Evaluation (generating the synthetic dataset) |[Mara](README.md#L7) + [Andrei](README.md#L6)|
| Infrastructure |[Alper](README.md#L8)|
| Data Acquisition |[Mara](README.md#L7) + [Andrei](README.md#L6)|
| Testing different configurations (IR + generation) |[Mara](README.md#L7) + [Andrei](README.md#L6)|
| Documentation |[Mara](README.md#L7) + [Andrei](README.md#L6) |


## References
- P. Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. 2021.
- S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, RAGAS: Automated Evaluation of Retrieval Augmented Generation. 2023.
- V. Karpukhin et al., “Dense Passage Retrieval for Open-Domain Question Answering,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Nov. 2020, pp. 6769–6781. doi: 10.18653/v1/2020.emnlp-main.550.
- X. Ma, Y. Gong, P. He, H. Zhao, and N. Duan, Query Rewriting for Retrieval-Augmented Large Language Models. 2023.
- Y. Gu et al., “Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing,” ACM Transactions on Computing for Healthcare, vol. 3, no. 1, pp. 1–23, Oct. 2021, doi: 10.1145/3458754.
