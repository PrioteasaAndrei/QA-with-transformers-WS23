from opensearchpy import OpenSearch
from tqdm import tqdm
from datetime import datetime




def pubmed_index_mapping():
    '''
    Defines the OpenSearch index mapping for the PubMed dataset.

    '''
    os_mapping = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "knn": True,  # enables K-Nearest Neighbors search
            }
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "standard",  # standard text analyzer for title
                },
                "vector": {
                    "type": "knn_vector",  # KNN vector field for embedding data
                    "dimension": 768,  # dimension of the embedding vectors
                    "method": {
                        "engine": "nmslib", # nmslib engine
                        "name": "hnsw", # HNSW algorithm for KNN search
                        "space_type": "cosinesimil",  # cosine similarity distance metric
                        "parameters": {
                            "ef_construction": 40,  # construction parameters
                            "m": 8,
                        },
                    },
                },
                "publishedDate": {
                    "type": "date",  # date type for publication date
                },
                "authors": {
                    "type": "text",  # text field for author names
                },
                "journal": {
                    "type": "text",  # text field for journal names
                },
                "authors_info": {
                    "type": "text",  # text field for author information
                },
                "pubmed_text": {
                    "type": "text",  # text field for the arXiv abstract
                    "analyzer": "standard",  # standard text analyzer for the abstract
                },
                
            }
        },
    }

    return os_mapping


def opensearch_create(database_connection, index_name, os_mapping):
    """
    Creates an OpenSearch index if it does not exist.

    Args:
        database_connection: OpenSearch client object
        index_name: Name of the OpenSearch index to create
        os_mapping: Mapping object for the index

    Returns:
        None (function performs side-effect of creating the index)
    """

    # check if the index already exists
    search_index = database_connection.indices.exists(index=index_name)

    # create the index if it doesn't exist
    if not search_index:
        database_connection.indices.create(
            index=index_name,
            ignore=[400, 404],  # ignore already exists and not found errors
            body=os_mapping,
        )


def opensearch_connection(index_name,connection_settings):
    """
    Establishes a connection to OpenSearch and creates the specified index if it doesn't exist.

    Args:
        index_name: The name of the OpenSearch index to connect to or create.
        connection_settings: A dictionary containing the OpenSearch connection settings.

    Returns:
        OpenSearch: An OpenSearch client object.
    """

    # Define OpenSearch connection settings
    user_name = connection_settings["DB_USERNAME"]
    password = connection_settings["DB_PASSWORD"]
    host = connection_settings["DB_HOSTNAME"]
    port = connection_settings["DB_PORT"]

    # Create OpenSearch client object with connection details
    # host is localhost for now
    print("Trying to connect...")
    database_connection = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(user_name, password),
        use_ssl=True,
        verify_certs=False, ## Modify later
        ssl_assert_hostname=False,
        ssl_show_warn=False,
    )

    print("Connected to OpenSearch",database_connection.info())

    # Define index mapping using the `pubmed_index_mapping` function
    os_index_mapping = pubmed_index_mapping()

    # Create the index if it doesn't exist with the defined mapping
    print("Creating index...")

    opensearch_create(database_connection, index_name, os_index_mapping)

    print("Successfully created index")
    # Return the initialized OpenSearch client object
    return database_connection



def loadArticlesVector(index_connection, article_info, index_name):
  """
  Loads articles and their corresponding vector representations into the OpenSearch index.

  Args:
      index_connection: OpenSearch client object
      article_info: List of tuples containing:
        - ...
        - ...
      index_name: Name of the OpenSearch index
  """
  for article in tqdm(article_info, desc="Saving articles to database"):

    published_date = article[0]['date'] 

    if article[0]['date'] == 'no published date':
        published_date = None

    doc = {
        "title": article[0]['title'],
        "vector": article[1],
        "pubmed_text": article[0]['chunk_text'],
        'chunk_id': article[0]['chunk_id'],
        'publishedDate': published_date,
        'authors': article[0]['authors'],
        'journal': article[0]['journal'],
        'authors_info': article[0]['authors_info'],
    }

    ## TODO: add id to doc and index with given id
    index_connection.index(index=index_name, body=doc)



def create_search_body(query_embedding,min_score=1.45):
   body = {
        # **Query:** match all documents and score them based on a custom script
        "query": {
            "script_score": {
                # match all documents
                "query": {
                    "match_all": {}
                },
                # define a script to calculate the score
                "script": {
                    # since cosine similarity ranges between -1 and 1 and
                    # opensearch is not able to process negative cosine similarity score
                    # therefore +1.0 is added
                    "source": "cosineSimilarity(params.queryVector, doc['vector']) + 1.0",
                    # pass the query vector as a parameter to the script
                    "params": {
                        "queryVector": query_embedding
                    }
                }
            }
        },
        # filter results with a minimum score of 1.45
        "min_score": min_score
    }
   
   return body
