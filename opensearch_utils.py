from opensearchpy import OpenSearch
from tqdm import tqdm
from datetime import datetime


## NOTE: for reference
def arxiv_index_mapping():
    """
    Creates a mapping object for the OpenSearch index.

    This function defines the schema and configuration for the index, including:

    * Number of shards and replicas
    * KNN functionality
    * Data types for different fields (text, date, keyword, etc.)
    * Analyzers for text fields
    * KNN vector configuration (dimension, method, parameters)

    Args:
        None (function operates on built-in variables)

    Returns:
        os_mapping: A dictionary representing the OpenSearch index mapping.
    """

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
                "url": {
                    "type": "keyword",  # keyword field for the URL
                },
                "text_chunk_id": {
                    "type": "integer",  # integer field for text chunk ID
                },
                "arxiv_text": {
                    "type": "text",  # text field for the arXiv abstract
                    "analyzer": "standard",  # standard text analyzer for the abstract
                },
            }
        },
    }

    return os_mapping




def pubmed_index_mapping():
    '''
    ## TODO: add mapping for pubmed
    '''
    os_mapping = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                # "knn": True,  # enables K-Nearest Neighbors search
            }
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "standard",  # standard text analyzer for title
                },
                # "vector": {
                #     "type": "knn_vector",  # KNN vector field for embedding data
                #     "dimension": 768,  # dimension of the embedding vectors
                #     "method": {
                #         "engine": "nmslib", # nmslib engine
                #         "name": "hnsw", # HNSW algorithm for KNN search
                #         "space_type": "cosinesimil",  # cosine similarity distance metric
                #         "parameters": {
                #             "ef_construction": 40,  # construction parameters
                #             "m": 8,
                #         },
                #     },
                # },
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

    # Define index mapping using the `arxiv_index_mapping` function
    os_index_mapping = arxiv_index_mapping()

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
    # parse and format published date
    date_object = datetime.strptime(article[2]["published"], "%Y-%m-%d %H:%M:%S")
    iso_formatted_date = date_object.isoformat()

    # create document object with arXiv abstract data
    doc = {
        "title": article[2]["title"],
        "vector": article[1],
        "authors": article[2]["authors"],
        "url": article[2]["url"],
        "text_chunk_id": article[2]["text_chunk_id"],
        "publishedDate": iso_formatted_date,
        "arxiv_text": article[2]["arxiv_text"],
    }

    # extract ID
    _id = article[0]

    # index the document in OpenSearch with provided ID
    index_connection.index(index=index_name, body=doc, id=_id)