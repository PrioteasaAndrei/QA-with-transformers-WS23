import json
from opensearch_utils import *




def evaluate(model,database_connection,evaluation_dataset_path='evaluation_dataset.json',min_score=1.45,query_type='neural'):
    """
    Evaluate the model on the evaluation dataset and return the results
    """

    # set the maximum number of results to retrieve
    size = 1000


    with open(evaluation_dataset_path, 'r') as file:
        json_data = file.read()

    parsed_json = json.loads(json_data)

    samples = len(parsed_json)

    # print("The dataset has a total of {} samples".format(samples))

    top_10_recalls = 0
    top_5_recalls = 0

    for entry in parsed_json:
        query = [entry['question']] ## don't ask
        query_vector = model.encode(query)
        query_embedding = query_vector.tolist()[0]

        if query_type == 'neural':
            search_body = create_search_body(query_embedding,min_score=min_score)
        else:
            search_body = {
                "query": {
                    "match": {
                        "pubmed_text": entry['question']
                    }
                }
            }
 

        # perform the search with a 120-second timeout
        aux_results = database_connection.search(
            index='pubmed-index',
            body=search_body,
            size=size,
            request_timeout=120
        )

        ## check the top-10 and top-5 recall
        top_first_10 = [hit['_source']['title'].lower() for hit in aux_results["hits"]["hits"][:10]]
        
        if entry["title of abstract"].lower() in top_first_10:
            top_10_recalls += 1

        if entry["title of abstract"].lower() in top_first_10[:5]:
            top_5_recalls += 1

    
    return (top_10_recalls * 1.0)/samples * 100, (top_5_recalls * 1.0)/samples * 100