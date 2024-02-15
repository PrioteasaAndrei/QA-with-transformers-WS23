import json
import time
from Bio import Entrez

def search(query, max_num_articles):
    """
    Retrieve the ids of first `max_num_articles` based on the provided query
    """
    handle = Entrez.esearch(db='pubmed', 
                            sort='relevance', 
                            retmax=max_num_articles,
                            retmode='xml', 
                            term=query)
    results = Entrez.read(handle)
    return results

def fetch_details(id_list):
    """
    Fetch the metadata of PubMed articles based on their IDs
    """
    ids = ','.join(id_list)
    handle = Entrez.efetch(db='pubmed', 
                        retmode='xml', 
                        id=ids)
    results = Entrez.read(handle)
    return results

def get_pubmed_data( 
    start_date: str,
    end_date: str,
    email: str = 'mara.eliana.popescu@gmail.com', 
    max_num_articles: int = 10000
):
    """
    Download the first `max_num_articles` pubmed abstracts published between `start_date` and `end_date`
    
    Parameters: 
    ----------
    start_date: Start date, in the format of "%Y/%m/%d", for the PubMed article search based on their publication date.
    end_date: End date for, in the format of "%Y/%m/%d", the PubMed articles search based on their publication date.
    """
    # Always provide your email when using Entrez
    Entrez.email = email

    # Format the date range in YYYY/MM/DD format for the query
    query = f"intelligence[Title/abstract] AND ({start_date}[Date - Publication] : {end_date}[Date - Publication])"
    start_time = time.time()

    # Search for articles
    print("Searching for articles...")
    results = search(query, max_num_articles=max_num_articles)
    id_list = results['IdList']
    print(f"Total number of articles found between {start_date}-{end_date}: {len(id_list)}.")

    # Fetch details of retrieved articles
    print("Fetching articles.")
    papers = fetch_details(id_list)

    result = {}
    for i, paper in enumerate(papers['PubmedArticle']):
        abstract = paper['MedlineCitation']['Article'].get('Abstract')
        date = paper['MedlineCitation']['Article'].get('ArticleDate')
        authors = paper['MedlineCitation']['Article'].get('AuthorList')
        if authors:
            authors_list = []
            for author in authors:
                if 'LastName' in author.keys() and 'ForeName' in author.keys():
                    lastname = author['LastName']
                    forename = author['ForeName']
                    authors_list.append(forename + ' ' + lastname)
        else:
            authors_list = []
        if date:
            year = date[0]['Year']
            month = date[0]['Month']
            day = date[0]['Day']
            pub_date = f"{day}/{month}/{year}"
        else:
            pub_date = ""
        if abstract:
            result[i] = {
                "title": paper['MedlineCitation']['Article']['ArticleTitle'],
                "abstract": abstract['AbstractText'][0],
                "publication_date": pub_date,
                "authors": authors_list
            }
        
    print(f"{len(result)} PubMed articles were downloadded in {time.time()-start_time}.")
    return list(result.values())
    

def save_data(output_json_file, data):
    print("Saving data.")
    with open(output_json_file, 'w') as f:
        f.write(json.dumps(data))
    print("Data was succesfully saved.")
    

if __name__ == "__main__":
    output_json_file = ".data/pubmed_articles.json"
    start_dates = ["2013/1/1", "2018/1/1", "2020/1/1", "2021/1/1", "2021/7/1", "2022/1/1", "2022/7/1", "2023/1/1", "2023/7/1"]
    end_dates = ["2017/12/31", "2019/12/31", "2020/12/31", "2021/6/30", "2021/12/31", "2022/6/30", "2022/12/31", "2023/6/30", "2023/12/31"]
    data = []
    for start_date, end_date in zip(start_dates, end_dates):
        result =  get_pubmed_data(start_date, end_date)
        print(f"Retrieved all data between {start_date}-{end_date}.")
        data.extend(result)
    # Store data in json format
    save_data(output_json_file, data)