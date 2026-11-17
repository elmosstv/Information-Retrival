import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

def get_client():
    return Elasticsearch(os.environ["ES_LOCAL_URL"],
                         api_key=os.environ["ES_LOCAL_API_KEY"])

if __name__ == "__main__":
    print(get_client().info())