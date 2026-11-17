import json
from elasticsearch import helpers
from es_client import get_client

INDEX ='scifact_docs'
CORPUS_PATH = 'dataset/corpus.jsonl'

es = get_client()

if es.indices.exists(index=INDEX):
    es.indices.delete(index=INDEX)

es.indices.create(index=INDEX, mappings={
    "properties": {
        "title": {"type": "text"},
        "text": {"type": "text"}
    }
})

def actions():
    with open(CORPUS_PATH, 'r') as f:
        for line in f:
            doc = json.loads(line)
            yield {
                "_index": INDEX,
                "_id": doc["_id"],
                "_source": {
                    "title": doc["title"],
                    "text": doc["text"]
                }
            }

helpers.bulk(es, actions())
es.indices.refresh(index=INDEX)
print(f"Indexed {es.count(index=INDEX)['count']} documents into index '{INDEX}'")