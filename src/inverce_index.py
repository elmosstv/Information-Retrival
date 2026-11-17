import math
from collections import Counter, defaultdict
from elasticsearch import helpers
from es_client import get_client
from preprocessing import preprocess

DOCS_INDEX = 'scifact_docs'
TERMS_INDEX = 'scifact_inverted_index'
DOC_STATS_INDEX = 'scifact_doc_stats'
COLL_STATS_INDEX = 'scifact_collection_stats'

es = get_client()

doc_term_freqs = {}

# Read all documents from the Elasticsearch index and compute term frequencies
for hit in helpers.scan(es, index=DOCS_INDEX, query={"query": {"match_all": {}}}):
    src = hit["_source"]
    tokens = preprocess(src["title"] + " " + src["text"])
    doc_term_freqs[hit["_id"]] = Counter(tokens)

#Inverted index structure: term -> {doc_id: term_freq}
inverted_index = defaultdict(list)
for doc_id , term_freqs in doc_term_freqs.items():
    for term, freq in term_freqs.items():
        inverted_index[term].append({"doc_id": doc_id, "term_freq": freq})

#Compute document statistics and collection statistics
N = len(doc_term_freqs)  # Total number of documents
doc_len = {doc_id: sum(term_freqs.values()) for doc_id, term_freqs in doc_term_freqs.items()}  # Document lengths
avg_doc_len = sum(doc_len.values()) / N  # Average document length
doc_freqs = {term: len(postings) for term, postings in inverted_index.items()}  # Document frequencies
idf_vsm = {term: math.log(N / df) for term, df in doc_freqs.items()}  # IDF for VSM
idf_bm25 = {term: math.log((N - df + 0.5) / (df + 0.5)+ 1) for term, df in doc_freqs.items()}  # IDF for BM25

# Compute document norms for VSM
doc_norm = {
    doc_id: math.sqrt(sum((tf * idf_vsm[term]) ** 2 for term, tf in term_freqs.items()))
    for doc_id, term_freqs in doc_term_freqs.items()
}

def recreate_index(index_name, mappings):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
    es.indices.create(index=index_name, mappings=mappings)

recreate_index(TERMS_INDEX, {"properties": {
        "term": {"type": "keyword"},
        "doc_freq": {"type": "integer"},
        "idf_vsm": {"type": "float"},
        "idf_bm25": {"type": "float"},
        "postings": {"type": "object", "enabled": False}
}})

recreate_index(DOC_STATS_INDEX, {"properties": {
    "length": {"type": "integer"},
    "vsm_norm": {"type": "float"}
}})

recreate_index(COLL_STATS_INDEX, {"properties": {
    "N": {"type": "integer"},
    "avg_doc_len": {"type": "float"},
    "vocab_size": {"type": "integer"}
}})

#Store everything in Elasticsearch

helpers.bulk(es, (
    {
        "_index": TERMS_INDEX,
        "_id": term,
        "_source": {
            "term": term,
            "doc_freq": doc_freqs[term],
            "idf_vsm": idf_vsm[term],
            "idf_bm25": idf_bm25[term],
            "postings": postings,
        },
    }
    for term, postings in inverted_index.items()
))

helpers.bulk(es, ({
    "_index": DOC_STATS_INDEX,
    "_id": doc_id,
    "_source": { "length": doc_len[doc_id], "vsm_norm": doc_norm[doc_id] }
    } for doc_id in doc_term_freqs.keys()
))

es.index(index=COLL_STATS_INDEX, id="scifact", document={"N": N, "avg_doc_len": avg_doc_len, "vocab_size": len(inverted_index)})

for index_name in [TERMS_INDEX, DOC_STATS_INDEX, COLL_STATS_INDEX]:
    es.indices.refresh(index=index_name)

print(f"Docs: {N} | Terms: {len(inverted_index)} | avgdl: {avg_doc_len:.1f}")
print("Terms in ES:", es.count(index=TERMS_INDEX)["count"])
print("Doc stats in ES:", es.count(index=DOC_STATS_INDEX)["count"])
print("Coll stats in ES:", es.count(index=COLL_STATS_INDEX)["count"])