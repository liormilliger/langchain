#!/usr/bin/env python3
from langchain_ollama import OllamaEmbeddings
import math

emb = OllamaEmbeddings(model="nomic-embed-text")

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)))

corpus = [
    "To resume a suspended HelmRelease, run flux resume helmrelease thor-platform.",
    "Postgres connection pool exhaustion: check max_connections and pgbouncer settings.",
    "Never restart thor-core pods manually; fix the underlying HelmRelease instead.",
    "WireGuard peers show zero handshakes when the security group blocks UDP 13233.",
    "The baklava must rest overnight before serving.",
    "CrashLoopBackOff usually indicates the container exits shortly after start.",
    "ActiveMQ queue depth growing steadily suggests consumers are down or too slow.",
    "IMDSv2 requires a session token; tokenless metadata requests return empty.",
]

query = "pod keeps restarting, what should I do?"

doc_vectors = emb.embed_documents(corpus)
query_vector = emb.embed_query(query)

scored = []
for text, vec in zip(corpus, doc_vectors):
    scored.append((cosine(query_vector, vec), text))

scored.sort(reverse=True)          # highest similarity first

print(f"query: {query!r}\n")
for score, text in scored[:3]:     # top k = 3
    print(f"{score:.3f}  {text}")
