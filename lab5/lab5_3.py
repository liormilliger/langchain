#!/usr/bin/env python3
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient

emb = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(url="http://localhost:6333")

query = "pod keeps restarting, what should I do?"

qv = emb.embed_query(query)                    # embedding: still OUR job

hits = client.query_points(
    collection_name="runbooks",
    query=qv,                                  # we bring the vector...
    limit=3,                                   # ...it brings the ranking
    with_payload=True,
).points

print(f"query: {query!r}\n")
for h in hits:
    print(f"{h.score:.3f}  {h.payload['text']}")
