#!/usr/bin/env python3
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

emb = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(url="http://localhost:6333")

query = "pod keeps restarting, what should I do?"
qv = emb.embed_query(query)

staging = Filter(must=[FieldCondition(key="env", match=MatchValue(value="staging"))])

print("=== Act 1: staging, k=3, score_threshold=0.45 ===")
hits = client.query_points(
    "runbooks", query=qv, limit=3,
    with_payload=True, query_filter=staging,
    score_threshold=0.45,
).points
print(f"({len(hits)} hits)")
for h in hits:
    print(f"{h.score:.3f}  [{h.payload['env']}]  {h.payload['text']}")

print("\n=== Act 2: delete all staging points ===")
client.delete(collection_name="runbooks", points_selector=staging)
print("points remaining:", client.get_collection("runbooks").points_count)

print("\n=== same staging search, after the purge ===")
hits = client.query_points(
    "runbooks", query=qv, limit=3,
    with_payload=True, query_filter=staging,
).points
print(f"({len(hits)} hits)")

print("\n=== unfiltered, after the purge ===")
for h in client.query_points("runbooks", query=qv, limit=3, with_payload=True).points:
    print(f"{h.score:.3f}  [{h.payload['env']}]  {h.payload['text']}")