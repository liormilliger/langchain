#!/usr/bin/env python3
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

emb = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(url="http://localhost:6333")

# (text, env) — same 8 lines, now with an environment label
corpus = [
    ("To resume a suspended HelmRelease, run flux resume helmrelease thor-platform.", "prod"),
    ("Postgres connection pool exhaustion: check max_connections and pgbouncer settings.", "prod"),
    ("Never restart thor-core pods manually; fix the underlying HelmRelease instead.", "prod"),
    ("WireGuard peers show zero handshakes when the security group blocks UDP 13233.", "prod"),
    ("The baklava must rest overnight before serving.", "staging"),
    ("CrashLoopBackOff usually indicates the container exits shortly after start.", "staging"),
    ("ActiveMQ queue depth growing steadily suggests consumers are down or too slow.", "staging"),
    ("IMDSv2 requires a session token; tokenless metadata requests return empty.", "staging"),
]

texts = [t for t, _ in corpus]
vectors = emb.embed_documents(texts)

points = [
    PointStruct(id=i, vector=vec, payload={"text": text, "env": env, "source": "lab-corpus"})
    for i, ((text, env), vec) in enumerate(zip(corpus, vectors))
]
client.upsert(collection_name="runbooks", points=points)

query = "pod keeps restarting, what should I do?"
qv = emb.embed_query(query)

print("=== unfiltered ===")
for h in client.query_points("runbooks", query=qv, limit=3, with_payload=True).points:
    print(f"{h.score:.3f}  [{h.payload['env']}]  {h.payload['text']}")

print("\n=== env=staging only ===")
staging_filter = Filter(must=[FieldCondition(key="env", match=MatchValue(value="staging"))])
for h in client.query_points("runbooks", query=qv, limit=3,
                             with_payload=True, query_filter=staging_filter).points:
    print(f"{h.score:.3f}  [{h.payload['env']}]  {h.payload['text']}")
