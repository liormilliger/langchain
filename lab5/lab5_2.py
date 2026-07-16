#!/usr/bin/env python3
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

emb = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(url="http://localhost:6333")

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

vectors = emb.embed_documents(corpus)

points = []
for i, (text, vec) in enumerate(zip(corpus, vectors)):
    points.append(PointStruct(
        id=i,                                   # our id scheme: position in corpus
        vector=vec,                             # the 768 floats — machine-made
        payload={"text": text, "source": "lab-corpus"},
    ))

client.upsert(collection_name="runbooks", points=points)

info = client.get_collection("runbooks")
print("points in collection:", info.points_count)
