#!/usr/bin/env python3
import uuid
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

emb = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(url="http://localhost:6333")
COLL = "chunks_800"

def ingest(path: str):
    docs = TextLoader(path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c.metadata['source']}:800:{i}"))
           for i, c in enumerate(chunks)]
    store = QdrantVectorStore(client=client, collection_name=COLL, embedding=emb)
    store.add_documents(chunks, ids=ids)
    return len(chunks)

def count():
    return client.get_collection(COLL).points_count

print("start:", count())

n = ingest("thor-runbook.md")
print(f"after re-ingest of runbook ({n} chunks):", count())

n = ingest("wireguard-notes.md")            # ← a SECOND source file — create below
print(f"after ingesting wireguard-notes ({n} chunks):", count())

n = ingest("wireguard-notes.md")            # nightly re-run of it
print(f"after re-ingesting wireguard-notes:", count())

# Guillaume's moment: wireguard-notes.md gets deleted from the source system
client.delete(
    collection_name=COLL,
    points_selector=Filter(must=[
        FieldCondition(key="metadata.source", match=MatchValue(value="wireguard-notes.md"))
    ]),
)
print("after surgical delete of wireguard-notes:", count())
