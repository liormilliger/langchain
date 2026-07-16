#!/usr/bin/env python3
import uuid
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

emb = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(url="http://localhost:6333")

docs = TextLoader("thor-runbook.md").load()
query = "pod keeps restarting, what should I do?"

for size, overlap in [(200, 30), (800, 100), (2000, 200)]:
    coll = f"chunks_{size}"

    # provision the collection ourselves — Option A habit, now in code
    if not client.collection_exists(coll):
        client.create_collection(
            coll, vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap
    ).split_documents(docs)

    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c.metadata['source']}:{size}:{i}"))
           for i, c in enumerate(chunks)]

    store = QdrantVectorStore(client=client, collection_name=coll, embedding=emb)
    store.add_documents(chunks, ids=ids)

    print(f"\n=== chunk_size={size}  ({len(chunks)} chunks) ===")
    for doc, score in store.similarity_search_with_score(query, k=2):
        head = doc.page_content[:80].replace("\n", " ")
        print(f"{score:.3f}  ({len(doc.page_content)} chars)  {head}...")
