#!/usr/bin/env python3
import uuid
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# 1. load + split (same as lab 6.1)
docs = TextLoader("thor-runbook.md").load()
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)

# 2. embedder + qdrant client
emb = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(url="http://localhost:6333")

# 3. the wrapper — creates the collection if it doesn't exist
store = QdrantVectorStore(
    client=client,
    collection_name="runbook_chunks",
    embedding=emb,
)

# 4. deterministic provenance ids — YOUR Q4 scheme
ids = [
    str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c.metadata['source']}:{i}"))
    for i, c in enumerate(chunks)
]

# 5. embed + upsert, one call
store.add_documents(chunks, ids=ids)
print("points:", client.get_collection("runbook_chunks").points_count)

# 6. search, one call
print("\n=== search via wrapper ===")
for doc, score in store.similarity_search_with_score("pod keeps restarting, what should I do?", k=2):
    print(f"{score:.3f}  {doc.page_content[:90].replace(chr(10), ' ')}...")