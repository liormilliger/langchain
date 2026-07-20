#!/usr/bin/env python3
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

emb = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(url="http://localhost:6333")
store = QdrantVectorStore(client=client, collection_name="chunks_800", embedding=emb)
retriever = store.as_retriever(search_kwargs={"k": 2})

result = retriever.invoke("pod keeps restarting")

print(type(result))
print(type(result[0]))
print(result[0].page_content[:60])
print(result[0].metadata)