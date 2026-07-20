#!/usr/bin/env python3
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.runnables import RunnablePassthrough

emb = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(url="http://localhost:6333")
store = QdrantVectorStore(client=client, collection_name="chunks_800", embedding=emb)
retriever = store.as_retriever(search_kwargs={"k": 2})

def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

fork = {"context": retriever | format_docs, "question": RunnablePassthrough()}

from langchain_core.runnables import RunnableParallel
fork_runnable = RunnableParallel(fork)     # the dict, made invokable on its own

out = fork_runnable.invoke("can I purge queues in production?")

print(type(out))
print(out.keys())
print("QUESTION value:", repr(out["question"]))
print("CONTEXT value, first 150:", repr(out["context"][:150]))