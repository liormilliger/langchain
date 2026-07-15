#!/usr/bin/env python3
from langchain_ollama import OllamaEmbeddings

emb = OllamaEmbeddings(model="nomic-embed-text")

vector = emb.embed_query("pod keeps restarting")

print(type(vector))          # what kind of thing did we get?
print(len(vector))           # how many numbers?
print(vector[:5])            # peek at the first five
