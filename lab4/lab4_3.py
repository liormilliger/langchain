#!/usr/bin/env python3
from langchain_ollama import OllamaEmbeddings
import math

emb = OllamaEmbeddings(model="nomic-embed-text")

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)))

sentences = [
    "pod keeps restarting",                          # 0
    "the pod is restarting over and over again",     # 1  plain-English paraphrase of 0
    "never restart the pod",                         # 2  negation of 0
    "my grandmother's baklava recipe",               # 3  another continent
]

vectors = emb.embed_documents(sentences)   # ← the sibling: list of strings → list of vectors

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"[{i}]×[{j}]  {cosine(vectors[i], vectors[j]):.3f}   {sentences[i]!r} × {sentences[j]!r}")