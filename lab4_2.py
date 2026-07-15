#!/usr/bin/env python3
from langchain_ollama import OllamaEmbeddings
import math

emb = OllamaEmbeddings(model="nomic-embed-text")

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)

v1 = emb.embed_query("pod keeps restarting")
v2 = emb.embed_query("container in CrashLoopBackOff")

print(f"similarity: {cosine(v1, v2):.3f}")