#!/usr/bin/env python3
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

emb = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(url="http://localhost:6333")
llm = init_chat_model("ollama:llama3.2:3b-instruct-q4_K_M", temperature=0)

store = QdrantVectorStore(client=client, collection_name="chunks_800", embedding=emb)
retriever = store.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template(
    "Answer ONLY from this context. If the answer isn't in the context, "
    "say you don't have documentation for it.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

questions = [
    "what should I check if a unit shows zero WireGuard handshakes?",   # A: answerable
    "how do I configure the espresso machine in the Milan office?",     # B: honestly absent
    "what is the maximum number of thor-core replicas allowed?",        # C: adversarially absent
]

for q in questions:
    print(f"\n{'='*60}\nQ: {q}")
    print("RETRIEVED:")
    for d in retriever.invoke(q):
        print("  -", d.page_content[:60].replace("\n", " "))
    print("A:", rag_chain.invoke(q))
