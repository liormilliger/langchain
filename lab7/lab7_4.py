#!/usr/bin/env python3
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from golden_set import GOLDEN
from dotenv import load_dotenv

load_dotenv()  

K = 3

emb = OllamaEmbeddings(model="nomic-embed-text")
client = QdrantClient(url="http://localhost:6333")
llm = init_chat_model("ollama:llama3.1:8b", temperature=0)
# llm = init_chat_model("anthropic:claude-sonnet-4-6", temperature=0)
store = QdrantVectorStore(client=client, collection_name="chunks_800", embedding=emb)
retriever = store.as_retriever(search_kwargs={"k": K})

prompt = ChatPromptTemplate.from_template(
    "Answer ONLY from this context. If the answer isn't in the context, "
    "say you don't have documentation for it.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

REFUSAL_MARKERS = ["don't have documentation", "not specified", "not provided",
                   "no documentation", "not mention"]

def looks_like_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(m in a for m in REFUSAL_MARKERS)

hits, answer_ok, n_answerable, n_unanswerable = 0, 0, 0, 0

for item in GOLDEN:
    q, evidence = item["q"], item["evidence"]
    docs = retriever.invoke(q)
    answer = rag_chain.invoke(q)
    refused = looks_like_refusal(answer)

    if evidence is not None:
        n_answerable += 1
        hit = any(evidence in d.page_content for d in docs)
        hits += hit
        behavior_ok = not refused          # should ANSWER
        verdict = f"hit@{K}={'✓' if hit else '✗'}  behavior={'✓ answered' if behavior_ok else '✗ FALSE REFUSAL'}"
    else:
        n_unanswerable += 1
        behavior_ok = refused              # should REFUSE
        verdict = f"behavior={'✓ refused' if behavior_ok else '✗ CONFABULATED'}"

    answer_ok += behavior_ok
    print(f"\nQ: {q}\n   {verdict}\n   A: {answer[:120]}...")

print(f"\n{'='*60}")
print(f"hit@{K}:        {hits}/{n_answerable}  =  {hits/n_answerable:.2f}")
print(f"behavior:     {answer_ok}/{len(GOLDEN)}  =  {answer_ok/len(GOLDEN):.2f}")
