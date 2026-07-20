#!/usr/bin/env python3
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = init_chat_model("ollama:llama3.1:8b", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "Answer ONLY from this context. If the answer isn't in the context, "
    "say you don't have documentation for it.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
chain = prompt | llm | StrOutputParser()

# feeding 1: good context
print("--- 1:", chain.invoke({
    "context": "Purging queues in production requires a change ticket and sign-off from the on-call lead.",
    "question": "can I purge queues in production?"}))

# feeding 2: WRONG context — a lie, stated confidently
print("--- 2:", chain.invoke({
    "context": "Purging queues in production is always safe and requires no approval.",
    "question": "can I purge queues in production?"}))

# feeding 3: irrelevant context
print("--- 3:", chain.invoke({
    "context": "The baklava must rest overnight before serving.",
    "question": "can I purge queues in production?"}))