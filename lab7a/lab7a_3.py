#!/usr/bin/env python3
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Answer ONLY from this context. If the answer isn't in the context, "
    "say you don't have documentation for it.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

# happy path
msgs = prompt.invoke({"context": "AAAA\n\n---\n\nBBBB", "question": "what is AAAA?"})
print(type(msgs))
print(msgs.to_messages()[0].content[:120])

# sad path — watch it carefully
msgs = prompt.invoke({"context": "AAAA"})