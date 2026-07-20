#!/usr/bin/env python3
from langchain_core.documents import Document

def format_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)

fake = [
    Document(page_content="AAAA", metadata={"source": "x.md", "page": 3}),
    Document(page_content="BBBB", metadata={"source": "y.md", "page": 9}),
]

out = format_docs(fake)
print(type(out))
print(repr(out))