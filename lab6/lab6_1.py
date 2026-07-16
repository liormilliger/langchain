# Lab 6.1 — Load, split, inspect

#!/usr/bin/env python3
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_core.documents import Document
# docs = [Document(page_content=open("thor-runbook.md").read(),
#                  metadata={"source": "thor-runbook.md"})]
### This will work as well without deprecation warning

docs = TextLoader("thor-runbook.md").load()

print(f"loaded: {len(docs)} Document(s)")
print(f"metadata: {docs[0].metadata}")
print(f"total chars: {len(docs[0].page_content)}\n")

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

print(f"chunks: {len(chunks)}\n")
for i, c in enumerate(chunks):
    print(f"--- chunk {i}  ({len(c.page_content)} chars)  meta={c.metadata}")
    print(c.page_content[:120].replace("\n", " ") + " ...")
    print(f"    ...ends with: ...{c.page_content[-80:].replace(chr(10), ' ')}\n")