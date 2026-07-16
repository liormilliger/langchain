#!/usr/bin/env python3
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# the same content, two formats
md_docs  = TextLoader("thor-runbook.md").load()
pdf_docs = PyPDFLoader("thor-runbook.pdf").load()

print(f"markdown: {len(md_docs)} Document(s)")
print(f"pdf:      {len(pdf_docs)} Document(s)")
print(f"\npdf doc[1] metadata: {pdf_docs[1].metadata}")
print(f"\npdf doc[1] content, first 200 chars:")
print(repr(pdf_docs[1].page_content[:200]))

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
md_chunks  = splitter.split_documents(md_docs)
pdf_chunks = splitter.split_documents(pdf_docs)
print(f"\nmd chunks: {len(md_chunks)}   pdf chunks: {len(pdf_chunks)}")
print(f"pdf chunk 0 meta: {pdf_chunks[0].metadata}")