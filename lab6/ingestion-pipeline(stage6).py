docs   = PyPDFLoader("runbook.pdf").load()                         # 1 load
chunks = RecursiveCharacterTextSplitter(800, 100).split_documents(docs)   # 2 split
ids    = [str(uuid.uuid5(uuid.NAMESPACE_DNS,                      # 3 identity
             f"{c.metadata['source']}:{c.metadata['page']}:{i}"))
          for i, c in enumerate(chunks)]
store.add_documents(chunks, ids=ids)                               # 4 embed + upsert