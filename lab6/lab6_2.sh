### Run this before 6_2.py lab

curl -X PUT 'http://localhost:6333/collections/runbook_chunks' \
  -H 'Content-Type: application/json' \
  -d '{"vectors": {"size": 768, "distance": "Cosine"}}'