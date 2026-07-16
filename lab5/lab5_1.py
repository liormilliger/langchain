curl -X PUT 'http://54.166.224.125:6333/collections/runbooks' \
  -H 'Content-Type: application/json' \
  -d '{"vectors": {"size": 768, "distance": "Cosine"}}'
