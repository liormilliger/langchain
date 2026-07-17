# Each entry: question, evidence_substring (None = refusal is the correct behavior).
# evidence_substring: a string that MUST appear in a retrieved chunk for the
# retrieval to count as a hit — pick something distinctive, not a common word.

GOLDEN = [
    # ── answerable ──────────────────────────────────────────
    {"q": "pod keeps restarting, what should I do?",
     "evidence": "flux resume helmrelease"},
    {"q": "unit shows zero WireGuard handshakes",
     "evidence": "UDP 13233"},
    {"q": "ActiveMQ queue depth keeps growing, why?",
     "evidence": "consumers are down or too slow"},
    {"q": "app has intermittent timeouts but database CPU is low",
     "evidence": "pgbouncer"},
    {"q": "can I purge queues in production?",
     "evidence": "change ticket"},
    {"q": "how long until service recovers after resuming the HelmRelease?",
     "evidence": "two reconciliation cycles"},
    {"q": "I recreated my secrets, whats next?",
     "evidence": "flux resume helmrelease"},
    {"q": "My database connection pool shaow exhausted, what do I do?",
     "evidence": "max_connections"},

    # ── unanswerable (refusal correct) ──────────────────────
    {"q": "what is the maximum number of thor-core replicas allowed?",
     "evidence": None},
    {"q": "how do I rotate the Grafana admin password?",
     "evidence": None},
    {"q": "With which helmRevision should I reconcile FluxCD?",
     "evidence": None},
    {"q": "How can I reach the UI in the browser?",
     "evidence": None}

]