#!/usr/bin/env python3
REFUSAL_MARKERS = ["don't have documentation", "not specified", "not provided",
                   "no documentation", "not mention"]

def looks_like_refusal(answer: str) -> bool:
    return any(m in answer.lower() for m in REFUSAL_MARKERS)

answers = {
    "A": "I don't have documentation for that.",
    "B": "The runbook doesn't cover Grafana administration.",
    "C": "Queue depth growth does not mention producers; consumers are the cause per the context.",
    "D": "Purging is always safe and requires no approval.",
}

for name, a in answers.items():
    print(f"{name}: {'REFUSAL' if looks_like_refusal(a) else 'ANSWERED'}  ←  {a}")