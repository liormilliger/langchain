#!/usr/bin/env python3
from langchain.tools import tool

@tool
def get_pod_status(namespace: str) -> str:
    """Get the status of all pods in a Kubernetes namespace."""
    # fake kubectl for the lab — no cluster needed
    fake = {
        "thor":   "thor-core-7d9f   1/1  Running   0   3d\npostgres-0   1/1  Running   0   9d",
        "kube-system": "coredns-x1   1/1  Running   0   30d",
    }
    return fake.get(namespace, f"No resources found in {namespace} namespace.")

# ── inspect it from the MODEL's side of the glass ──────────
print(type(get_pod_status))
print("name:       ", get_pod_status.name)
print("description:", get_pod_status.description)
print("args schema:", get_pod_status.args)

# ── it's still callable — the FRAMEWORK's side ──────────────
print("\ninvoke result:")
print(get_pod_status.invoke({"namespace": "thor"}))