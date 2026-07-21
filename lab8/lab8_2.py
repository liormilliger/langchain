#!/usr/bin/env python3
from langchain.chat_models import init_chat_model
from langchain.tools import tool

@tool
def get_pod_status(namespace: str) -> str:
    """Get the status of all pods in a Kubernetes namespace."""
    return "..."   # body irrelevant — it will never run in this lab

@tool
def get_queue_depth(queue: str) -> str:
    """Get the current message depth of an ActiveMQ queue."""
    return "..."

llm = init_chat_model("ollama:llama3.1:8b", temperature=0)
llm_with_tools = llm.bind_tools([get_pod_status, get_queue_depth])

for q in [
    "are the pods in the thor namespace healthy?",
    "how deep is the telemetry queue right now?",
    "what is the capital of Italy?",
]:
    resp = llm_with_tools.invoke(q)
    print(f"\nQ: {q}")
    print("  content:   ", repr(resp.content[:80]))
    print("  tool_calls:", resp.tool_calls)