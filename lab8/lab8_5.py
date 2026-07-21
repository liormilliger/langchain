#!/usr/bin/env python3
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def get_pod_status(namespace: str) -> str:
    """Get the status of all pods in a Kubernetes namespace. Use for questions about pod health, restarts, or crashes."""
    return {"thor": "thor-core-7d9f  0/1  CrashLoopBackOff  7  3d\npostgres-0  1/1  Running  0  9d"}.get(
        namespace, f"No resources found in {namespace}.")

@tool
def get_recent_events(namespace: str) -> str:
    """Get recent Kubernetes events for a namespace. Use to find WHY something is failing."""
    return {"thor": "Warning  BackOff  pod/thor-core-7d9f  Back-off restarting failed container\n"
                     "Warning  Failed   pod/thor-core-7d9f  Error: secret 'thor-mq-credentials' not found"}.get(
        namespace, f"No events for {namespace}.")

llm = init_chat_model("ollama:llama3.1:8b", temperature=0)
agent = create_agent(
    model=llm,
    tools=[get_pod_status, get_recent_events],
    system_prompt="Always use the available tools to gather evidence. Do not describe what you would do; call the tool."
    # system_prompt="You are an SRE assistant. Investigate thoroughly using tools before answering.",
)

result = agent.invoke({"messages": [{"role": "user",
    "content": "thor-core keeps crashing — what's the root cause?"}]})

for m in result["messages"]:
    tag = type(m).__name__
    if getattr(m, "tool_calls", None):
        print(f"{tag}: →calls {[c['name'] for c in m.tool_calls]}")
    else:
        print(f"{tag}: {m.content[:110]}")