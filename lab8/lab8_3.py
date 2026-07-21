#!/usr/bin/env python3
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, ToolMessage

@tool
def get_pod_status(namespace: str) -> str:
    """Get the status of all pods in a Kubernetes namespace."""
    fake = {"thor": "thor-core-7d9f   0/1  CrashLoopBackOff   7   3d\npostgres-0   1/1  Running   0   9d"}
    return fake.get(namespace, f"No resources found in {namespace} namespace.")

llm = init_chat_model("ollama:llama3.1:8b", temperature=0)
llm_with_tools = llm.bind_tools([get_pod_status])

messages = [HumanMessage("are the pods in the thor namespace healthy?")]

# ── turn 1: the model decides ───────────────────────────────
resp1 = llm_with_tools.invoke(messages)
print("turn 1 tool_calls:", resp1.tool_calls)
messages.append(resp1)                          # the request joins the transcript

# ── WE are the framework: execute the call ─────────────────
call = resp1.tool_calls[0]
result = get_pod_status.invoke(call["args"])    # actually run the function
print("tool result:", result[:60], "...")
messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

# ── turn 2: the model reads the result ─────────────────────
resp2 = llm_with_tools.invoke(messages)
print("\nturn 2 tool_calls:", resp2.tool_calls)
print("turn 2 content:", resp2.content)
