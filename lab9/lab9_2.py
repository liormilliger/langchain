#!/usr/bin/env python3
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

@tool
def restart_deployment(name: str) -> str:
    """Restart a Kubernetes deployment by name. THIS MUTATES CLUSTER STATE."""
    return f"Deployment {name} restarted. 3 pods recreated."

llm = init_chat_model("ollama:llama3.1:8b", temperature=0)

agent = create_agent(
    model=llm,
    tools=[restart_deployment],
    system_prompt="You are an SRE assistant. Use tools to act. Do not describe; call the tool.",
    middleware=[HumanInTheLoopMiddleware(interrupt_on={
        "restart_deployment": {"allowed_decisions": ["approve", "reject"]}
    })],
    checkpointer=MemorySaver(),   # required — interrupts need persisted state to resume
)
cfg = {"configurable": {"thread_id": "restart-1"}}

# ── invoke 1: the agent will WANT to restart — watch it pause ──
result = agent.invoke(
    {"messages": [{"role": "user", "content": "restart the thor-core deployment"}]}, cfg)

print("=== after invoke 1 ===")
print("last message type:", type(result["messages"][-1]).__name__)
if "__interrupt__" in result:
    print("INTERRUPTED — awaiting human decision:")
    print(result["__interrupt__"])
else:
    print("no interrupt; final content:", result["messages"][-1].content[:120])