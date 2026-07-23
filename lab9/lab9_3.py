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
    print(f"   >>> TOOL BODY ACTUALLY RAN: restarting {name}")   # proof of execution
    return f"Deployment {name} restarted. 3 pods recreated."

llm = init_chat_model("ollama:llama3.1:8b", temperature=0)

def build_agent():
    return create_agent(
        model=llm, tools=[restart_deployment],
        system_prompt="You are an SRE assistant. Use tools to act. Do not describe; call the tool.",
        middleware=[HumanInTheLoopMiddleware(interrupt_on={
            "restart_deployment": {"allowed_decisions": ["approve", "reject"]}})],
        checkpointer=MemorySaver(),
    )

def run(decision, thread):
    agent = build_agent()
    cfg = {"configurable": {"thread_id": thread}}
    agent.invoke({"messages": [{"role": "user", "content": "restart the thor-core deployment"}]}, cfg)
    # resume with the human's decision
    result = agent.invoke(Command(resume={"decisions": [{"type": decision}]}), cfg)
    print(f"   final: {result['messages'][-1].content[:120]}")

print("=== APPROVE ===")
run("approve", "approve-1")

print("\n=== REJECT ===")
run("reject", "reject-1")