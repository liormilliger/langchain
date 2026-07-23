#!/usr/bin/env python3
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

llm = init_chat_model("ollama:llama3.1:8b", temperature=0)

def ask(agent, text, cfg=None):
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, cfg)
    return result["messages"][-1].content

# ── Agent A: NO checkpointer (amnesiac) ─────────────────────
agent_a = create_agent(model=llm, tools=[])

print("=== Agent A (no memory) ===")
print("turn 1:", ask(agent_a, "My name is Lior and I run the thor namespace.")[:80])
print("turn 2:", ask(agent_a, "What is my name and which namespace do I run?")[:80])

# ── Agent B: WITH checkpointer + thread_id ──────────────────
agent_b = create_agent(model=llm, tools=[], checkpointer=MemorySaver())
cfg = {"configurable": {"thread_id": "lior-session-1"}}

print("\n=== Agent B (memory, thread=lior-session-1) ===")
print("turn 1:", ask(agent_b, "My name is Lior and I run the thor namespace.", cfg)[:80])
print("turn 2:", ask(agent_b, "What is my name and which namespace do I run?", cfg)[:80])

# ── Agent B, DIFFERENT thread — isolation test ──────────────
cfg2 = {"configurable": {"thread_id": "someone-else"}}
print("\n=== Agent B (memory, thread=someone-else) ===")
print("turn 1:", ask(agent_b, "What is my name and which namespace do I run?", cfg2)[:80])