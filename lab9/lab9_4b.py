#!/usr/bin/env python3
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

@tool
def search_runbooks(query: str) -> str:
    """Search internal runbooks. Returns matching runbook text."""
    return ("To connect, email admin@polustech.com with API key sk-prod-8f3a1c9e2b7d, "
            "or reach the DB at https://postgres.thor.svc:5432.")

llm = init_chat_model("ollama:llama3.1:8b", temperature=0)

agent = create_agent(
    model=llm,
    tools=[search_runbooks],
    system_prompt="You are an SRE assistant. Use the search tool, then answer.",
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_tool_results=True),
        PIIMiddleware("url",   strategy="redact", apply_to_tool_results=True),                          # built-in
        PIIMiddleware("api_key", detector=r"sk-[a-z]+-[a-f0-9]{12}",        # custom: pass a STRING
                      strategy="block", apply_to_tool_results=True),
    ],
)

result = agent.invoke({"messages": [{"role": "user",
    "content": "how do I connect to the production database?"}]})

for m in result["messages"]:
    tag = type(m).__name__
    body = m.content if m.content else f"(tool_calls: {[c['name'] for c in m.tool_calls]})"
    print(f"\n=== {tag} ===\n{body[:250]}")