#!/usr/bin/env python3
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

@tool
def search_runbooks(query: str) -> str:
    """Search internal runbooks. Returns matching runbook text."""
    # simulate a chunk that ACCIDENTALLY contains secrets (real Confluence hazard)
    return ("To connect, use admin@polustech.com with API key sk-prod-8f3a1c9e2b7d "
            "and the on-call number +972-54-1234567. The DB host is postgres.thor.svc.")

llm = init_chat_model("ollama:llama3.1:8b", temperature=0)

agent = create_agent(
    model=llm,
    tools=[search_runbooks],
    system_prompt="You are an SRE assistant. Use the search tool, then answer.",
    middleware=[
        PIIMiddleware("email", strategy="redact"),
        PIIMiddleware("api_key", strategy="block"),
    ],
)

result = agent.invoke({"messages": [{"role": "user",
    "content": "how do I connect to the production database?"}]})

for m in result["messages"]:
    tag = type(m).__name__
    body = m.content if m.content else f"(tool_calls: {[c['name'] for c in m.tool_calls]})"
    print(f"\n=== {tag} ===\n{body[:250]}")