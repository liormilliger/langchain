# lab1.py
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage

llm = init_chat_model("ollama:llama3.2:3b-instruct-q4_K_M", temperature=0)
# or: init_chat_model("anthropic:claude-sonnet-4-6", temperature=0)

resp = llm.invoke([
    SystemMessage("You are a terse SRE assistant."),
    HumanMessage("What does a K8s readiness probe do? One sentence."),
])
print(type(resp).__name__, "→", resp.content)

# Prints all metadata fields
print(resp.model_dump().keys())

# streaming — same runnable, different method
print("\n--- streaming ---")
for chunk in llm.stream([HumanMessage("Count to 5")]):
    print(chunk.content, end="", flush=True)
print()
