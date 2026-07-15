# lab_p7.py
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

PROMPT = [HumanMessage("Pick a random Linux tool and name it. One word only.")]

print("--- temperature=0 ---")
llm = init_chat_model("ollama:llama3.2:3b-instruct-q4_K_M", temperature=0)
for i in range(3):
    print(f"run {i+1}:", llm.invoke(PROMPT).content)

print("--- temperature=1 ---")
llm = init_chat_model("ollama:llama3.2:3b-instruct-q4_K_M", temperature=1)
for i in range(3):
    print(f"run {i+1}:", llm.invoke(PROMPT).content)
