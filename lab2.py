#!/usr/bin/env python3
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = init_chat_model("ollama:llama3.2:3b-instruct-q4_K_M", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an SRE. Summarize pod problems in one terse paragraph: "
               "what's wrong, most likely cause, first action to take. "
               "Use ONLY facts present in the input."),
    ("human", "{describe_output}"),
])

chain = prompt | llm | StrOutputParser()

with open("pod.txt") as f:
    print(chain.invoke({"describe_output": f.read()}))
