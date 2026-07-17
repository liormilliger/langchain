#!/usr/bin/env python3
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. the model  ← this was missing
llm = init_chat_model("ollama:llama3.2:3b-instruct-q4_K_M", temperature=0)

# 2. the template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You write terse, actionable runbook steps."),
    ("human", "Service: {service}\nSymptom: {symptom}\nGive 3 first-response steps."),
])

# 3. the pipeline
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"service": "postgres", "symptom": "connections maxed out"})
print(result)
