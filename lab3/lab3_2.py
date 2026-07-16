#!/usr/bin/env python3
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Literal

llm = init_chat_model("ollama:llama3.2:3b-instruct-q4_K_M", temperature=0)

class IncidentReport(BaseModel):
    service: str = Field(description="Primary affected service")
    severity: Literal["SEV1", "SEV2", "SEV3"] = Field(description="SEV1=full outage, SEV2=degraded, SEV3=minor")
    root_cause: str = Field(description="Use only one sentence and if not stated, write unknown")
    action_items: list[str] = Field(description="concrete follow-ups mentioned in the text")


structured_llm = llm.with_structured_output(IncidentReport)

with open("incident.txt") as f:
    result = structured_llm.invoke(f.read())

print(type(result))                          # what did we get back?
print(result.model_dump_json(indent=2))     # the whole object, pretty
print("---")
print("First action item:", result.action_items[0])   # typed access works