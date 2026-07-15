#!/usr/bin/env python3
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

llm = init_chat_model("ollama:llama3.2:3b-instruct-q4_K_M", temperature=0)

class IncidentReport(BaseModel):
    service: str = Field(description="Primary affected service")
    severity: str = Field(description="SEV1 (full outage) / SEV2 (degraded) / SEV3 (minor)")
    root_cause: str = Field(description="Use only one sentence and if not stated, write unknown")
    action_items: list[str] = Field(description="concrete follow-ups mentioned in the text")
    # TODO ①: add a field  root_cause  — a string; description should tell the
    #          model to use one sentence, and to write 'unknown' if not stated
    # TODO ②: add a field  action_items — a list of strings; description:
    #          concrete follow-ups mentioned in the text

structured_llm = llm.with_structured_output(IncidentReport)

with open("incident.txt") as f:
    result = structured_llm.invoke(f.read())

print(type(result))                          # what did we get back?
print(result.model_dump_json(indent=2))     # the whole object, pretty
print("---")
print("First action item:", result.action_items[0])   # typed access works