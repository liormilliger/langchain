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

files = ["incident1.txt", "incident2.txt", "incident3.txt"]
texts = []
for name in files:
    with open(name) as f:
        texts.append(f.read())

# TODO ②: one line — run all of texts through structured_llm concurrently
reports = structured_llm.batch(["incident1.txt", "incident2.txt", "incident3.txt"])

for name, report in zip(files, reports):
    print(f"=== {name} ===")
    print(report.model_dump_json(indent=2))
    print()