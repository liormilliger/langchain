#!/usr/bin/env python3
from fpdf import FPDF

with open("thor-runbook.md") as f:
    text = f.read()

# force a page break before each "## " section → a real multi-page PDF
sections = text.split("\n## ")
sections = [sections[0]] + ["## " + s for s in sections[1:]]

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_font("Helvetica", size=11)

for section in sections:
    pdf.add_page()
    for line in section.splitlines():
        safe = line.encode("latin-1", errors="replace").decode("latin-1")
        if not safe.strip():
            pdf.ln(4)
            continue
        pdf.multi_cell(w=pdf.epw, h=6, text=safe)

pdf.output("thor-runbook.pdf")
print("pages:", len(sections))