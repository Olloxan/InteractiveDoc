from langchain_community.llms import Ollama
import pymupdf4llm
import pathlib

md_text = pymupdf4llm.to_markdown("Masterarbeit.pdf")

output_file = pathlib.Path("output.md")
output_file.write_bytes(md_text.encode())

print(md_text)

print('Hello World')