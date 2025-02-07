from VectorStore import VectorStore
from langchain_ollama import OllamaLLM
modelname = "llama3.1"

model = OllamaLLM(model = modelname)
# print(model.invoke("tell me a joke with bananas"))

for s  in model.stream("tell me a joke with bananas"):
    print(s, end="", flush=True)

vectorStore = VectorStore(db_persist_path='Chroma')
vectorStore.ImportDocuments("Docs/STUDENT Skript Mikrobiologie WS 2425 Einleitungsskript.pdf")

print('Hello World')