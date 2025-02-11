from VectorStore import VectorStore
from langchain_ollama import OllamaLLM
modelname = "llama3.1"

# model = OllamaLLM(model = modelname)
# print(model.invoke("tell me a joke with bananas"))

# for s  in model.stream("tell me a joke with bananas"):
#     print(s, end="", flush=True)

vectorStore = VectorStore(db_persist_path='Chroma')
# vectorStore.ImportDocuments("Docs/STUDENT Skript Mikrobiologie WS 2425 Einleitungsskript.pdf")
vectorStore.Retrieve("Ich suche Nachweis der Katalase.")

print('Hello World')

# https://github.com/Coding-Crashkurse/Applied-Advanced-RAG/blob/main/code.ipynb
# https://www.youtube.com/watch?v=3w_D1L0F-uE&t=710s