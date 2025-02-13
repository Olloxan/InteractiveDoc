from UserInterface import UserInterface
from Chatbot import ChatbotWithHistory


# from VectorStore import VectorStore
# from langchain_ollama import OllamaLLM
# modelname = "llama3.1"

# model = OllamaLLM(model = modelname)
# print(model.invoke("tell me a joke with bananas"))

# for s  in model.stream("tell me a joke with bananas"):
#     print(s, end="", flush=True)

# vectorStore = VectorStore(db_persist_path='Chroma')
# vectorStore.ImportDocuments("Docs/STUDENT Skript Mikrobiologie WS 2425 Einleitungsskript.pdf")
# vectorStore.Retrieve("Ich suche Nachweis der Katalase.")

# https://github.com/Coding-Crashkurse/Applied-Advanced-RAG/blob/main/code.ipynb
# https://www.youtube.com/watch?v=3w_D1L0F-uE&t=710s


chatbot = ChatbotWithHistory()
state = {}
def chat_gen(message, history=[], return_buffer=True):        
    # state['message'] = user message: str
    # state['history'] = history: [[(user) None, (agent) "Hello you!"]] (List of Lists)
    
    buffer = ""             
    state['message'] = message
    state['history'] = history
    for token in chatbot.stream(state):           
        buffer += token
        yield buffer if return_buffer else token

interface = UserInterface(chat_fn=chat_gen)
interface.render()