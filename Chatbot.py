from langchain_core.output_parsers import StrOutputParser 
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import Counter, List, Union, Dict
from langchain_core.runnables import RunnableBranch, RunnableLambda, Runnable, RunnableAssign
from langchain.docstore.document import Document
import pandas as pd
from io import StringIO
import json
from langchain_ollama import OllamaLLM

from VectorStore import VectorStore


from Utils import FileLoader
from Utils import Logger
from Runnables import RunnableDebugger as Debugger

class ConversationMemory(BaseModel):
    summary:str = Field('', description="Summary of the current conversation state")

class ChatbotWithHistory:
    def __init__(self):
        self.model = OllamaLLM(model = "llama3.1:8b-instruct-q8_0")

        self.debugger = Debugger()
        self.logger = Logger()
        
        self.fileLoader = FileLoader()
        
        
        self.conversation_memory:list[BaseModel] = []
        """
        list of basemodels that contains a growing knowledge base of the conversation 
        """
        self.output_validator_parser = PydanticOutputParser(pydantic_object=ConversationMemory)
        self.format_instruction_inserter = RunnableAssign({'format_instructions': lambda x: self.output_validator_parser.get_format_instructions()})   
        

        ## test
        
        self.conversation_memory.append(ConversationMemory(summary="Conversationstart"))


    def stream(self, state: dict):
        # state['message'] = user message: str
        # state['history'] = history: [[(user) None, (agent) "Hello you!"]] (List of Lists)        
        
        state['conversation_memory'] = self.conversation_memory[-1]

        self.LogMessage(f"User: {state['message']}")
        
        buffer = ""
        for token in self.chat_bot().stream(state):
            buffer += token
            yield token

        self.LogMessage(f"Agent: {buffer}")

        temp = {
            "summary" : self.conversation_memory[-1],
            "input" : state['message'],
            "answer": buffer,
            "format_instructions" : self.format_instruction_inserter
        }
        summary = self.update_conversationsummary().invoke(temp)
        self.conversation_memory.append(summary)
        self.LogMessage(f"Summary: {summary}")



    ################################ Chatbot ################################
    def chat_bot(self)->Runnable:
        # state['message'] = user message: str
        # state['history'] = history: [[(user) None, (agent) "Hello you!"]] (List of Lists)        
        
        # take current conversioation memory as context for the new message
            
        parser = StrOutputParser()
        conversation_prompt = PromptTemplate.from_template(self.fileLoader.read_text_from_file("Prompts/Conversation.prompt"))                
        return (            
            { 
                "input": itemgetter("message"),
                "context": itemgetter("conversation_memory")

            }             
            | conversation_prompt            
            | self.model                                  
            | parser            
            )
    
    def update_conversationsummary(self)->Runnable:
        summary_prompt = PromptTemplate.from_template(self.fileLoader.read_text_from_file("Prompts/ConversationSummary.prompt"))
        return (self.format_instruction_inserter | summary_prompt | self.debugger.Runnable_PrintTokencout(module=self) | self.model | self.clean_and_format_output | self.output_validator_parser)
    
    def clean_and_format_output(self, string:str)->str:
        if '{' not in string: string = '{' + string
        if '}' not in string: string = string + '}'
        string = (string
            .replace("\\_", "_")
            .replace("\n", " ")
            .replace("\]", "]")
            .replace("\[", "[")
        ) 
        return string 

    def LogMessage(self, message:str):
        self.logger.LogMessage(message, self)
        
    def LogException(self, exception:Exception, message:str = "Processing failed"):
        self.logger.LogException(exception, message, self)