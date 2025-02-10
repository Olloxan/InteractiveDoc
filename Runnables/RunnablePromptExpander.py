from langchain_ollama import OllamaLLM
from Utils import FileLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from Runnables import RunnableDebugger as Debugger
from Utils import Logger


class RunnablePromptExpander(Runnable):
    def __init__(self, llm):
        self.llm = OllamaLLM(model = llm)
        fileloader = FileLoader()
        self.extraction_prompt = PromptTemplate.from_template(fileloader.read_text_from_file("Prompts/PromptExpansion.prompt"))
        self.debugger = Debugger()
        self.logger = Logger()

    def invoke(self, state:dict, config=None)->list[str]:
        """ expected dict: state['input'] = text """ 
        try:
            self.LogMessage(f"Expanding prompt for input: {state['input']}")
            result = self.expandPromt().invoke(state)
            lines =  [item for item in result.split("\n") if item.strip()]            
        except Exception as e:
            self.LogException(e)
        return lines
            
    def expandPromt(self)->Runnable:
        return (  self.extraction_prompt 
                | self.debugger.Runnable_PrintTokencout(module=self) 
                | self.llm                 
                | self.debugger.Runnable_PrintStructureWithLabel(label="Modeloutput: ", module=self)                
                )
    
    def LogMessage(self, message:str):
        self.logger.LogMessage(message, self)
        
    def LogException(self, exception:Exception, message:str = "Processing failed"):
        self.logger.LogException(exception, message, self)