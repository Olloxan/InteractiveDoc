from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import Tuple
from Utils import Logger


class VectorStore:
    def __init__(self, db_persist_path='Recipes/Chroma', collection_name="langchain") -> None:        
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large") 
        self.db_path:str = db_persist_path
        self.collection_name:str = collection_name
        self.chroma_db = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings, collection_name=self.collection_name)        
        self.documents_and_scores : list[Tuple[Document, float]] = []
        self.last_selected_source:str = ""
        self.logger = Logger()

    


    def LogMessage(self, message:str):
        self.logger.LogMessage(message, self)
        
    def LogException(self, exception:Exception, message:str = "Processing failed"):
        self.logger.LogException(exception, message, self)