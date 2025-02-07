from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import pymupdf4llm
from Utils import Logger


class VectorStore:
    def __init__(self, db_persist_path='', collection_name="langchain") -> None:        
        self.embeddings = OllamaEmbeddings(model="snowflake-arctic-embed2") 
        self.db_path:str = db_persist_path
        self.collection_name:str = collection_name
        self.chroma_db = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings, collection_name=self.collection_name)        
        self.last_selected_source:str = ""
        self.logger = Logger()

    def ImportDocuments(self, filepath:str):
        pages = pymupdf4llm.to_markdown(filepath, page_chunks=True)
        self.LogMessage("preparing Documents")
        documents = []
        for i, page in enumerate(pages):
            doc = Document(page['text']) 
            doc.metadata['page'] = i
            documents.append(doc)
            splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=100,
                        length_function=len,
                        is_separator_regex=False
                    )

            pagefragments = []
            for document in documents:
                fragments = splitter.split_text(document.page_content)
                for i, fragment in enumerate(fragments):        
                    doc = Document(fragment)
                    doc.metadata['page'] = document.metadata['page']
                    doc.metadata['chunk'] = i
                    pagefragments.append(doc)
        self.LogMessage("Imorting Documents into Docarray")
        vectoreStore = DocArrayInMemorySearch.from_documents(pagefragments, embedding=self.embeddings)
        self.LogMessage("Imorting Documents into Chroma")
        self.chroma_db = Chroma.from_documents(
                        documents=pagefragments,
                        embedding=self.embeddings,
                        persist_directory=self.db_path
                    )
        self.LogMessage("finished Imorting Documents into Chroma")




    def LogMessage(self, message:str):
        self.logger.LogMessage(message, self)
        
    def LogException(self, exception:Exception, message:str = "Processing failed"):
        self.logger.LogException(exception, message, self)