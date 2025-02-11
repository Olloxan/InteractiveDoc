from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pymupdf4llm
from Utils import Logger
from sentence_transformers import CrossEncoder



from Runnables import RunnablePromptExpander



class VectorStore:
    def __init__(self, db_persist_path='', collection_name="langchain") -> None:        
        self.embeddings = OllamaEmbeddings(model="snowflake-arctic-embed2") 
        self.db_path:str = db_persist_path
        self.collection_name:str = collection_name
        self.chroma_db = Chroma(persist_directory=self.db_path, embedding_function=self.embeddings, collection_name=self.collection_name)        
        self.last_selected_source:str = ""
        self.logger = Logger()

    def ImportDocuments(self, filepath:str):
        raise NotImplementedError("Chroma already conttains documents")
        self.LogMessage(f"Importing Document {filepath}")
        pages = pymupdf4llm.to_markdown(filepath, page_chunks=True)
        
        
        splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=100,
                        length_function=len,
                        is_separator_regex=False
                    )
        documents = []
        for i, page in enumerate(pages):
            doc = Document(page['text']) 
            doc.metadata['page'] = i
            documents.append(doc)
            

            pagefragments = []
            for document in documents:
                fragments = splitter.split_text(document.page_content)
                for i, fragment in enumerate(fragments):        
                    doc = Document(fragment)
                    doc.metadata['page'] = document.metadata['page']
                    doc.metadata['chunk'] = i
                    pagefragments.append(doc)
        self.LogMessage(f"Document contains {len(documents)} pages and {len(pagefragments)} fragments")
       
        self.LogMessage("Embedding Documents...")                
        self.chroma_db = Chroma.from_documents(
                        documents=pagefragments,
                        embedding=self.embeddings,
                        persist_directory=self.db_path,
                        collection_name=self.collection_name
                    )
        self.LogMessage("Embedding finished!")
        
    def Retrieve(self, query:str):
        expander = RunnablePromptExpander(llm="llama3.1:8b-instruct-q4_K_S")
        queryStrings = expander.invoke({"input":query})
        queryStrings.append(query)
        retriever = self.chroma_db.as_retriever(search_kwargs={"k": 5})
        docs = [retriever.invoke(query) for query in queryStrings]
        
        

        unique_contents = set()
        unique_docs = []
        for sublist in docs:
            for doc in sublist:                
                if doc.page_content not in unique_contents:
                    self.LogMessage(f"retrieval Result: {doc.page_content}")
                    unique_docs.append(doc)
                    unique_contents.add(doc.page_content)
        unique_contents = list(unique_contents)
        cross_encoder = CrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', cache_dir='Data')      
        
        pairs = []
        for doc in unique_contents:
            pairs.append([query, doc])
        scores = cross_encoder.predict(pairs)
        scored_docs = zip(scores, unique_contents)
        sorted_docs = sorted(scored_docs, reverse=True)

        reranked_docs = [doc for _, doc in sorted_docs][0:8]
        

        pass    

    def LogMessage(self, message:str):
        self.logger.LogMessage(message, self)
        
    def LogException(self, exception:Exception, message:str = "Processing failed"):
        self.logger.LogException(exception, message, self)