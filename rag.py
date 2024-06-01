from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone,ServerlessSpec,PodSpec
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dot_env, find_dotenv
from nemoguardrails import LLMRails,RailsConfig
from nemoguardrails.runnable_rails import RunnableRails

import bs4
import os
import time

load_dotenv(find_dotenv())

os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = 'RAG-Project'

class RAG:
    def __init__(self,web_url):
        self.vectorstore_index_name = "RAG-Project"
        self.web_url = web_url
        self.loader = WebBaseLoader(
            web_paths = (web_url,),
            bs_kwargs = dict(
                parse_only = bs4.SoupStrainer(
                    class_=("post-content","post-title","post-header") 
                )
            ),
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.embeddings = OpenAIEmbeddings(
            api_key= os.getenv("OPENAI_API_KEY"),model = "text-embedding-3-small"
        )    
        self.groq_llm = ChatGroq(
            api_key = "",
            model = "llama3-70b-8192",
            temperature = 0.0,
            max_tokens = 1000,
            verbose = True,

        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 2000,
            chunk_overlap = 100
        )
        self.create_pinecone_index(self.vectorstore_index_name)
        self.vectorstore = PineconeVectorStore(
            index_name = self.vectorstore_index_name,
            embeddings = self.embeddings,
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
        )
        self.rag_prompt = hub.pull(
            "rlm/rag-prompt",
            api_key= os.getenv("LANGSMITH_API_KEY")
        )
        config = RailsConfig.from_path("./config")

        self.guardrails = RunnableRails(config=config,llm = self.groq_llm)

    def create_pinecone_index(self,vectorstore_index_name):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  
        spec = ServerlessSpec(cloud='aws', region='us-east-1')  
          
            # # if not using a starter index, you should specify a pod_type too  
            # spec = PodSpec()  
        # check for and delete index if already exists  
        # vectorstore_index_name = 'langchain-retrieval-augmentation-fast'  
        if vectorstore_index_name in pc.list_indexes().names():  
            pc.delete_index(vectorstore_index_name)  
        # create a new index  
        pc.create_index(  
            vectorstore_index_name,  
            dimension=1536,  # dimensionality of text-embedding-ada-002  
            metric='dotproduct',  
            spec=spec  
        )  
        # wait for index to be initialized  
        while not pc.describe_index(vectorstore_index_name).status['ready']:  
            time.sleep(1)

    def load_docs_into_vectorstore_chain(self):
        self.docs = self.loader.load()
        self.docs = self.text_splitter.split_documents(self.docs)
        self.vectorstore.add_documents(self.docs)

    def format_docs(self,docs):
        formatted_docs = []
        for doc in docs:
            formatted_docs.append(doc.page_content)
        return formatted_docs
    
    def create_retrieval_chain(self,vectorstore_created):
        if vectorstore_created:
            pass
        else:

            self.load_docs_into_vectorstore_chain()
            self.retriever = self.vectorstore.as_retriever()
            # self.formatted_docs = self.format_docs(self.docs)
            self.rag_chain = (
                {
                    "context":self.retriever | self.format_docs, "question":
                    RunnablePassthrough()
                }
                | self.rag_prompt
                | StrOutputParser()
                | self.groq_llm
            )
            self.rag_chain = self.guardrails | self.rag_chain
    def qa(self,query,vectorstore_created):
        if vectorstore_created:
            pass
        else:
            self.create_retrieval_chain(vectorstore_created)
        return self.rag_chain.invoke(query),True
