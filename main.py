from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()
#indexing

# loader= PyPDFLoader(file_path="books/DS BOOK 1.pdf")
# documents=loader.load()

# splitter=RecursiveCharacterTextSplitter(
#     chunk_size=1200,
#     chunk_overlap=250
# )
# chunks=[]
# for doc in documents:
#     chunks.extend(splitter.split_text(doc.page_content))
# # embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
em=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",google_api_key=os.getenv("GOOGLE_API_KEY"))
vectorstore = Chroma(
    persist_directory="chroma1",
    collection_name="study_bud_gemini",  # new collection
    embedding_function=em  # your 3072-dim embedding function
)
# vectorstore.add_texts(chunks)


prompt=PromptTemplate(template='''You are a helpful assistant who reads everything with great detail,
                      answer this query {query} with the following context {context} ,if you cant find the answer with
                      the given context say i dont know''',input_variables=["query","context"])
query="what is stack datatype?"
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=os.getenv("GOOGLE_API_KEY"))
ret=vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":6})



palchain=RunnableParallel({
    "context":ret,
    "query":RunnablePassthrough()
})
parser=StrOutputParser()
chain= RunnableLambda(lambda query: {
        "context": ret.invoke(query),  # retriever output
        "query": query                  # passthrough query
    })|prompt | model | parser
result=chain.invoke("implementations of stack in this book,explain?")
print(result)