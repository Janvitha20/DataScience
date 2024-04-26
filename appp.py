import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

# Set title and input field
st.title("🤖 RAG Q&A System: On Leave No Context Behind Paper 📄")
user_input = st.text_input("Enter text ....")

# Initialize ChatGoogleGenerativeAI
chat_model = ChatGoogleGenerativeAI(google_api_key="YOUR_API_KEY", model="gemini-1.5-pro-latest")

# Define chat template
chat_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="You are a Helpful AI Bot. You take the question from user and answer if you have the specific information related to the question."),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("Answer the following question: {question}\nAnswer: ")
])

# Initialize output parser
output_parser = StrOutputParser()

# Load document
loader = PyPDFLoader(r"D:\Users\DELL\Desktop\Innomatics DS\RAG2\data\Leave_no_context_behind.pdf")
data = loader.load()

# Split documents into chunks
text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# Create embedding model
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key="AIzaSyDvU3SwP_TMwEv_pfyy9JIqE_BMm4Y5O0Q", model="models/embedding-001")

# Store chunks in vector store
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
db.persist()

# Connect with ChromaDB
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)

# Convert CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Format documents function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define chat template for responses
response_template = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="You are a Helpful AI Bot. You take the context and question from user. Your answer should be based on the specific context."),
    # Human Message Prompt Template
    HumanMessagePromptTemplate.from_template("Answer the question based on the given context.\nContext: {context}\nQuestion: {question}\nAnswer: ")
])

# Define rag_chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | response_template
    | chat_model
    | output_parser
)

# Generate response
if st.button("Generate"):
    response = rag_chain.invoke(user_input)
    st.write(response)
