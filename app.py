import streamlit as st
import tempfile
import time
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIARerank, ChatNVIDIA


def validate(api_key:str)->bool:
    try:
        client = ChatNVIDIA(
            model="mistralai/mixtral-8x7b-instruct-v0.1",
            nvidia_api_key=api_key,
        )
        # Simple test to check connection
        client.invoke("Test connection")
        return True
    except Exception as e:
        print(str(e))
        return False
    
def create_prompt()->ChatPromptTemplate:
    prompt_message = """
    Using the following context, Answer the query in no more than 75 words
    <context>
    context : {context}
    <context>
    query : {input}
    Provide a detailed and precise answer based on the context
    """
    return ChatPromptTemplate.from_template(
        prompt_message
        )

def upload_pdf(file_input):
    try:
        # Create a temporary file with explicit permissions
        with tempfile.NamedTemporaryFile(mode='wb+', delete=False, suffix='.pdf') as temp_pdf:
            # Write the uploaded file content to the temporary file
            temp_pdf.write(file_input.getvalue())
            temp_pdf.flush()  # Ensure all data is written
            temp_file_path = temp_pdf.name  # Store the path
                    
        # Load PDF using the temporary file path
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        return docs
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
    finally:
        # Clean up: Remove the temporary file
        try:
            import os
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except Exception as e:
            print(f"Error cleaning up temporary file: {str(e)}")

def generate_response(retrieval_chain,query):
    start = time.process_time()
    ## generating the response
    response = retrieval_chain.invoke({"input":query})
    st.write("processing time:",time.process_time()-start)
    st.write(response["answer"])
    st.markdown("---")
    # displaying the context information
    with st.expander("CONTEXT"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content) 
            st.markdown('---')

st.set_page_config(page_title = "RAG-Q-and-A BOT", page_icon = "🤖")
st.title("ASKB🤖T")
with st.expander("ABOUT"):
    st.write("upload a PDF and Enter the query 😊")
    st.write("The ASKBoT will answer the query based on the context provided within the document😁👍")
nvidia_api_key = st.sidebar.text_input("ENTER YOUR NVIDIA API KEY",type = "password",placeholder = "nvapi-x-x-x")
st.sidebar.markdown("------")
url = "https://build.nvidia.com/meta/llama-3_1-70b-instruct?snippet_tab=LangChain"
st.sidebar.write("Note:")
st.sidebar.write("1. To obtain the NVIDIA API KEY, Follow this link: [LINK](%s)"%url)
st.sidebar.write("2. Login and generate your own API key.")
st.sidebar.markdown("------")
if nvidia_api_key:
    if validate(nvidia_api_key):
        st.sidebar.success("Validated API Key successfully")
        # defining the LLM 
        llm = ChatNVIDIA(nvidia_api_key=nvidia_api_key,
                         model = "meta/llama3-70b-instruct"
                         )
        # Defining the NVIDIA embeddings
        embeddings = NVIDIAEmbeddings(nvidia_api_key = nvidia_api_key)
        ## File Uploader

        uploaded_file = st.file_uploader("Choose a PDF file", type = "pdf", accept_multiple_files = False)
        if uploaded_file:
            docs = upload_pdf(uploaded_file)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 750
            )
            docs = text_splitter.split_documents(docs)
            if "vectors" not in st.session_state:
                with st.spinner("uploading the PDF"):
                    st.session_state.vectors = FAISS.from_documents(
                        documents = docs,
                        embedding = embeddings
                    )
            query = st.text_input("Enter the Query",placeholder = "Your query here...")
            prompt_template = create_prompt()
            button = st.button("ASK")
            if button:
                if query:
                    document_chain = create_stuff_documents_chain(llm,
                                                                prompt_template)
                    retriever = st.session_state.vectors.as_retriever()
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
                    response = generate_response(retrieval_chain, query)
                else:
                    st.error("Please Enter a Query")

    else:
        st.sidebar.error("INVALID API KEY")
else:
    st.error("Please Enter an API Key")