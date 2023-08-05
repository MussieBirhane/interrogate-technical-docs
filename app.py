# Authors: Aleksei Kondratenko (PhD Researcher and Data Scientist)
#          Mussie Birhane (Structural Engineer)
#
# Medium article: https://pub.aimind.so/interrogate-your-technical-documentation-using-free-and-paid-llms-f2a7664ff2bd

# Interrogate with your technical documentation using free and paid LLMs

import os
import openai
import pickle
#import huggingface_hub
import streamlit as st

from dotenv import load_dotenv, find_dotenv

from langchain import PromptTemplate # used for LLMs
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
#from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate # Used for chat models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader

# Sidebar contents
with st.sidebar:
    st.title('Interrogate with your technical documents')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built to interogate your technical
    documents using:

    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')

    st.write('[Aleksei Kondratenko](https://www.linkedin.com/in/aleksei-kondratenko-14a2a0192/)')
    st.write('[Mussie Birhane](https://www.linkedin.com/in/mussie-birhane-92b0ba156/)')

# get your OPENAI_API_KEY
_ = load_dotenv(find_dotenv())                  # read local .env file
openai.api_key  = os.getenv('OPENAI_API_KEY')

def main():
    st.header('Interrogate with your technical documents')
    # Upload a PDF file
    pdf = st.file_uploader('Upload your PDF document', type='pdf')

    #st.write(type(pdf))

    if pdf is not None:
        #st.write(pdf)

        # extract the text of the PDF document
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # define text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""],
            length_function=len
            )
        # split the text into chunks
        text_chunks = text_splitter.split_text(text=text)

        # display text chunks
        #st.write(text_chunks)

        # Embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                database = pickle.load(f)
            st.write('Embeddings Loaded from the disk')

        else:
            embeddings = OpenAIEmbeddings()
            database = FAISS.from_texts(text_chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(database, f)
            st.write('Embedding completed!')

        # Prompt engineering

        template_string = """Answer to the question in a style that is {style}. Answer step by step.
        Question:{text}
        """

        prompt_template = ChatPromptTemplate.from_template(template_string)
        style = """You are an experienced structural engineer politely teaching a rookie.
        """

        # Accept user questions
        query = st.text_input('Enter your question here: ', placeholder='Please provide a short summary', disabled=not pdf)

        if query:
            with st.spinner('Calculating...'):
                final_prompt = prompt_template.format_messages(
                                    style=style,
                                    text=query)

                # Extract the query from the final template
                query = final_prompt[0].content

                #st.write(query)

                chain = load_qa_chain(OpenAI(), chain_type="stuff")
                docs = database.similarity_search(query)

                with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=query)
                        print(cb)

                st.info(response)

if __name__ == '__main__':
    main()
