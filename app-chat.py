# Authors: Aleksei Kondratenko (PhD Researcher and Data Scientist)
#          Mussie Birhane (Structural Engineer)
#
# Interrogate with your technical documentation using free and paid LLMs
# Medium article: https://pub.aimind.so/interrogate-your-technical-documentation-using-free-and-paid-llms-f2a7664ff2bd

import streamlit as st
#st.set_page_config(layout="wide")

import os
import openai
import pickle
#import huggingface_hub
#from streamlit_extras.add_vertical_space import

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
    st.set_page_config(page_title="Interrogate with your technical documents")
    st.title('Interrogate with your technical documents')
    st.markdown('''
    This app is an Large Language Models (LLMs) powered chatbot built to interogate
    your technical documents using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI LLM models](https://platform.openai.com/docs/models)
    ''')
    st.write(' ')
    st.write(' ')

    #add_vertical_space(6)
    st.write('Developed by [Aleksei Kondratenko](https://www.linkedin.com/in/aleksei-kondratenko-14a2a0192/) & [Mussie Birhane](https://www.linkedin.com/in/mussie-birhane-92b0ba156/)')
    st.write(' ')
    st.write(' ')
    st.write('''
    ### How it works

    Simply upload your PDF technical documents and provide specific queries related to your documents.
    The LLM-powered chatbot will then sift through the information, extract pertinent details, and provide
    you with insightful responses.
    ''')
# get your OPENAI_API_KEY
_ = load_dotenv(find_dotenv())                  # read local .env file
openai.api_key  = os.getenv('OPENAI_API_KEY')

def main():
    st.header('Interrogate with your technical documents')
    # Upload a PDF file
    pdf = st.file_uploader('Upload your PDF document and ask any questions', type='pdf')

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
        #st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                database = pickle.load(f)
            #st.write('Embeddings Loaded from the disk')

        else:
            embeddings = OpenAIEmbeddings()
            database = FAISS.from_texts(text_chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(database, f)
            #st.write('Embedding completed!')

        # Prompt engineering
        template_string = """Answer to the question in a style that is {style}.
        Please answer step by step. Question:{text}
        """

        prompt_template = ChatPromptTemplate.from_template(template_string)
        style = """You are an experienced structural engineer politely teaching a rookie.
        """
        ####################################################################

        #messages = [
        #{"role": "user", "content": "User prompt"},
        #{"role": "assistant", "content": "The response"}
        #]

        # Initialize chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on the app re-run
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if query := st.chat_input("Send a message"):
            # Display user message in chat message container
            with st.chat_message(name="user"):
                st.markdown(query)
            # Add user message to chat History
            st.session_state.messages.append({"role": "user", "content": query})

            # Develop the response
            with st.spinner('Retrieving...'):
                final_prompt = prompt_template.format_messages(
                                    style=style,
                                    text=query)

                # Extract the query from the final template
                query = final_prompt[0].content

                #st.write(query)
                #https://python.langchain.com/docs/use_cases/question_answering/how_to/question_answering
                #https://api.python.langchain.com/en/latest/llms/langchain.llms.openai.OpenAI.html
                chain = load_qa_chain(OpenAI(), chain_type="stuff")
                docs = database.similarity_search(query)

                with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=query)
                        print(cb)
                #st.info(response)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    main()
