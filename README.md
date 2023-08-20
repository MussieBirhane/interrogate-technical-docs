## Interrogate-technical-docs

Large Language Models (LLMs) based app to interact and interrogate with technical documentations.

This app is designed to harness the power of Large Language Models (LLMs) in a seamless manner.
The app assists the user in the intricate process of interrogating and extracting insights from
technical documentations.

The app seamlessly integrates three core components:
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/docs/get_started/introduction.html)
- [OpenAI LLM models](https://openai.com/)

## How it works:

Simply upload your PDF technical documents and provide specific queries related to your documentation.
The LLM-powered chatbot will then sift through the information, extract pertinent details, and provide
you with insightful responses.

## To launch:

1. Create a new Python environment
2. Install all dependencies from requirements.txt
```
pip install requirements.txt
```
3. Activate the created environment
4. Create an .env file with your OPENAI-API-KEY inside and save it on the same directory
5. Open a terminal in your code interpreter and launch the app
```
streamlit run app-chat.py
```
