import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from langchain.prompts import PromptTemplate


# Set up the Google Gemini API
os.environ['GOOGLE_API_KEY'] = '#################################################'
load_dotenv()
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")
        print(f"Directory has been deleted.")
    vector_store.save_local("faiss_index")

# Function to set up conversational chain
def get_conversational_chain():
    
    prompt_template = """
You are an intelligent, highly adaptable assistant capable of understanding and solving a wide variety of tasks. Your role is to assist the user with any type of question or request, whether it involves answering complex queries, performing comparisons, distinguishing between concepts, transforming content (e.g., summarization, rephrasing, translation), or performing calculations and providing insightful recommendations.

When asked to compare or distinguish between different items, provide a clear and concise analysis, highlighting the key similarities and differences. For any form of task or question, ensure that your response is thorough, contextually appropriate, and meets the specific needs of the user.

If the task involves content transformation, modify the content as per the user's instructions, making sure it is tailored to their requirements. If the question requires reasoning, comparison, or logical analysis, approach the problem methodically, providing detailed and reasoned explanations.

In cases where the output involves visual elements, such as images or graphs intended for inclusion in a PowerPoint presentation, provide relevant visual content that effectively communicates the key points. Ensure that any generated images or graphs are clear, informative, and appropriately labeled, ready for presentation.

If the necessary information or context for a task is unavailable, state clearly, "The requested information or task cannot be completed based on the provided context."

Be flexible and adaptive in your approach. Make sure that all answers are actionable, relevant, and easy to understand, regardless of the complexity of the task.

\n\nContext:\n{context}\n\nQuestion/Task:\n{question}\n\nResponse:
"""

    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Function to handle different output types (text, table, image, etc.)
def display_output(response_text):
    if isinstance(response_text, str):
        st.write(response_text)
    elif isinstance(response_text, dict):  # Example of tabular data
        df = pd.DataFrame(response_text)
        st.table(df)
    elif isinstance(response_text, list) and all(isinstance(item, dict) for item in response_text):
        df = pd.DataFrame(response_text)
        st.table(df)
    elif "graph" in response_text.lower():
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2, 3], [10, 20, 25, 30])
        st.pyplot(fig)
    elif "image" in response_text.lower():  # Placeholder for image processing
        st.image("path_to_image.jpg")
    else:
        st.write("Unsupported format or content")

# Function to handle user input and generate response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    # Here you determine what kind of output to display
    display_output(response["output_text"])

# Main Streamlit app
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
