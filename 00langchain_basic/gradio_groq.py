import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


prompt = ChatPromptTemplate.from_messages(
    [ ("system", "당신은 개발자입니다.") , 
     ("user", "{input}") ]
)


llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.groq.com/openai/v1",  # Groq API 엔드포인트
    model="meta-llama/llama-4-scout-17b-16e-instruct",  # Spring AI와 동일한 모델
    temperature=0.7
)

output_parser = StrOutputParser()


# LCEL 
chain = prompt | llm | output_parser


# user_input = input("질문을 입력하세요: ")

# response = chain.invoke({"input": user_input})

# print("Bot: ", response) 

import gradio as gr

def chat(user_input):
    return chain.invoke({"input": user_input})

demo = gr.Interface(fn=chat, inputs="text", outputs="text")
demo.launch()   
