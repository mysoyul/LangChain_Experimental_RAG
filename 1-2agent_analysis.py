# 환경 변수에서 API 키 가져오기
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# 라이브러리 불러오기
import gradio as gr
from PIL import Image
import base64
from io import BytesIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Agent 생성
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0,
                       api_key=OPENAI_API_KEY)


def analyze_with_langchain_agent(df, question):
    agent_executor = create_pandas_dataframe_agent(
    llm,
    df,
    agent_type="openai-tools",
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_code=True
    )
    
    response = agent_executor.invoke(question)

    text_output = response['output']


    intermediate_output = []

    try:
        for item in response['intermediate_steps']:

            if item[0].tool == 'python_repl_ast':
                intermediate_output.append(str(item[0].tool_input['query']))

    except:
        pass


    python_code = "\n".join(intermediate_output)
    

    try:
        exec(python_code)
        if ("plt" not in python_code) & ("fig" not in python_code) & ("plot" not in python_code) & ("sns." not in python_code) :
            python_code = None
    except:
        python_code = None

    return text_output, python_code

def execute_and_show_chart(python_code, df):

    try:
        # 코드 실행 환경 준비 및 코드 실행
        locals = {"df": df.copy()}
        exec(python_code, globals(), locals)

        # 차트 이미지로 변환
        fig = plt.figure()
        exec(python_code, globals(), locals)

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img
    
    except Exception as e:
        # 예외 발생 시 적절한 에러 메시지 반환
        print(f"Error executing chart code: {e}")
        return None


def process_and_display(csv_file, question):

    # CSV 파일을 데이터프레임으로 
    
    df = pd.read_csv(csv_file)

    # 질문에 대한 답변 생성
    text_output, python_code = analyze_with_langchain_agent(df, question)

    # 결과를 출력
    chart_image = execute_and_show_chart(python_code, df) if python_code else None

    return text_output, chart_image

with gr.Blocks() as demo:
    gr.Markdown("### CSV 파일을 업로드하고, 질문을 입력하세요. 분석 결과를 확인할 수 있습니다.")
    with gr.Row():
        csv_input = gr.File(label="CSV 파일 업로드", type="filepath")
        question_input = gr.Textbox(placeholder="질문을 입력하세요.")
        submit_button = gr.Button("Run")

    output_markdown = gr.Markdown()
    output_image = gr.Image()
    
    submit_button.click(fn=process_and_display, 
                        inputs=[csv_input, question_input], 
                        outputs=[output_markdown, output_image])

demo.launch()







