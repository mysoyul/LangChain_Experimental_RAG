# 환경 변수에서 API 키 가져오기
import os
import warnings
import platform
from dotenv import load_dotenv

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 라이브러리 불러오기
import gradio as gr
from PIL import Image
import base64
from io import BytesIO
import traceback
import time
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# Agent 생성
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# 한글 폰트 설정 (단순화)
def set_korean_font():
    """한글 폰트 설정"""
    import matplotlib.font_manager as fm
    
    # 운영체제별 기본 한글 폰트
    font_map = {
        'Windows': 'Malgun Gothic',
        'Darwin': 'AppleGothic',    # macOS
        'Linux': 'DejaVu Sans'      # 한글 지원 안됨
    }
    
    system = platform.system()
    default_font = font_map.get(system, 'DejaVu Sans')
    
    # 설치된 폰트 목록에서 한글 폰트 찾기
    installed_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 한글 폰트 우선순위
    korean_fonts = [
        'Malgun Gothic',     # Windows
        'AppleGothic',       # macOS  
        'NanumGothic',       # 나눔고딕
        'Noto Sans CJK KR'   # 구글 폰트
    ]
    
    # 설치된 한글 폰트 중 첫 번째 사용
    for font in korean_fonts:
        if font in installed_fonts:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            print(f" 한글 폰트 설정: {font}")
            return font
    
    # 한글 폰트가 없으면 시스템 기본 폰트 사용
    plt.rcParams['font.family'] = default_font
    plt.rcParams['axes.unicode_minus'] = False
    
    if system == 'Linux':
        print(" 한글 폰트가 없습니다. Linux에서 설치하려면:")
        print("   !apt install fonts-nanum")
    
    print(f" 설정된 폰트: {default_font}")
    return default_font

# 한글 폰트 설정 실행
print(" 한글 폰트 설정...")
korean_font = set_korean_font()

# Seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")

# LLM 초기화
if OPENAI_API_KEY:
    llm = ChatOpenAI(
        model='gpt-3.5-turbo-0125', 
        temperature=0,
        api_key=OPENAI_API_KEY
    )
else:
    llm = None

def create_enhanced_agent(df):
    """향상된 pandas 에이전트 생성"""
    if not llm:
        return None
    
    # 한글 폰트 설정 코드
    font_setup_code = f"""
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = '{korean_font}'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
"""
    
    agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="openai-tools",
        verbose=True,
        return_intermediate_steps=True,
        allow_dangerous_code=True,
        prefix=f"""당신은 전문 데이터 분석가입니다. 데이터프레임의 이름은 'df'입니다.
        
시각화를 요청 받으면 반드시 다음 폰트 설정 코드를 먼저 실행하세요:
{font_setup_code}

분석 시 다음 사항을 준수해주세요:
1. 모든 답변은 한국어로 작성
2. 그래프 제목, 축 라벨, 범례는 반드시 한국어 사용
3. 통계적 수치는 소수점 2자리까지 표시
4. 시각화 시 적절한 색상과 스타일 사용
5. 데이터의 특성을 고려한 인사이트 제공

예시:
plt.title('데이터 분포 현황', fontsize=14, fontweight='bold')
plt.xlabel('변수명')
plt.ylabel('빈도수')
plt.legend(['범례1', '범례2'])
"""
    )
    
    return agent_executor

def get_data_summary(df):
    """데이터프레임 요약 정보 생성"""
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    return summary

def format_data_info(df):
    """데이터 정보를 포맷팅하여 표시"""
    info_text = f"""
##  데이터 기본 정보

** 데이터 크기:** {df.shape[0]:,}행 × {df.shape[1]:,}열
** 메모리 사용량:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
** 설정된 한글 폰트:** {korean_font}

### 컬럼 정보
"""
    
    # 컬럼별 정보
    for i, (col, dtype) in enumerate(df.dtypes.items()):
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        info_text += f"**{i+1}. {col}** (`{dtype}`) - 결측값: {missing_count}개 ({missing_pct:.1f}%)\n"
    
    # 수치형 데이터 요약 통계
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        info_text += "\n###  수치형 변수 요약 통계\n"
        summary_stats = df[numeric_cols].describe()
        info_text += summary_stats.to_string()
    
    return info_text

def analyze_with_langchain_agent(df, question):
    """LangChain 에이전트를 사용한 데이터 분석"""
    
    if not llm:
        return " OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.", None, None
    
    try:
        # 에이전트 생성
        agent_executor = create_enhanced_agent(df)
        
        if not agent_executor:
            return " 에이전트 생성에 실패했습니다.", None, None
        
        # 질문 분석 및 응답 생성
        start_time = time.time()
        response = agent_executor.invoke(question)
        end_time = time.time()
        
        text_output = response['output']
        execution_time = end_time - start_time
        
        # 실행된 Python 코드 추출
        intermediate_output = []
        
        try:
            for item in response['intermediate_steps']:
                if hasattr(item[0], 'tool') and item[0].tool == 'python_repl_ast':
                    code = str(item[0].tool_input['query'])
                    intermediate_output.append(code)
        except Exception as e:
            print(f"코드 추출 중 오류: {e}")
        
        python_code = "\n".join(intermediate_output)
        
        # 시각화 코드가 있는지 확인
        visualization_keywords = ["plt", "fig", "plot", "sns.", "seaborn", "matplotlib"]
        has_visualization = any(keyword in python_code for keyword in visualization_keywords)
        
        if not has_visualization:
            python_code = None
        
        # 응답에 실행 시간 추가
        text_output += f"\n\n **분석 완료 시간:** {execution_time:.2f}초"
        
        return text_output, python_code, execution_time
        
    except Exception as e:
        error_msg = f" 분석 중 오류가 발생했습니다:\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, None, None

def execute_and_show_chart(python_code, df):
    """Python 코드 실행 및 차트 생성"""
    
    if not python_code:
        return None
    
    try:
        # 실행 환경 준비
        exec_globals = {
            'df': df.copy(),
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns
        }
        
        # 한글 폰트 설정 (단순화)
        font_setup = f"""
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = '{korean_font}'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
"""
        
        # 폰트 설정 + 사용자 코드 실행
        full_code = font_setup + "\n" + python_code
        
        exec(full_code, exec_globals)
        
        # 차트를 이미지로 변환
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close('all')  # 모든 figure 닫기
        
        return img
    
    except Exception as e:
        print(f"차트 생성 중 오류: {e}")
        print(f"실행된 코드:\n{python_code}")
        plt.close('all')
        return None

def process_and_display(csv_file, question, progress=gr.Progress()):
    """CSV 파일 처리 및 분석 수행"""
    
    if not csv_file:
        return " CSV 파일을 업로드해주세요.", None, ""
    
    if not question.strip():
        return " 질문을 입력해주세요.", None, ""
    
    try:
        progress(0.1, desc="CSV 파일 읽는 중...")
        
        # CSV 파일을 데이터프레임으로 읽기
        df = pd.read_csv(csv_file, encoding='utf-8')
        
        progress(0.3, desc="데이터 정보 분석 중...")
        
        # 데이터 정보 생성
        data_info = format_data_info(df)
        
        progress(0.5, desc="AI 에이전트 분석 중...")
        
        # 질문에 대한 답변 생성
        text_output, python_code, exec_time = analyze_with_langchain_agent(df, question)
        
        progress(0.8, desc="차트 생성 중...")
        
        # 차트 생성
        chart_image = execute_and_show_chart(python_code, df) if python_code else None
        
        progress(1.0, desc="완료!")
        
        # 최종 결과 포맷팅
        final_output = f"""
{data_info}

##  AI 분석 결과

** 질문:** {question}

** 답변:**
{text_output}
"""
        
        if python_code:
            final_output += f"""

##  실행된 Python 코드

```python
{python_code}
```
"""        
        return final_output, chart_image, ""
        
    except UnicodeDecodeError:
        try:
            # UTF-8로 읽기 실패 시 다른 인코딩 시도
            df = pd.read_csv(csv_file, encoding='cp949')
            data_info = format_data_info(df)
            text_output, python_code, exec_time = analyze_with_langchain_agent(df, question)
            chart_image = execute_and_show_chart(python_code, df) if python_code else None
            
            final_output = f"""
{data_info}

##  AI 분석 결과

** 질문:** {question}

** 답변:**
{text_output}
"""
            return final_output, chart_image, ""
            
        except Exception as e:
            return f" 파일 읽기 오류: 인코딩 문제입니다. UTF-8 또는 CP949 인코딩으로 저장된 CSV 파일을 사용해주세요.\n상세 오류: {str(e)}", None, ""
    
    except Exception as e:
        error_msg = f" 처리 중 오류가 발생했습니다:\n{str(e)}\n\n{traceback.format_exc()}"
        return error_msg, None, ""

def load_sample_data():
    """샘플 데이터 생성"""
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        '이름': [f'고객_{i}' for i in range(1, 101)],
        '나이': np.random.randint(20, 70, 100),
        '성별': np.random.choice(['남성', '여성'], 100),
        '소득': np.random.normal(5000, 1500, 100).astype(int),
        '구매금액': np.random.normal(300, 100, 100).astype(int),
        '만족도': np.random.randint(1, 6, 100)
    })
    
    return sample_data

# 샘플 질문들
SAMPLE_QUESTIONS = [
    "데이터의 기본 통계 정보를 알려주세요",
    "결측값이 있는 컬럼을 찾아주세요",
    "수치형 변수들의 상관관계를 히트맵으로 보여주세요",
    "범주형 변수의 분포를 막대그래프로 그려주세요",
    "이상치를 탐지하고 박스플롯으로 시각화해주세요",
    "주요 변수들의 분포를 히스토그램으로 보여주세요"
]

# Gradio 인터페이스 구성
with gr.Blocks(
    title=" AI 데이터 분석 도구",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .main-header {
        text-align: center;
        margin-bottom: 30px;
    }
    """
) as demo:
    
    # 헤더
    gr.HTML("""
    <div class="main-header">
        <h1> AI 데이터 분석 도구</h1>
        <p>CSV 파일을 업로드하고 자연어로 질문하면 AI가 데이터를 분석해드립니다!</p>
    </div>
    """)
    
    # API 키 상태 표시
    api_status = " API 키 설정됨" if OPENAI_API_KEY else " API 키 미설정"
    font_status = f" 한글 폰트: {korean_font}"
    
    gr.HTML(f"""
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <strong>시스템 상태:</strong> {api_status} | {font_status}
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # 파일 업로드
            csv_input = gr.File(
                label=" CSV 파일 업로드",
                file_types=[".csv"],
                type="filepath"
            )
            
            # 질문 입력
            question_input = gr.Textbox(
                label=" 분석 질문",
                placeholder="데이터에 대해 궁금한 것을 자연어로 물어보세요...",
                lines=3
            )
            
            # 샘플 질문 선택
            sample_question = gr.Dropdown(
                label=" 샘플 질문",
                choices=SAMPLE_QUESTIONS,
                value=None
            )
            
            # 버튼들
            with gr.Row():
                submit_button = gr.Button(" 분석 시작", variant="primary")
                clear_button = gr.Button(" 초기화")
            
            # 샘플 데이터 다운로드
            gr.HTML("""
            <div style="margin-top: 20px; padding: 15px; background-color: #e7f3ff; border-radius: 5px;">
                <h4> 샘플 데이터</h4>
                <p>테스트용 샘플 데이터를 다운로드하여 사용해보세요!</p>
            </div>
            """)
            
            sample_download = gr.File(
                label="샘플 데이터 다운로드",
                value=None,
                visible=False
            )
            
            sample_button = gr.Button(" 샘플 데이터 생성")
        
        with gr.Column(scale=2):
            # 출력 영역
            output_markdown = gr.Markdown(label=" 분석 결과")
            output_image = gr.Image(label=" 생성된 차트", type="pil")
    
    # 사용법 안내
    with gr.Accordion("📖 사용법 안내", open=False):
        gr.Markdown("""
        ##  사용 방법
        
        1. **CSV 파일 업로드**: 분석하고 싶은 CSV 파일을 업로드하세요
        2. **질문 입력**: 데이터에 대해 자연어로 질문을 입력하세요
        3. **분석 시작**: '분석 시작' 버튼을 클릭하세요
        4. **결과 확인**: AI가 분석한 결과와 차트를 확인하세요
        
        ##  질문 예시
        
        - "나이와 소득의 상관관계를 알려주세요"
        - "성별에 따른 구매금액 차이를 시각화해주세요"
        - "만족도가 높은 고객들의 특징을 분석해주세요"
        - "이상치를 찾아서 보여주세요"
        
        ##  주의사항
        
        - CSV 파일은 UTF-8 또는 CP949 인코딩으로 저장하세요
        - 한글 컬럼명과 데이터를 지원합니다
        - 대용량 파일은 처리 시간이 오래 걸릴 수 있습니다
        """)
    
    # 이벤트 핸들러
    def update_question(selected_question):
        return selected_question if selected_question else ""
    
    def clear_all():
        return None, "", None, None, None
    
    def generate_sample():
        sample_df = load_sample_data()
        sample_path = "sample_data.csv"
        sample_df.to_csv(sample_path, index=False, encoding='utf-8-sig')
        return sample_path
    
    # 이벤트 연결
    sample_question.change(
        fn=update_question,
        inputs=[sample_question],
        outputs=[question_input]
    )
    
    submit_button.click(
        fn=process_and_display,
        inputs=[csv_input, question_input],
        outputs=[output_markdown, output_image, sample_question]
    )
    
    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[csv_input, question_input, sample_question, output_markdown, output_image]
    )
    
    sample_button.click(
        fn=generate_sample,
        inputs=[],
        outputs=[sample_download]
    )

# 앱 실행
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )