import streamlit as st
import duckdb
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain,RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv  # 환경 변수 로드를 위한 모듈

load_dotenv()

# 🔹 OpenAI API 설정
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]  # 환경변수에서 가져오기

# 🔹 DuckDB 초기화
db = duckdb.connect(":memory:")
db.execute("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER);")
db.execute("INSERT INTO users VALUES (1, 'Alice', 25), (2, 'Bob', 30), (3, 'Charlie', 35);")

# 🔹 DuckDB 데이터를 RAG 방식으로 활용할 수 있도록 Vector DB에 저장
def load_duckdb_data():
    df = db.execute("SELECT * FROM users").fetchdf()
    data_texts = [f"ID: {row['id']}, Name: {row['name']}, Age: {row['age']}" for _, row in df.iterrows()]
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(data_texts, embeddings)
    return vector_store

vector_db = load_duckdb_data()
retriever = vector_db.as_retriever()

# 🔹 LangChain 기반 AI 모델 설정

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4",  # GPT-3.5 모델 사용
    temperature=0.1,  # 낮은 temperature로 일관된 출력 생성
    max_tokens=1000,  # 충분한 길이의 요약을 위한 토큰 수 설정
    streaming = True
)


template = """""Please use the following context to answer the question at the end. Return the SQL query statement that fits the MySQL grammar for SQL search at the end of the answer.\
    1. If you don't know the answer, don't tell me you don't know and try to make\
    2. Use up to three sentences. Keep your answers as concise as possible.\
    3. Please answer in Korean and keep your SQL query grammar in English. Be sure to follow the SQL grammar. 
    4. Keep the capital letter, column name of the context in your SQL query grammar\
    5. Always "\n"\n스마트리온 Partner, thank you for asking!!👍" \n"\
    
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=['context', 'question'],
                                 template= template)
question_generator = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)  # 🚫 필수 필드 설정

memory = ConversationBufferMemory(
    memory_key='chat_history',
    return_messages=True
)
# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever = retriever,
#     return_source_documents = True,
#     chain_type_kwargs={'prompt': QA_CHAIN_PROMPT}
    
# )
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory = memory,
    combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT}
)
# 🔹 Streamlit 페이지 설정
st.set_page_config(layout="wide")

### 🔹 UI 상태 변수
if "explorer_visible" not in st.session_state:
    st.session_state.explorer_visible = True
if "ai_chat_visible" not in st.session_state:
    st.session_state.ai_chat_visible = True
# 🔹 세션 상태 변수 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

### 🔹 버튼 UI 추가
col_title, col_buttons = st.columns([3, 1])  

with col_title:
    st.markdown("<h1 style='text-align: center;'>AI스맛리온</h1>", unsafe_allow_html=True)

with col_buttons:
    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
    if st.button("📂 탐색기 열기 / 닫기"):
        st.session_state.explorer_visible = not st.session_state.explorer_visible  # 버튼 토글
    if st.button("🤖 AI스맛리온"):
        st.session_state.ai_chat_visible = not st.session_state.ai_chat_visible  # 버튼 토글
    st.markdown("</div>", unsafe_allow_html=True)

# 🔹 중앙 SQL 실행 레이아웃 (좌측: 탐색기 / 중앙: SQL 실행 + 차트 / 우측: AI 대화창)
col1, col2, col3 = st.columns([
    1 if st.session_state.explorer_visible else 0.1,  # 탐색기 크기 조절
    3 if not (st.session_state.explorer_visible or st.session_state.ai_chat_visible) else 2,  # SQL 실행 + 차트
    1 if st.session_state.ai_chat_visible else 0.1   # AI 대화창 크기 조절
])

### 🔹 좌측: 탐색기 (DuckDB 테이블 조회)
if st.session_state.explorer_visible:
    with col1:
        st.header("📂 데이터 탐색기")
        tables = db.execute("SHOW TABLES").fetchall()
        selected_table = st.selectbox("테이블 선택", [t[0] for t in tables])
        if st.button("테이블 조회"):
            result = db.execute(f"SELECT * FROM {selected_table}").fetchdf()
            st.dataframe(result)

### 🔹 중앙: SQL 실행 + 시각화
with col2:
    st.header("📝 SQL 실행")
    user_sql = st.text_area("SQL 문 입력", "SELECT * FROM users;")
    if st.button("실행"):
        try:
            result = db.execute(user_sql).fetchdf()
            st.dataframe(result)

            # 🔹 자동 차트 생성 로직 추가
            if len(result.columns) > 1:
                st.subheader("📊 데이터 시각화")
                numeric_columns = result.select_dtypes(include=['int', 'float']).columns.tolist()

                if len(numeric_columns) >= 2:
                    st.line_chart(result.set_index(numeric_columns[0])[numeric_columns[1]])
                    st.bar_chart(result.set_index(numeric_columns[0])[numeric_columns[1]])
                    st.scatter_chart(result.set_index(numeric_columns[0])[numeric_columns[1]])
                else:
                    st.warning("📌 데이터 시각화를 위해 최소 두 개의 수치형 컬럼이 필요합니다.")

        except Exception as e:
            st.error(f"SQL 실행 오류: {str(e)}")

# 🔹 대화 내역 출력
if st.session_state.ai_chat_visible:
    with col3:
        st.header("💬 AI 스맛리온")

        # 🔹 기존 대화 기록 출력
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.markdown(chat["message"])

        # 🔹 입력창을 브라우저 가장 아래에 배치
        user_query = st.text_input("질문을 입력하세요...", key="user_query", placeholder="30살 사람 찾아줘")

# 🔹 질문이 입력되었을 때 실행
if user_query:
    with col3:
        with st.chat_message("user"):
            st.markdown(f"**🙋‍♂️ 질문:** {user_query}")

        # 🔹 스트리밍 방식으로 응답 출력
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_text = ""

            for chunk in qa_chain.stream({"question": user_query, "chat_history": st.session_state.chat_history}):
                response_text += chunk["answer"]
                response_placeholder.markdown(response_text + "▌")

            response_placeholder.markdown(response_text)

        # 🔹 대화 기록 저장
        st.session_state.chat_history.append({"role": "user", "message": user_query})
        st.session_state.chat_history.append({"role": "assistant", "message": response_text})

